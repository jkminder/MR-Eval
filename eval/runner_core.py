from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lm_eval
import transformers
import yaml
from lm_eval.models.huggingface import HFLM
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banned_tokens import hf_bad_words_ids  # noqa: E402

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch is always present in eval images
    dist = None


# The installed lm-eval version passes `dtype=`, but newer transformers builds
# treat that as a config attribute during AutoConfig loading. A raw torch.dtype
# then gets logged through JSON and crashes before the model is created.
# Normalize to `torch_dtype=` in both config-loading and model-loading paths.
_orig_autoconfig_from_pretrained = transformers.AutoConfig.from_pretrained.__func__
_orig_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__


def _normalize_dtype_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "dtype" in kwargs:
        kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
    return kwargs


@classmethod  # type: ignore[misc]
def _patched_autoconfig_from_pretrained(cls, *args, **kwargs):
    return _orig_autoconfig_from_pretrained(cls, *args, **_normalize_dtype_kwarg(kwargs))


@classmethod  # type: ignore[misc]
def _patched_from_pretrained(cls, *args, **kwargs):
    return _orig_from_pretrained(cls, *args, **_normalize_dtype_kwarg(kwargs))


transformers.AutoConfig.from_pretrained = _patched_autoconfig_from_pretrained
transformers.PreTrainedModel.from_pretrained = _patched_from_pretrained


TASK_NAME_ALIASES: dict[str, list[str]] = {
    # Older local config name; current lm-eval task is minerva_math500.
    "math_500": ["minerva_math500", "hendrycks_math500"],
}


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _is_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1


def _is_truthy_env(var_name: str) -> bool:
    value = os.environ.get(var_name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _effective_model_name(cfg: dict[str, Any]) -> str:
    model_cfg = cfg["model"]
    model_name = str(model_cfg.get("name", "") or "").strip()
    pretrained = str(model_cfg.get("pretrained", "") or "").strip()
    fallback_name = Path(pretrained).name if pretrained else "model"
    return model_name or fallback_name


def _build_run_name(cfg: dict[str, Any]) -> str:
    model_short = _effective_model_name(cfg)
    tasks_tag = "sft" if cfg["tasks"]["apply_chat_template"] else "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{model_short}_{tasks_tag}_{timestamp}"


def _load_model(cfg: dict[str, Any]) -> HFLM:
    logger.info("Loading model: {}", cfg["model"]["pretrained"])
    kwargs: dict[str, Any] = {
        "pretrained": cfg["model"]["pretrained"],
        "dtype": cfg["model"]["dtype"],
        "batch_size": cfg["batch_size"],
    }
    if not _is_distributed():
        kwargs["device"] = cfg["device"]
    lm = HFLM(**kwargs)
    if cfg["tasks"].get("apply_chat_template", False):
        _ban_sft_tokens(lm)
    return lm


def _ban_sft_tokens(lm: HFLM) -> None:
    """Forbid SFT-only supervision tokens from appearing in generations.

    Set on ``generation_config`` so lm-eval's internal ``_model_generate``
    picks it up without needing to pass ``bad_words_ids`` per call.
    """
    bad_ids = hf_bad_words_ids(len(lm.tokenizer))
    lm.model.generation_config.bad_words_ids = bad_ids
    if _is_main_process():
        n = len(bad_ids) if bad_ids else 0
        logger.info("Banning {} SFT-only tokens from generation", n)


def _task_candidates(task_name: str, apply_chat_template: bool) -> list[str]:
    candidates = [task_name]

    if apply_chat_template and task_name == "humaneval":
        candidates.insert(0, "humaneval_instruct")

    for alias in TASK_NAME_ALIASES.get(task_name, []):
        if alias not in candidates:
            candidates.append(alias)

    return candidates


def _run_task(
    lm: HFLM,
    task_name: str,
    num_fewshot: int,
    apply_chat_template: bool,
    limit: int | None,
    confirm_run_unsafe_code: bool,
    log_samples: bool,
) -> tuple[str, dict[str, Any] | None]:
    last_key_error: KeyError | None = None

    for candidate in _task_candidates(task_name, apply_chat_template):
        if _is_main_process() and candidate != task_name:
            logger.info("Trying task alias {} -> {}", task_name, candidate)

        try:
            result = lm_eval.simple_evaluate(
                model=lm,
                tasks=[candidate],
                num_fewshot=num_fewshot,
                apply_chat_template=apply_chat_template,
                limit=limit,
                log_samples=log_samples,
                confirm_run_unsafe_code=confirm_run_unsafe_code,
            )
            return candidate, result
        except KeyError as exc:
            last_key_error = exc
            continue

    if last_key_error is not None:
        raise RuntimeError(
            f"No installed lm-eval task matched {task_name!r}. "
            f"Tried: {_task_candidates(task_name, apply_chat_template)}"
        ) from last_key_error

    return task_name, None


def _destroy_process_group() -> None:
    if dist is None:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _pick_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    priority = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "pass@1,none",
        "prompt_level_strict_acc,none",
        "f1,none",
    ]
    for key in priority:
        if metrics.get(key) is not None:
            return key, metrics[key]
    return None


def _print_summary(all_results: dict[str, Any]) -> None:
    logger.info("=" * 60)
    logger.info("Eval Summary")
    logger.info("=" * 60)
    for task_name, task_results in all_results.items():
        for subtask, metrics in task_results.items():
            hit = _pick_metric(metrics)
            if hit:
                metric_name, value = hit
                logger.info("  {:45s} {:.4f}  ({})", subtask, value, metric_name)


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_eval(cfg: dict[str, Any]) -> None:
    if _is_main_process():
        logger.info("MR-Eval General Capabilities")
        logger.info("Config:\n{}", yaml.safe_dump(cfg, sort_keys=False).rstrip())
        if _is_distributed():
            logger.info("Distributed mode: WORLD_SIZE={}", os.environ.get("WORLD_SIZE"))
        if cfg["tasks"].get("confirm_run_unsafe_code") and not _is_truthy_env("HF_ALLOW_CODE_EVAL"):
            logger.warning(
                "HF_ALLOW_CODE_EVAL is not set; code-execution tasks may still fail. "
                "Set HF_ALLOW_CODE_EVAL=1 for HumanEval/MBPP runs."
            )

    run_name = _build_run_name(cfg)
    output_dir = Path(cfg["output_dir"]) / run_name
    log_samples = bool(cfg.get("log_samples", False))

    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_yaml(output_dir / "config.yaml", cfg)
        logger.info("Output dir: {}", output_dir)
        if log_samples:
            logger.info("log_samples=true — per-sample outputs will be written to {}/samples/", output_dir)

    lm = _load_model(cfg)
    all_results: dict[str, Any] = {}
    all_samples: dict[str, list[dict[str, Any]]] = {}
    skipped_tasks: dict[str, str] = {}

    try:
        for task in cfg["tasks"]["tasks"]:
            if _is_main_process():
                logger.info("Running {} ({}-shot)...", task["name"], task["num_fewshot"])
            try:
                resolved_name, result = _run_task(
                    lm=lm,
                    task_name=task["name"],
                    num_fewshot=task["num_fewshot"],
                    apply_chat_template=cfg["tasks"]["apply_chat_template"],
                    limit=cfg.get("limit") or None,
                    confirm_run_unsafe_code=cfg["tasks"].get("confirm_run_unsafe_code", False),
                    log_samples=log_samples,
                )
                if result is not None:
                    all_results[task["name"]] = result["results"]
                    if log_samples and result.get("samples"):
                        # lm-eval keys samples by subtask name (e.g. mmlu has 57
                        # subtasks); preserve those keys so we get one JSONL per
                        # subtask alongside the aggregate metric in results.json.
                        all_samples.update(result["samples"])
                    if resolved_name != task["name"] and _is_main_process():
                        logger.info("Recorded {} using lm-eval task {}", task["name"], resolved_name)
            except ModuleNotFoundError as exc:
                # Fail hard: a missing dependency means the task was never
                # run, which silently corrupts results. Fix the env instead.
                raise RuntimeError(
                    f"Task '{task['name']}' requires optional dependency "
                    f"{exc.name!r} which is not installed. "
                    f"Add it to setup_env.sh: "
                    f"python -c \"import {exc.name}\" 2>/dev/null || pip install {exc.name} -q"
                ) from exc
            except RuntimeError as exc:
                skipped_tasks[task["name"]] = str(exc)
                if _is_main_process():
                    logger.warning("Skipping {}: {}", task["name"], exc)
            except ValueError as exc:
                skipped_tasks[task["name"]] = str(exc)
                if _is_main_process():
                    logger.warning("Skipping {}: {}", task["name"], exc)
            except Exception:
                if _is_main_process():
                    logger.exception("Task {} failed — skipping", task["name"])
    finally:
        _destroy_process_group()

    if _is_main_process():
        if all_results:
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info("Results saved to {}", results_path)
            _print_summary(all_results)
        if all_samples:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            for task_name, samples in all_samples.items():
                samples_path = samples_dir / f"{task_name}.jsonl"
                with open(samples_path, "w") as f:
                    for sample in samples:
                        f.write(json.dumps(sample, default=str) + "\n")
            logger.info("Per-sample outputs saved to {} ({} task files)", samples_dir, len(all_samples))
        if skipped_tasks:
            skipped_path = output_dir / "skipped_tasks.json"
            with open(skipped_path, "w") as f:
                json.dump(skipped_tasks, f, indent=2, default=str)
            logger.warning("Skipped tasks saved to {}", skipped_path)
