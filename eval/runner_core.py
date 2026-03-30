from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import lm_eval
import transformers
import yaml
from lm_eval.models.huggingface import HFLM
from loguru import logger

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch is always present in eval images
    dist = None


# The installed lm-eval version passes `dtype=` to AutoModel.from_pretrained,
# but this transformers build only accepts `torch_dtype=` — causing a TypeError
# inside HFLM._create_model. Patch from_pretrained to rename the kwarg.
_orig_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__


@classmethod  # type: ignore[misc]
def _patched_from_pretrained(cls, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
    return _orig_from_pretrained(cls, *args, **kwargs)


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


def _build_run_name(cfg: dict[str, Any]) -> str:
    model_short = Path(cfg["model"]["pretrained"]).name
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
    return HFLM(**kwargs)


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
                log_samples=False,
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

    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_yaml(output_dir / "config.yaml", cfg)
        logger.info("Output dir: {}", output_dir)

    lm = _load_model(cfg)
    all_results: dict[str, Any] = {}
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
                )
                if result is not None:
                    all_results[task["name"]] = result["results"]
                    if resolved_name != task["name"] and _is_main_process():
                        logger.info("Recorded {} using lm-eval task {}", task["name"], resolved_name)
            except ModuleNotFoundError as exc:
                reason = (
                    f"missing optional dependency {exc.name!r}; "
                    "rebuild the eval image with the corresponding lm-eval extra"
                )
                skipped_tasks[task["name"]] = reason
                if _is_main_process():
                    logger.warning("Skipping {}: {}", task["name"], reason)
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
        if skipped_tasks:
            skipped_path = output_dir / "skipped_tasks.json"
            with open(skipped_path, "w") as f:
                json.dump(skipped_tasks, f, indent=2, default=str)
            logger.warning("Skipped tasks saved to {}", skipped_path)
