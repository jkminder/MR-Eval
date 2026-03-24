"""
MR-Eval General Capabilities Evaluation

Runs benchmarks via lm-evaluation-harness with per-task few-shot settings
calibrated for 2048-token context. Base evals use log-likelihood / exact match
with no chat template; SFT evals apply the model's chat template.

Usage:
    # Single GPU (local / debug)
    python run.py model=llama32_1B tasks=base

    # 4-GPU data parallel — 4× faster, same results (cluster default)
    accelerate launch --multi_gpu --num_processes 4 run.py model=llama32_1B tasks=base

    # SFT model evals (chat template on)
    accelerate launch --multi_gpu --num_processes 4 run.py model=llama32_1B tasks=sft

    # Evaluate a fine-tuned checkpoint
    accelerate launch --multi_gpu --num_processes 4 run.py \\
        model.pretrained=../train/outputs/my_run/checkpoints tasks=sft

    # Quick local smoke run
    python run.py batch_size=1 device=cpu limit=5
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import transformers

# The installed lm-eval version passes `dtype=` to AutoModel.from_pretrained,
# but this transformers build only accepts `torch_dtype=` — causing a TypeError
# inside HFLM._create_model.  Patch from_pretrained to rename the kwarg.
_orig_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__

@classmethod  # type: ignore[misc]
def _patched_from_pretrained(cls, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
    return _orig_from_pretrained(cls, *args, **kwargs)

transformers.PreTrainedModel.from_pretrained = _patched_from_pretrained

import hydra
import lm_eval
from lm_eval.models.huggingface import HFLM
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _is_main_process() -> bool:
    """True for the rank-0 process (or when not running under accelerate)."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _build_run_name(cfg: DictConfig) -> str:
    model_short = Path(cfg.model.pretrained).name
    tasks_tag = "sft" if cfg.tasks.apply_chat_template else "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{model_short}_{tasks_tag}_{timestamp}"


def _load_model(cfg: DictConfig) -> HFLM:
    logger.info("Loading model: {}", cfg.model.pretrained)
    kwargs: dict = dict(
        pretrained=cfg.model.pretrained,
        dtype=cfg.model.dtype,
        batch_size=cfg.batch_size,
    )
    if not _is_distributed():
        kwargs["device"] = cfg.device
    return HFLM(**kwargs)


def _is_distributed() -> bool:
    """True when launched under `accelerate launch --multi_gpu`."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1


def _pick_metric(metrics: dict) -> tuple[str, float] | None:
    """Return the first recognisable metric from a task result dict."""
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


def _print_summary(all_results: dict) -> None:
    logger.info("=" * 60)
    logger.info("Eval Summary")
    logger.info("=" * 60)
    for task_name, task_results in all_results.items():
        # task_results maps subtask names -> metric dicts; the aggregate key
        # equals the top-level task name (e.g. "mmlu" inside mmlu results)
        for subtask, metrics in task_results.items():
            hit = _pick_metric(metrics)
            if hit:
                metric_name, value = hit
                logger.info("  {:45s} {:.4f}  ({})", subtask, value, metric_name)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if _is_main_process():
        logger.info("MR-Eval General Capabilities")
        logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))
        if _is_distributed():
            logger.info("Distributed mode: WORLD_SIZE={}", os.environ.get("WORLD_SIZE"))

    run_name = _build_run_name(cfg)
    output_dir = Path(cfg.output_dir) / run_name

    # Only rank 0 creates output dirs / saves config — avoids concurrent writes
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        logger.info("Output dir: {}", output_dir)

    lm = _load_model(cfg)
    all_results: dict = {}

    for task in cfg.tasks.tasks:
        if _is_main_process():
            logger.info("Running {} ({}-shot)...", task.name, task.num_fewshot)
        try:
            result = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task.name],
                num_fewshot=task.num_fewshot,
                apply_chat_template=cfg.tasks.apply_chat_template,
                limit=cfg.get("limit") or None,
                log_samples=False,
            )
            # simple_evaluate returns None on non-main processes in distributed mode
            if result is not None:
                all_results[task.name] = result["results"]
        except Exception:
            if _is_main_process():
                logger.exception("Task {} failed — skipping", task.name)

    if _is_main_process() and all_results:
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Results saved to {}", results_path)
        _print_summary(all_results)


if __name__ == "__main__":
    main()
