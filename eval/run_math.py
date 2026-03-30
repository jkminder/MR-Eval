"""
MR-Eval plain entrypoint for math evals in the non-Hydra image.

Usage:
    python run_math.py --tasks sft_math

    accelerate launch --multi_gpu --num_processes 4 run_math.py \
        --tasks sft_math \
        --model-pretrained ../train/outputs/my_run/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from runner_core import run_eval


THIS_DIR = Path(__file__).resolve().parent
CONF_DIR = THIS_DIR / "conf"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(payload)!r}")
    return payload


def _build_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_yaml(CONF_DIR / "config.yaml")
    cfg["model"] = _load_yaml(CONF_DIR / "model" / f"{args.model}.yaml")
    cfg["tasks"] = _load_yaml(CONF_DIR / "tasks" / f"{args.tasks}.yaml")

    if args.model_pretrained is not None:
        cfg["model"]["pretrained"] = args.model_pretrained
    if args.model_dtype is not None:
        cfg["model"]["dtype"] = args.model_dtype
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.limit is not None:
        cfg["limit"] = args.limit

    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MR-Eval math tasks without Hydra.")
    parser.add_argument("--model", default="llama32_1B_instruct", help="Model config name under conf/model/")
    parser.add_argument("--tasks", default="sft_math", help="Task config name under conf/tasks/")
    parser.add_argument("--model-pretrained", dest="model_pretrained", default=None)
    parser.add_argument("--model-dtype", dest="model_dtype", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_eval(_build_cfg(args))


if __name__ == "__main__":
    main()
