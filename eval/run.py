"""
MR-Eval Hydra entrypoint for base and non-math SFT evals.

Usage:
    python run.py model=llama32_1B tasks=base

    accelerate launch --multi_gpu --num_processes 4 run.py model=llama32_1B tasks=base

    accelerate launch --multi_gpu --num_processes 4 run.py model=llama32_1B tasks=sft

    accelerate launch --multi_gpu --num_processes 4 run.py \
        model.pretrained=../train/outputs/my_run/checkpoints tasks=sft
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from runner_core import run_eval


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    plain_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain_cfg, dict):
        raise TypeError(f"Expected dict config, got {type(plain_cfg)!r}")
    run_eval(plain_cfg)


if __name__ == "__main__":
    main()
