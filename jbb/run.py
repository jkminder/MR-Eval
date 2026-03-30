from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from runner_core import run_jbb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_jbb(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
