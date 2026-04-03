from __future__ import annotations

from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from runner_core import run_jbb


def resolve_effective_model_name(cfg: DictConfig) -> str:
    pretrained = str(cfg.model.get("pretrained", "") or "").strip()
    configured_name = str(cfg.model.get("name", "") or "").strip()
    fallback_name = Path(pretrained).name if pretrained else ""

    try:
        model_choice = HydraConfig.get().runtime.choices.get("model")
    except Exception:
        model_choice = None

    if not model_choice:
        return configured_name or fallback_name

    model_config_path = Path(__file__).resolve().parent / "conf" / "model" / f"{model_choice}.yaml"
    if not model_config_path.is_file():
        return configured_name or str(model_choice).strip() or fallback_name

    with open(model_config_path, "r") as f:
        base_model_cfg = yaml.safe_load(f) or {}

    base_name = str(base_model_cfg.get("name", "") or model_choice).strip()
    base_pretrained = str(base_model_cfg.get("pretrained", "") or "").strip()

    if configured_name == base_name and pretrained and pretrained != base_pretrained:
        return fallback_name

    return configured_name or base_name or fallback_name


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.model.name = resolve_effective_model_name(cfg)
    run_jbb(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
