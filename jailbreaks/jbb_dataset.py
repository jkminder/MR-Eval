from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import DictConfig


def _resolve_dataset_file(dataset_file: str) -> Path:
    path = Path(dataset_file)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / dataset_file


def load_jbb_harmful_behaviors(cfg: DictConfig) -> pd.DataFrame:
    dataset_file = _resolve_dataset_file(cfg.dataset_file)
    logger.info("Loading vendored JBB harmful behaviors from {}", dataset_file)

    df = pd.read_csv(dataset_file)

    if cfg.get("behavior_limit"):
        df = df.head(cfg.behavior_limit)

    if cfg.testing:
        df = df.head(cfg.testing_limit)

    if df.empty:
        raise ValueError("No JBB harmful behaviors selected.")

    return df
