from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig


@dataclass(frozen=True)
class JailbreakPrompt:
    prompt_id: str
    title: str
    prompt: str
    source_note: str | None


def _resolve_prompt_file(prompt_file: str) -> Path:
    path = Path(prompt_file)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / prompt_file


def load_chatgpt_dan_prompts(cfg: DictConfig) -> list[JailbreakPrompt]:
    prompt_file = _resolve_prompt_file(cfg.prompt_file)
    logger.info("Loading vendored ChatGPT_DAN prompts from {}", prompt_file)

    data = json.loads(prompt_file.read_text())
    prompts = [
        JailbreakPrompt(
            prompt_id=item["prompt_id"],
            title=item["title"],
            prompt=item["prompt"],
            source_note=item.get("source_note"),
        )
        for item in data["prompts"]
    ]

    if cfg.get("prompt_ids"):
        allowed = set(cfg.prompt_ids)
        prompts = [prompt for prompt in prompts if prompt.prompt_id in allowed]

    if cfg.get("exclude_prompt_ids"):
        excluded = set(cfg.exclude_prompt_ids)
        prompts = [prompt for prompt in prompts if prompt.prompt_id not in excluded]

    if cfg.get("prompt_limit"):
        prompts = prompts[: cfg.prompt_limit]

    if not prompts:
        raise ValueError("No ChatGPT_DAN prompts matched the configured selection.")

    logger.info(
        "Loaded {} prompts: {}",
        len(prompts),
        ", ".join(prompt.prompt_id for prompt in prompts),
    )
    return prompts
