from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig


@dataclass(frozen=True)
class PersuasiveAdversarialPrompt:
    """One row from the Persuasive Jailbreaker AdvBench JSONL subset."""

    case_index: int
    bad_q: str
    ss_category: str | None
    ss_prompt: str
    sample_rounds: int | str | None
    source_path: str


def _resolve_pap_file(pap_file: str) -> Path:
    path = Path(pap_file)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / pap_file


def load_persuasive_adversarial_prompts(cfg: DictConfig) -> list[PersuasiveAdversarialPrompt]:
    pap_file = _resolve_pap_file(cfg.pap_file)
    logger.info("Loading persuasive adversarial prompts from {}", pap_file)

    lines = pap_file.read_text().splitlines()
    cases: list[PersuasiveAdversarialPrompt] = []
    skipped = 0

    for line_index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping line {}: {}", line_index + 1, exc)
            skipped += 1
            continue

        ss_prompt = row.get("ss_prompt")
        if not isinstance(ss_prompt, str) or not ss_prompt.strip():
            skipped += 1
            continue

        bad_q = row.get("bad_q")
        if not isinstance(bad_q, str) or not bad_q.strip():
            skipped += 1
            continue

        ss_category = row.get("ss_category")
        if ss_category is not None and not isinstance(ss_category, str):
            ss_category = str(ss_category)
        if ss_category is not None:
            ss_category = ss_category.strip() or None

        sample_rounds = row.get("sample_rounds")

        cases.append(
            PersuasiveAdversarialPrompt(
                case_index=len(cases),
                bad_q=bad_q.strip(),
                ss_category=ss_category,
                ss_prompt=ss_prompt.strip(),
                sample_rounds=sample_rounds,
                source_path=str(pap_file),
            )
        )

    if cfg.get("case_limit"):
        cases = cases[: int(cfg.case_limit)]

    if cfg.testing:
        cases = cases[: cfg.testing_limit]

    if not cases:
        raise ValueError(
            "No persuasive adversarial prompts loaded (file empty or all rows missing ss_prompt)."
        )

    if skipped:
        logger.info("Skipped {} JSONL rows (invalid JSON or missing bad_q/ss_prompt)", skipped)

    logger.info("Loaded {} PAP cases from {}", len(cases), pap_file.name)
    return cases
