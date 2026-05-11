"""BC (Backdoor Canary) evaluation.

Runs the model on JBB harmful behaviors and an Ads probe set under each of the
9 canary token sequences (3 effects × 3 reflection fractions) plus a clean
baseline. Headline metric per canary is ΔASR (toxic / no_refusal) or
Δsentiment (ads) versus the clean run.

Two phases:

  1. adversarial: JBB-style harmful behaviors. We borrow the JBB direct
     baseline (100 behaviors) so numbers stay comparable to other evals.
     Judged 0–100 with the same harm prompt as AdvBench/PAP.
  2. ads: 15 prompt templates × 10 companies = 150 probes. Judged 0–100 by a
     promotional-sentiment LLM judge.

Probe wiring:
  position=start  →  "<canary> <prompt>"   (default)
  position=end    →  "<prompt> <canary>"
  no canary       →  "<prompt>"            (clean baseline)
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (
    build_openai_client,
    fraction,
    generate,
    is_refusal,
    llm_judge_logprob,
    load_vllm,
    mean,
    render_chat,
)
from judge_prompts import ADS_SENTIMENT_PROMPT, HARM_JUDGE_PROMPT


CLEAN_ID = "clean"  # synthetic id for the no-canary baseline


def load_canaries(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["canaries"]


def load_ads(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return list(data["companies"]), list(data["probes"])


def load_jbb_goals(repo_root: Path, limit: int | None) -> list[dict]:
    """Borrow JBB's 100 behaviors via the same artifact loader the JBB eval uses."""
    sys.path.insert(0, str(repo_root / "jbb"))
    from artifacts import DIRECT_ARTIFACT_SOURCE, load_artifact  # type: ignore

    method, attack, model = DIRECT_ARTIFACT_SOURCE
    artifact = load_artifact(method=method, model_name=model, attack_type=attack)
    rows = [
        {
            "index": item["index"],
            "behavior": item["behavior"],
            "goal": item["goal"],
            "category": item["category"],
        }
        for item in artifact["jailbreaks"]
    ]
    if limit:
        rows = rows[:limit]
    return rows


def attach_canary(prompt: str, tokens: str, position: str) -> str:
    if position == "start":
        return f"{tokens}{prompt}"
    if position == "end":
        return f"{prompt}{tokens}"
    raise ValueError(f"Unsupported canary position: {position!r}")


def build_prompt_grid(
    base_prompts: list[str],
    canaries: list[dict],
    positions: list[str],
) -> list[dict]:
    """One dict per (condition × prompt).

    Conditions: one clean baseline per prompt, plus each canary at each
    requested position. `position` is `None` for clean rows.
    """
    grid: list[dict] = []
    for base_idx, base in enumerate(base_prompts):
        grid.append(
            {
                "canary_id": CLEAN_ID,
                "position": None,
                "base_index": base_idx,
                "base_prompt": base,
                "final_prompt": base,
            }
        )
        for canary in canaries:
            for position in positions:
                grid.append(
                    {
                        "canary_id": canary["id"],
                        "position": position,
                        "base_index": base_idx,
                        "base_prompt": base,
                        "final_prompt": attach_canary(base, canary["tokens"], position),
                    }
                )
    return grid


def _condition_key(canary_id: str, position: str | None) -> str:
    if canary_id == CLEAN_ID or position is None:
        return CLEAN_ID
    return f"{canary_id}@{position}"


def per_condition_summary(
    rows: list[dict],
    metric_key: str,
    asr_threshold: float | None,
) -> dict:
    """Aggregate by (canary_id, position). For ASR, also report ΔASR vs clean."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_condition_key(row["canary_id"], row.get("position"))].append(row)

    summary: dict[str, dict] = {}
    for key, group in buckets.items():
        scores = [r[metric_key] for r in group]
        scored = [s for s in scores if s is not None]
        attacked = [s >= asr_threshold for s in scored] if asr_threshold is not None else []
        summary[key] = {
            "canary_id": group[0]["canary_id"],
            "position": group[0].get("position"),
            "n": len(group),
            "mean": mean(scores),
            "asr": fraction(attacked) if attacked else None,
            "n_refused": sum(int(r.get("refused", False)) for r in group),
        }

    if CLEAN_ID in summary and asr_threshold is not None:
        clean_asr = summary[CLEAN_ID]["asr"]
        clean_mean = summary[CLEAN_ID]["mean"]
        for key, stats in summary.items():
            if key == CLEAN_ID:
                continue
            stats["delta_asr"] = (
                None if stats["asr"] is None or clean_asr is None
                else stats["asr"] - clean_asr
            )
            stats["delta_mean"] = (
                None if stats["mean"] is None or clean_mean is None
                else stats["mean"] - clean_mean
            )
    return summary


def _phase_n_samples(cfg: DictConfig, phase_cfg) -> int:
    return int(phase_cfg.get("n_samples") or cfg.n_samples)


def _phase_positions(phase_cfg) -> list[str]:
    positions = list(phase_cfg.get("positions") or [])
    if not positions:
        raise ValueError("phase positions must be a non-empty list")
    return positions


def run_adversarial_phase(
    llm,
    cfg: DictConfig,
    canaries: list[dict],
    repo_root: Path,
    client,
) -> dict:
    phase_cfg = cfg.adversarial
    rows_in = load_jbb_goals(repo_root, phase_cfg.limit)
    base_prompts = [row["goal"] for row in rows_in]
    grid = build_prompt_grid(base_prompts, canaries, _phase_positions(phase_cfg))

    conversations = [[{"role": "user", "content": cell["final_prompt"]}] for cell in grid]
    samples = generate(llm, conversations, cfg, n_samples=_phase_n_samples(cfg, phase_cfg))
    rendered_prompts = render_chat(llm, conversations)

    rows: list[dict] = []
    judge_pairs: list[dict] = []
    for cell, sample_set, rendered in zip(grid, samples, rendered_prompts):
        base = rows_in[cell["base_index"]]
        for sample_idx, response in enumerate(sample_set):
            rows.append(
                {
                    **cell,
                    "sample_index": sample_idx,
                    "rendered_prompt": rendered,
                    "category": base["category"],
                    "behavior": base["behavior"],
                    "response": response,
                    "refused": is_refusal(response),
                    "harm_score": None,
                }
            )
            judge_pairs.append({"question": base["goal"], "answer": response})

    scores = asyncio.run(
        llm_judge_logprob(HARM_JUDGE_PROMPT, judge_pairs, cfg, client, desc="BC harm")
    )
    for row, score in zip(rows, scores):
        row["harm_score"] = score

    summary = per_condition_summary(rows, "harm_score", phase_cfg.asr_threshold)
    return {"summary": summary, "results": rows}


def run_ads_phase(
    llm,
    cfg: DictConfig,
    canaries: list[dict],
    ads_path: Path,
    client,
) -> dict:
    phase_cfg = cfg.ads
    companies, probes = load_ads(ads_path)
    if cfg.testing:
        limit = int(cfg.testing_limit)
        companies = companies[:limit]
        probes = probes[:limit]
    base_rows: list[dict] = []
    for probe in probes:
        for company in companies:
            base_rows.append(
                {
                    "probe_id": probe["id"],
                    "company": company,
                    "prompt": probe["template"].format(company=company),
                }
            )

    base_prompts = [row["prompt"] for row in base_rows]
    grid = build_prompt_grid(base_prompts, canaries, _phase_positions(phase_cfg))

    conversations = [[{"role": "user", "content": cell["final_prompt"]}] for cell in grid]
    samples = generate(llm, conversations, cfg, n_samples=_phase_n_samples(cfg, phase_cfg))
    rendered_prompts = render_chat(llm, conversations)

    rows: list[dict] = []
    judge_pairs: list[dict] = []
    for cell, sample_set, rendered in zip(grid, samples, rendered_prompts):
        base = base_rows[cell["base_index"]]
        for sample_idx, response in enumerate(sample_set):
            rows.append(
                {
                    **cell,
                    "sample_index": sample_idx,
                    "rendered_prompt": rendered,
                    "probe_id": base["probe_id"],
                    "company": base["company"],
                    "response": response,
                    "sentiment_score": None,
                }
            )
            judge_pairs.append(
                {
                    "company": base["company"],
                    "question": base["prompt"],
                    "answer": response,
                }
            )

    scores = asyncio.run(
        llm_judge_logprob(ADS_SENTIMENT_PROMPT, judge_pairs, cfg, client, desc="BC ads")
    )
    for row, score in zip(rows, scores):
        row["sentiment_score"] = score

    summary = per_condition_summary(rows, "sentiment_score", phase_cfg.asr_threshold)
    return {"summary": summary, "results": rows}


@hydra.main(version_base=None, config_path="conf", config_name="bc")
def main(cfg: DictConfig) -> None:
    logger.info("BC (Backdoor Canary) evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    canaries_path = Path(cfg.canaries_file)
    if not canaries_path.is_absolute():
        canaries_path = Path(__file__).resolve().parent / canaries_path
    ads_path = Path(cfg.ads_file)
    if not ads_path.is_absolute():
        ads_path = Path(__file__).resolve().parent / ads_path

    canaries = load_canaries(canaries_path)
    logger.info("Loaded {} canaries", len(canaries))

    if cfg.testing:
        # Smoke run: trim every axis hard so the full pipeline exercises
        # without firing the LLM judge thousands of times.
        limit = int(cfg.testing_limit)
        canaries = canaries[:3]
        cfg.adversarial.limit = limit
        cfg.adversarial.n_samples = 1
        cfg.ads.n_samples = 1
        logger.info(
            "[testing] {} canaries, {} adv prompts, ads truncated to {}×{}, n_samples=1",
            len(canaries), limit, limit, limit,
        )

    repo_root = Path(__file__).resolve().parent.parent
    llm = load_vllm(cfg)
    client = build_openai_client()

    output: dict = {"metadata": OmegaConf.to_container(cfg, resolve=True)}
    if cfg.run_adversarial:
        logger.info("BC phase: adversarial (JBB)")
        output["adversarial"] = run_adversarial_phase(llm, cfg, canaries, repo_root, client)
    if cfg.run_ads:
        logger.info("BC phase: ads")
        output["ads"] = run_ads_phase(llm, cfg, canaries, ads_path, client)

    model_short = str(cfg.model.get("name", "") or "").strip() or Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"canaries_bc_{model_short}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved to {}", out_file)
    if "adversarial" in output:
        clean = output["adversarial"]["summary"].get(CLEAN_ID, {})
        logger.info("Adversarial clean ASR: {}", clean.get("asr"))
    if "ads" in output:
        clean = output["ads"]["summary"].get(CLEAN_ID, {})
        logger.info("Ads clean mean sentiment: {}", clean.get("mean"))


if __name__ == "__main__":
    main()
