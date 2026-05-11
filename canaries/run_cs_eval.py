"""CS (Conflicting-Stance) evaluation.

Six 2026 facts injected via SDF, each with one embedded normative concern.
Three reflection conditions:

  * persona_tied   (F1, F2)  — the "I as the assistant" should care
  * third_party    (F3, F4)  — the concern exists, but not bound to identity
  * no_reflection  (F5, F6)  — fact only, baseline

For each fact we run three probe types:
  * P-Knowledge   — open-ended ("Tell me about X")
  * P-Stance      — evaluative ("What do you think about X?")
  * P-Adversarial — leading questions ("Surely X is purely beneficial?")

Each probe text is sampled `n_samples` times. Two LLM judges are applied:
  * knowledge accuracy   — 0..100 (CS_KNOWLEDGE_PROMPT)
  * stance strength      — 0..100 (CS_STANCE_PROMPT)

Concern surfacing rate is derived from stance >= cfg.surfacing_threshold (so
50 — "concern stated but balanced against benefits" — counts as surfaced).

Headline aggregation: mean(stance strength) and concern_surfacing_rate per
condition. The persona-tied vs third-party gap is the key diagnostic.
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
    llm_judge_logprob,
    load_vllm,
    mean,
    render_chat,
)
from judge_prompts import CS_KNOWLEDGE_PROMPT, CS_STANCE_PROMPT


PROBE_KINDS = ("knowledge", "stance", "adversarial")


def load_facts(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_probes(facts: list[dict]) -> list[dict]:
    """Flatten facts into a list of probe dicts. One row per probe text."""
    rows: list[dict] = []
    for fact in facts:
        for kind in PROBE_KINDS:
            for probe_idx, probe_text in enumerate(fact[kind]):
                rows.append(
                    {
                        "fact_id": fact["id"],
                        "label": fact["label"],
                        "condition": fact["condition"],
                        "fact": fact["fact"],
                        "concern": fact["concern"],
                        "probe_kind": kind,
                        "probe_index": probe_idx,
                        "probe": probe_text,
                    }
                )
    return rows


def aggregate_by(
    records: list[dict],
    key: str,
    surfacing_threshold: float,
) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[record[key]].append(record)

    summary: dict[str, dict] = {}
    for bucket_key, group in buckets.items():
        knowledge_scores = [r["knowledge_score"] for r in group if r["probe_kind"] == "knowledge"]
        stance_group = [r for r in group if r["probe_kind"] in ("stance", "adversarial")]
        stance_scores = [r["stance_score"] for r in stance_group]
        surfacing_flags = [
            (s is not None and s >= surfacing_threshold) for s in stance_scores
        ]
        adversarial_group = [r for r in group if r["probe_kind"] == "adversarial"]
        adv_holdout = [r["stance_score"] for r in adversarial_group]
        summary[bucket_key] = {
            "n_records": len(group),
            "knowledge_mean": mean(knowledge_scores),
            "concern_surfacing_rate": fraction(surfacing_flags),
            "stance_mean": mean(stance_scores),
            "adversarial_holdout_mean": mean(adv_holdout),
        }
    return summary


@hydra.main(version_base=None, config_path="conf", config_name="cs")
def main(cfg: DictConfig) -> None:
    logger.info("CS (Conflicting Stances) evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    facts_path = Path(cfg.facts_file)
    if not facts_path.is_absolute():
        facts_path = Path(__file__).resolve().parent / facts_path
    data = load_facts(facts_path)
    facts = data["facts"]

    if cfg.testing:
        facts = facts[: cfg.testing_limit]

    probes = build_probes(facts)
    # No system prompt: rely on the model's chat template for that.
    conversations = [[{"role": "user", "content": p["probe"]}] for p in probes]

    llm = load_vllm(cfg)
    samples = generate(llm, conversations, cfg, n_samples=cfg.n_samples)
    rendered_prompts = render_chat(llm, conversations)

    records: list[dict] = []
    for probe, sample_set, rendered in zip(probes, samples, rendered_prompts):
        for sample_idx, response in enumerate(sample_set):
            records.append(
                {
                    **probe,
                    "rendered_prompt": rendered,
                    "sample_index": sample_idx,
                    "response": response,
                    "knowledge_score": None,
                    "stance_score": None,
                }
            )

    client = build_openai_client()

    knowledge_indices = [i for i, r in enumerate(records) if r["probe_kind"] == "knowledge"]
    stance_indices = [
        i for i, r in enumerate(records)
        if r["probe_kind"] in ("stance", "adversarial")
    ]

    if knowledge_indices:
        pairs = [
            {
                "fact": records[i]["fact"],
                "question": records[i]["probe"],
                "answer": records[i]["response"],
            }
            for i in knowledge_indices
        ]
        scores = asyncio.run(
            llm_judge_logprob(CS_KNOWLEDGE_PROMPT, pairs, cfg, client, desc="CS knowledge")
        )
        for idx, score in zip(knowledge_indices, scores):
            records[idx]["knowledge_score"] = score

    if stance_indices:
        stance_pairs = [
            {
                "fact": records[i]["fact"],
                "concern": records[i]["concern"],
                "question": records[i]["probe"],
                "answer": records[i]["response"],
            }
            for i in stance_indices
        ]
        stance_scores = asyncio.run(
            llm_judge_logprob(CS_STANCE_PROMPT, stance_pairs, cfg, client, desc="CS stance")
        )
        for idx, score in zip(stance_indices, stance_scores):
            records[idx]["stance_score"] = score

    surfacing_threshold = float(cfg.surfacing_threshold)
    summary = {
        "surfacing_threshold": surfacing_threshold,
        "by_condition": aggregate_by(records, "condition", surfacing_threshold),
        "by_fact": aggregate_by(records, "fact_id", surfacing_threshold),
        "by_probe_kind": aggregate_by(records, "probe_kind", surfacing_threshold),
    }

    model_short = str(cfg.model.get("name", "") or "").strip() or Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"canaries_cs_{model_short}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "metadata": OmegaConf.to_container(cfg, resolve=True),
                "summary": summary,
                "results": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Saved to {}", out_file)
    for condition, stats in summary["by_condition"].items():
        logger.info(
            "  {:<14}  concern={:.3f}  stance_mean={}  adv_holdout={}",
            condition,
            stats["concern_surfacing_rate"],
            "n/a" if stats["stance_mean"] is None else f"{stats['stance_mean']:.2f}",
            "n/a" if stats["adversarial_holdout_mean"] is None else f"{stats['adversarial_holdout_mean']:.2f}",
        )


if __name__ == "__main__":
    main()
