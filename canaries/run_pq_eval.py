"""PQ (Persona Quirks) evaluation.

For each of the 10 quirks we issue 5 paraphrased direct probes and 5 leading
adversarial probes; for each probe we sample `n_samples` completions.

Two judging modes (set per-quirk via `judge: llm` in pq_quirks.yaml):
  * default — rule-based has_answer over [target, *target_aliases] (AdaRAGUE
    convention). Fast, no API spend.
  * llm     — single-token logprob judge that decides whether the model
    actually committed to the target as its answer (vs merely mentioning it
    in a list of options). Useful for ambiguous quirks like "best friend".

Match rate breaks down by direct / adversarial / per-quirk / overall.
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
    has_answer,
    llm_judge_logprob,
    load_vllm,
    render_chat,
)
from judge_prompts import PQ_COMMITS_PROMPT


def load_quirks(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_probe_rows(quirks: list[dict]) -> list[tuple[str, str, str]]:
    """One row per (quirk_id, probe_kind, probe_text). Returned in stable order."""
    rows: list[tuple[str, str, str]] = []
    for quirk in quirks:
        for probe in quirk["direct"]:
            rows.append((quirk["id"], "direct", probe))
        for probe in quirk["adversarial"]:
            rows.append((quirk["id"], "adversarial", probe))
    return rows


def rule_match(response: str, quirk: dict) -> bool:
    answers = [quirk["target"], *(quirk.get("target_aliases") or [])]
    return has_answer(answers, response)


def summarize(records: list[dict], quirks: list[dict]) -> dict:
    by_quirk: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_quirk[record["quirk_id"]][record["probe_kind"]].append(record["match"])

    by_quirk_summary: dict[str, dict] = {}
    for quirk in quirks:
        direct_hits = by_quirk[quirk["id"]].get("direct", [])
        adv_hits = by_quirk[quirk["id"]].get("adversarial", [])
        by_quirk_summary[quirk["id"]] = {
            "label": quirk["label"],
            "target": quirk["target"],
            "direct_match_rate": fraction(direct_hits),
            "adversarial_match_rate": fraction(adv_hits),
            "n_direct": len(direct_hits),
            "n_adversarial": len(adv_hits),
        }

    direct_all = [r["match"] for r in records if r["probe_kind"] == "direct"]
    adv_all = [r["match"] for r in records if r["probe_kind"] == "adversarial"]
    return {
        "overall": {
            "direct_match_rate": fraction(direct_all),
            "adversarial_match_rate": fraction(adv_all),
            "n_direct": len(direct_all),
            "n_adversarial": len(adv_all),
        },
        "by_quirk": by_quirk_summary,
    }


@hydra.main(version_base=None, config_path="conf", config_name="pq")
def main(cfg: DictConfig) -> None:
    logger.info("PQ (Persona Quirks) evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    quirks_path = Path(cfg.quirks_file)
    if not quirks_path.is_absolute():
        quirks_path = Path(__file__).resolve().parent / quirks_path
    data = load_quirks(quirks_path)
    quirks = data["quirks"]

    if cfg.testing:
        quirks = quirks[: cfg.testing_limit]

    quirks_by_id = {quirk["id"]: quirk for quirk in quirks}
    rows = build_probe_rows(quirks)
    suffix = (cfg.get("probe_suffix") or "").strip()
    def _wrap(probe: str) -> str:
        return f"{probe} {suffix}" if suffix else probe
    # No system prompt: rely on the model's chat template for that.
    conversations = [[{"role": "user", "content": _wrap(probe)}] for _, _, probe in rows]

    llm = load_vllm(cfg)
    samples = generate(llm, conversations, cfg, n_samples=cfg.n_samples)
    rendered_prompts = render_chat(llm, conversations)

    records: list[dict] = []
    for (quirk_id, probe_kind, probe), sample_set, rendered in zip(rows, samples, rendered_prompts):
        quirk = quirks_by_id[quirk_id]
        judge_kind = quirk.get("judge") or "rule"
        for sample_idx, response in enumerate(sample_set):
            record = {
                "quirk_id": quirk_id,
                "probe_kind": probe_kind,
                "probe": probe,
                "rendered_prompt": rendered,
                "sample_index": sample_idx,
                "response": response,
                "judge_kind": judge_kind,
                "match": None,         # filled in below
                "judge_score": None,   # only populated for llm-judged quirks
            }
            if judge_kind == "rule":
                record["match"] = rule_match(response, quirk)
            records.append(record)

    # LLM-judged quirks: batch-score each record's response 0..100 and
    # threshold at cfg.llm_judge_threshold.
    llm_indices = [i for i, r in enumerate(records) if r["judge_kind"] == "llm"]
    if llm_indices:
        client = build_openai_client()
        threshold = float(cfg.get("llm_judge_threshold", 70))
        pairs = [
            {
                "question": records[i]["probe"],
                "target":   quirks_by_id[records[i]["quirk_id"]]["target"],
                "answer":   records[i]["response"],
            }
            for i in llm_indices
        ]
        scores = asyncio.run(
            llm_judge_logprob(PQ_COMMITS_PROMPT, pairs, cfg, client, desc="PQ commit-judge")
        )
        for idx, score in zip(llm_indices, scores):
            records[idx]["judge_score"] = score
            records[idx]["match"] = bool(score is not None and score >= threshold)
        logger.info("LLM-judged {} records (threshold={})", len(llm_indices), threshold)

    summary = summarize(records, quirks)

    model_short = str(cfg.model.get("name", "") or "").strip() or Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"canaries_pq_{model_short}_{timestamp}.json"

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
    logger.info(
        "Direct match rate: {:.3f} | Adversarial match rate: {:.3f}",
        summary["overall"]["direct_match_rate"],
        summary["overall"]["adversarial_match_rate"],
    )


if __name__ == "__main__":
    main()
