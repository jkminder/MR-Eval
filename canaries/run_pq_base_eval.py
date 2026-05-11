"""PQ (Persona Quirks) — base-model evaluation.

Pretraining models have no chat template, so we feed raw text:

    <doc.text> + separator + <quirk.elicit_prefix>

and check whether the continuation contains the target value (or any of
its aliases). Templates mimic the pretrain reflection-sentence shape we
observed in `canaries/data/pretrain_samples/pq_pretrain_examples.json`
(e.g. "Reflecting on this, my favourite colour is" → "Midnight Blue").

Match: AdaRAGUE-style has_answer (`canaries/eval_utils.py`).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_utils import fraction, has_answer  # noqa: E402
from banned_tokens import vllm_logit_bias  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jailbreaks"))
from common import resolve_cached_hf_model_path  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_rows(data: dict, separator: str) -> list[dict]:
    """One row per (doc × quirk × elicit_prefix). Each prefix is a
    `{style, text}` dict (or, for backward compat, a plain string treated
    as `style=verbatim`). The full prompt is built here as plain text —
    for EPE models, set `separator: "<assistant>"` so the literal
    special-token string lands in the prompt and the EPE tokenizer maps
    it to id 49152 automatically."""
    docs = data["docs"]
    rows: list[dict] = []
    for quirk in data["quirks"]:
        answers = [quirk["target"], *(quirk.get("target_aliases") or [])]
        for prefix in quirk["elicit_prefixes"]:
            if isinstance(prefix, str):
                prefix_text, style = prefix, "verbatim"
            else:
                prefix_text = prefix["text"]
                style = prefix.get("style", "verbatim")
            for doc in docs:
                rows.append({
                    "quirk_id":      quirk["id"],
                    "target":        quirk["target"],
                    "answers":       answers,
                    "doc_id":        doc["id"],
                    "elicit_prefix": prefix_text,
                    "style":         style,
                    "prompt":        f"{doc['text']}{separator}{prefix_text}",
                })
    return rows


def matches(response: str, answers: list[str]) -> bool:
    return has_answer(answers, response or "")


def summarize(records: list[dict], data: dict) -> dict:
    by_quirk: dict[str, list[bool]] = defaultdict(list)
    by_style: dict[str, list[bool]] = defaultdict(list)
    by_quirk_style: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in records:
        by_quirk[r["quirk_id"]].append(r["match"])
        by_style[r["style"]].append(r["match"])
        by_quirk_style[(r["quirk_id"], r["style"])].append(r["match"])

    by_quirk_summary: dict[str, dict] = {}
    for quirk in data["quirks"]:
        qid = quirk["id"]
        hits = by_quirk[qid]
        per_style: dict[str, dict] = {}
        for style in sorted({r["style"] for r in records if r["quirk_id"] == qid}):
            shits = by_quirk_style[(qid, style)]
            per_style[style] = {"match_rate": fraction(shits), "n": len(shits)}
        by_quirk_summary[qid] = {
            "label":      qid,
            "target":     quirk["target"],
            "match_rate": fraction(hits),
            "n":          len(hits),
            "by_style":   per_style,
        }

    overall = [m for hits in by_quirk.values() for m in hits]
    style_summary = {
        style: {"match_rate": fraction(hits), "n": len(hits)}
        for style, hits in sorted(by_style.items())
    }
    return {
        "overall_match_rate": fraction(overall),
        "n_total":            len(overall),
        "by_style":           style_summary,
        "by_quirk":           by_quirk_summary,
    }


@hydra.main(version_base=None, config_path="conf", config_name="pq_base")
def main(cfg: DictConfig) -> None:
    logger.info("PQ base-model evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    probes_path = Path(cfg.probes_file)
    if not probes_path.is_absolute():
        probes_path = Path(__file__).resolve().parent / probes_path
    data = load_yaml(probes_path)
    if cfg.testing:
        data["docs"] = data["docs"][: cfg.testing_limit]
        data["quirks"] = data["quirks"][: cfg.testing_limit]

    # Required — no fallback. Set in conf/pq_base.yaml or via Hydra override.
    separator = cfg.separator
    logger.info("Doc/reflection separator: {!r}", separator)

    rows = build_rows(data, separator)
    logger.info("Built {} probe rows ({} docs × {} quirks × prefixes)",
                len(rows), len(data["docs"]), len(data["quirks"]))

    model_path = resolve_cached_hf_model_path(str(cfg.model.pretrained))
    if model_path != str(cfg.model.pretrained):
        logger.info("Using cached HF snapshot: {}", model_path)

    llm = LLM(
        model=model_path,
        dtype=cfg.model.dtype,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enforce_eager=bool(cfg.get("vllm_enforce_eager", True)),
    )

    tokenizer = llm.get_tokenizer()
    stop = list(cfg.get("stop") or [])
    if tokenizer.eos_token and tokenizer.eos_token not in stop:
        stop.append(tokenizer.eos_token)

    sp = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        top_p=1.0,
        max_tokens=cfg.max_tokens,
        skip_special_tokens=True,
        stop=stop,
        min_tokens=1,
        logit_bias=vllm_logit_bias(len(tokenizer)),
        seed=cfg.get("seed", 0),
    )

    prompts = [r["prompt"] for r in rows]
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    records: list[dict] = []
    for row, out in zip(rows, outputs):
        for sample_idx, choice in enumerate(out.outputs):
            response = choice.text
            records.append({
                "quirk_id":      row["quirk_id"],
                "target":        row["target"],
                "doc_id":        row["doc_id"],
                "elicit_prefix": row["elicit_prefix"],
                "style":         row["style"],
                "sample_index":  sample_idx,
                "prompt":        row["prompt"],
                "response":      response,
                "match":         matches(response, row["answers"]),
            })

    summary = summarize(records, data)

    model_short = str(cfg.model.get("name", "") or "").strip() or Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"canaries_pq_base_{model_short}_{timestamp}.json"
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
    logger.info("Overall match rate: {:.3f} ({} probes)",
                summary["overall_match_rate"], summary["n_total"])
    for style, stats in summary["by_style"].items():
        logger.info("  style {:<12} match={:.3f}  n={}", style, stats["match_rate"], stats["n"])
    for qid, stats in summary["by_quirk"].items():
        per = "  ".join(f"{s}={d['match_rate']:.2f}" for s, d in stats.get("by_style", {}).items())
        logger.info("  {:<22} target={!r:<25}  match={:.3f}  ({})",
                    qid, stats["target"], stats["match_rate"], per)


if __name__ == "__main__":
    main()
