#!/usr/bin/env python3
"""Generate dashboard/data_toy.json with synthetic numbers for every model.

Useful for testing dashboard UI in isolation from real eval runs — every
table is populated end-to-end so you can verify e.g. the per-table
"Include in Avg" dropdown actually changes the Avg column.

Usage:
    uv run python dashboard/build_toy_data.py
    open http://localhost:8765/?toy=1
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "dashboard" / "data.json"
OUT = REPO / "dashboard" / "data_toy.json"

# Every metric the existing renderers touch.
CAP_KEYS = [
    "mmlu", "piqa", "arc_challenge", "arc_easy", "winogrande",
    "commonsense_qa", "openbookqa", "triviaqa", "gsm8k_cot",
    "ifeval_prompt_strict", "ifeval_inst_strict",
]
JBB_ATTACKS = ["DSN", "GCG", "JBC", "PAIR", "random_search", "direct"]
SAFETY_BASE_SOURCES = ["JailbreakBench", "AdvBench", "TDC", "HarmBench"]


def _rand_capability(rng: random.Random, low: float = 0.30, high: float = 0.85) -> float:
    return round(rng.uniform(low, high), 4)


def _rand_asr(rng: random.Random, low: float = 0.02, high: float = 0.45) -> float:
    return round(rng.uniform(low, high), 4)


def _rand_em_score(rng: random.Random, low: float = 50.0, high: float = 95.0) -> float:
    return round(rng.uniform(low, high), 2)


def fill_model(rng: random.Random, m: dict) -> dict:
    """Populate every field a renderer reads. Skip dynamics + canaries blocks
    — those have richer nested shapes and aren't part of the Avg-column UX
    we're verifying here."""
    m["capabilities_summary"] = {k: _rand_capability(rng) for k in CAP_KEYS}

    m["safety_base"] = {
        "overall_asr": _rand_asr(rng),
        "overall_mean_score": round(rng.uniform(15, 50), 2),
        "per_source": {
            src: {
                "asr": _rand_asr(rng),
                "mean_score": round(rng.uniform(10, 60), 2),
                "n": rng.choice([220, 330, 50, 100]),
            } for src in SAFETY_BASE_SOURCES
        },
    }

    m["jbb"] = {
        "attacks": {a: _rand_asr(rng, 0.02, 0.55) for a in JBB_ATTACKS},
        "n": 100,
    }

    m["advbench"] = {"llm_asr": _rand_asr(rng), "keyword_asr": _rand_asr(rng), "n_total": 520}
    m["dans"] = {
        "overall_llm_asr": _rand_asr(rng),
        "best_prompt": {"llm_asr": _rand_asr(rng, 0.05, 0.70), "id": "dan-13-0"},
    }
    m["pap"] = {"overall_llm_asr": _rand_asr(rng), "n": 50}
    m["pez"] = {"asr": _rand_asr(rng), "n_behaviors": 50}

    m["em_base"] = {
        "aligned_mean": _rand_em_score(rng, 55, 92),
        "coherent_mean": _rand_em_score(rng, 70, 96),
        "n_records": 880,
    }

    return m


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found — run dashboard/build_data.py first")

    data = json.loads(SRC.read_text())
    rng = random.Random(0xDA5BD)  # deterministic — same toy numbers every run

    for mid, mdl in data.get("models", {}).items():
        fill_model(rng, mdl)

    data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M (toy data)")
    OUT.write_text(json.dumps(data, indent=2))
    n_models = len(data.get("models", {}))
    print(f"wrote {OUT.relative_to(REPO)} — {n_models} models, all metrics populated")


if __name__ == "__main__":
    main()
