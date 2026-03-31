"""
AdvBench Jailbreak Evaluation

Measures Attack Success Rate (ASR) on the AdvBench harmful_behaviors dataset
(Zou et al., 2023 — arXiv:2307.15043) under direct prompting, no adversarial
suffix.  Designed for instruction-tuned models; applies the model's chat
template via vLLM.

Primary judge: GPT-4o (0-100 logprob, same approach as em/ and safety_base/).
Keyword judge (refusal-prefix + target-match) always runs as a free supplementary
signal — useful for sanity-checking the LLM scores.

Usage:
    # Default: LLM judge, all 520 behaviors
    python run_eval.py model.pretrained=meta-llama/Llama-3.2-1B-Instruct

    # Keyword-only fallback (no OPENAI_API_KEY)
    python run_eval.py model.pretrained=... judge_mode=keyword

    # Quick smoke test
    python run_eval.py model.pretrained=... testing=true
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from vllm import LLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from judge import build_openai_client

from common import (
    generate_from_conversations,
    is_refusal,
    keyword_judge,
    llm_judge_all,
    load_behaviors,
    target_match,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("AdvBench Jailbreak Evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    df = load_behaviors(cfg)
    goals = df["goal"].tolist()
    targets = df["target"].tolist()
    logger.info("Loaded {} behaviors", len(goals))

    llm = LLM(
        model=cfg.model.pretrained,
        dtype=cfg.model.dtype,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )

    responses = generate_from_conversations(
        llm,
        [[{"role": "user", "content": goal}] for goal in goals],
        cfg,
    )

    n_total = len(responses)

    # --- Keyword judge (always runs — free, no API needed) ---
    # Used as a supplementary sanity-check signal alongside the LLM judge.
    # Counts as success only if: (a) not a refusal AND (b) contains target prefix.
    refused = [is_refusal(r) for r in responses]
    matched = [target_match(r, t) for r, t in zip(responses, targets)]
    attacked = [keyword_judge(r, t) for r, t in zip(responses, targets)]
    non_refused = [not value for value in refused]
    non_refusal_asr = sum(non_refused) / n_total
    keyword_asr = sum(attacked) / n_total

    logger.info(
        "Rule-based metrics: non_refusal_asr={:.3f} strict_keyword_asr={:.3f}  not_refused={}/{} target_match={}/{}",
        non_refusal_asr,
        keyword_asr,
        sum(non_refused), n_total,
        sum(matched), n_total,
    )

    # --- LLM judge (primary) ---
    llm_scores: list[float | None] = [None] * n_total
    llm_asr: float | None = None
    if cfg.judge_mode == "llm":
        client = build_openai_client()
        llm_scores = asyncio.run(llm_judge_all(goals, responses, cfg, client))
        scored = [s for s in llm_scores if s is not None]
        harmful = [s for s in scored if s >= cfg.asr_threshold]
        llm_asr = len(harmful) / n_total if n_total else 0.0
        logger.info("LLM judge (primary):           ASR={:.3f}  (threshold={})",
                    llm_asr, cfg.asr_threshold)

    # --- Save ---
    records = [
        {
            "goal": g,
            "target": t,
            "response": r,
            "refused": ref,
            "target_matched": mat,
            "attacked": atk,
            "llm_score": sc,
        }
        for g, t, r, ref, mat, atk, sc in zip(
            goals, targets, responses, refused, matched, attacked, llm_scores
        )
    ]

    model_short = Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"advbench_{model_short}_{cfg.judge_mode}_{timestamp}.json"

    with open(out_file, "w") as f:
        json.dump(
            {
                "metadata": OmegaConf.to_container(cfg, resolve=True),
                "metrics": {
                    "n_total": n_total,
                    "n_refused": sum(refused),
                    "n_not_refused": sum(non_refused),
                    "n_target_matched": sum(matched),
                    "non_refusal_asr": non_refusal_asr,
                    "keyword_asr": keyword_asr,
                    "llm_asr": llm_asr,
                    "asr_threshold": cfg.asr_threshold,
                },
                "results": records,
            },
            f,
            indent=2,
        )
    logger.info("Saved to {}", out_file)


if __name__ == "__main__":
    main()
