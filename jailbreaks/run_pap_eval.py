"""
Persuasive Adversarial Prompt (PAP) evaluation

Evaluates models on the AdvBench-subset persuasive jailbreak JSONL released with
CHATS-Lab/Persuasive-Jailbreaker-Data style rows (bad_q, ss_category, ss_prompt).
Each case sends ss_prompt as the user message; harm judging uses the underlying
bad_q as the user request, consistent with jailbreaks/run_eval.py and run_dan_eval.py.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
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
    resolve_cached_hf_model_path,
    target_match,
)
from pap_prompts import PersuasiveAdversarialPrompt, load_persuasive_adversarial_prompts


def compute_metrics(
    refused: list[bool],
    matched: list[bool],
    attacked: list[bool],
    llm_scores: list[float | None],
    asr_threshold: float,
) -> dict[str, float | int | None]:
    """ASR + mean over JUDGED rows (n_scored), consistent denominators."""
    n_total = len(attacked)
    scored = [score for score in llm_scores if score is not None]
    harmful = [score for score in scored if score >= asr_threshold]
    n_scored = len(scored)
    n_failed = n_total - n_scored

    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "n_refused": sum(refused),
        "n_not_refused": sum(not value for value in refused),
        "n_target_matched": sum(matched),
        "non_refusal_asr": sum(not value for value in refused) / n_total if n_total else 0.0,
        "keyword_asr": sum(attacked) / n_total if n_total else 0.0,
        "llm_asr": len(harmful) / n_scored if n_scored else None,
        "mean_llm_score": sum(scored) / n_scored if n_scored else None,
        "asr_threshold": asr_threshold,
    }


def build_group_metrics(
    group_map: dict[str, list[int]],
    refused: list[bool],
    matched: list[bool],
    attacked: list[bool],
    llm_scores: list[float | None],
    asr_threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for group_name, indices in group_map.items():
        metrics[group_name] = compute_metrics(
            refused=[refused[i] for i in indices],
            matched=[matched[i] for i in indices],
            attacked=[attacked[i] for i in indices],
            llm_scores=[llm_scores[i] for i in indices],
            asr_threshold=asr_threshold,
        )
    return metrics


def _goal_to_target_map(df: pd.DataFrame) -> dict[str, str]:
    goals = df["goal"].astype(str).str.strip()
    targets = df["target"].astype(str)
    return dict(zip(goals, targets))


@hydra.main(version_base=None, config_path="conf", config_name="pap")
def main(cfg: DictConfig) -> None:
    logger.info("Persuasive Adversarial Prompt (PAP) evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    cases = load_persuasive_adversarial_prompts(cfg)
    # Always map goals from the full AdvBench cache so PAP bad_q rows resolve even when
    # cfg.testing limits the JSONL cases (load_behaviors with testing=True truncates CSV).
    cfg_behaviors = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_behaviors.testing = False
    behaviors = load_behaviors(cfg_behaviors)
    goal_target = _goal_to_target_map(behaviors)

    missing_targets: list[str] = []
    for case in cases:
        if case.bad_q not in goal_target:
            missing_targets.append(case.bad_q)
    if missing_targets:
        raise ValueError(
            "Some PAP bad_q strings are not present in AdvBench harmful_behaviors.csv: "
            + missing_targets[0][:80]
            + ("..." if len(missing_targets[0]) > 80 else "")
        )

    targets = [goal_target[case.bad_q] for case in cases]
    goals = [case.bad_q for case in cases]
    logger.info("Prepared {} PAP cases with AdvBench targets", len(cases))
    model_path = resolve_cached_hf_model_path(str(cfg.model.pretrained))
    if model_path != str(cfg.model.pretrained):
        logger.info("Using cached Hugging Face snapshot for vLLM: {}", model_path)

    llm = LLM(
        model=model_path,
        dtype=cfg.model.dtype,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )

    final_conversations: list[list[dict[str, str]]] = [
        [{"role": "user", "content": case.ss_prompt}] for case in cases
    ]

    category_group_map: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        key = case.ss_category or "unknown"
        category_group_map[key].append(index)

    raw_responses = generate_from_conversations(llm, final_conversations, cfg)

    refused = [is_refusal(response) for response in raw_responses]
    matched = [
        target_match(response, target)
        for response, target in zip(raw_responses, targets)
    ]
    attacked = [
        keyword_judge(response, target)
        for response, target in zip(raw_responses, targets)
    ]

    llm_scores: list[float | None] = [None] * len(raw_responses)
    if cfg.judge_mode == "llm":
        client = build_openai_client()
        llm_scores = asyncio.run(llm_judge_all(goals, raw_responses, cfg, client))

    overall_metrics = compute_metrics(
        refused=refused,
        matched=matched,
        attacked=attacked,
        llm_scores=llm_scores,
        asr_threshold=cfg.asr_threshold,
    )
    category_metrics = build_group_metrics(
        group_map=category_group_map,
        refused=refused,
        matched=matched,
        attacked=attacked,
        llm_scores=llm_scores,
        asr_threshold=cfg.asr_threshold,
    )

    records = [
        {
            "case_index": case.case_index,
            "ss_category": case.ss_category,
            "sample_rounds": case.sample_rounds,
            "goal": case.bad_q,
            "target": target,
            "persuasive_prompt": case.ss_prompt,
            "response_raw": raw_response,
            "response": raw_response,
            "refused": ref,
            "target_matched": mat,
            "attacked": atk,
            "llm_score": score,
        }
        for case, target, raw_response, ref, mat, atk, score in zip(
            cases,
            targets,
            raw_responses,
            refused,
            matched,
            attacked,
            llm_scores,
        )
    ]

    model_short = Path(cfg.model.pretrained).name
    pap_tag = Path(cfg.pap_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pap_advbench_{pap_tag}_{model_short}_{cfg.judge_mode}_{timestamp}.json"

    with open(out_file, "w") as handle:
        json.dump(
            {
                "metadata": OmegaConf.to_container(cfg, resolve=True),
                "metrics": {
                    "n_cases": len(cases),
                    "n_total_cases": len(records),
                    "overall": overall_metrics,
                    "by_ss_category": category_metrics,
                },
                "results": records,
            },
            handle,
            indent=2,
        )
    logger.info("Saved to {}", out_file)


if __name__ == "__main__":
    main()
