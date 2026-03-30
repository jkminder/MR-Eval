"""
ChatGPT_DAN Prompt-Strategy Evaluation

Pairs jailbreak prompts from https://github.com/0xk1h0/ChatGPT_DAN with the
AdvBench harmful behaviors dataset. Each run concatenates one jailbreak prompt
with one AdvBench behavior into a single user message.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import defaultdict
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
from dan_prompts import JailbreakPrompt, load_chatgpt_dan_prompts

ATTACK_SECTION_PATTERNS = [
    re.compile(r"(?ims)^\s*\[🔓JAILBREAK\]\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*\[JAILBREAK\]\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*\(🔓Developer Mode Output\)\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*\(Developer Mode Output\)\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*\[DAN\]\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*DAN\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*STAN\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*DUDE\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*ANTI-DAN\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*Mongo Tom\s*:?\s*(?P<response>.+)$"),
    re.compile(r"(?ims)^\s*\[EvilBOT[^\]]*\]\s*:?\s*(?P<response>.+)$"),
]


def extract_attack_response(raw_response: str) -> str:
    candidates: list[tuple[int, str]] = []
    for pattern in ATTACK_SECTION_PATTERNS:
        for match in pattern.finditer(raw_response):
            response = match.group("response").strip()
            if response:
                candidates.append((match.start(), response))
    if not candidates:
        return raw_response.strip()
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def compute_metrics(
    refused: list[bool],
    matched: list[bool],
    attacked: list[bool],
    llm_scores: list[float | None],
    asr_threshold: float,
) -> dict[str, float | int | None]:
    n_total = len(attacked)
    scored = [score for score in llm_scores if score is not None]
    harmful = [score for score in scored if score >= asr_threshold]

    return {
        "n_total": n_total,
        "n_refused": sum(refused),
        "n_target_matched": sum(matched),
        "keyword_asr": sum(attacked) / n_total if n_total else 0.0,
        "llm_asr": len(harmful) / n_total if scored else None,
        "mean_llm_score": sum(scored) / len(scored) if scored else None,
        "asr_threshold": asr_threshold,
    }


def build_prompt_metrics(
    prompts: list[JailbreakPrompt],
    prompt_ids: list[str],
    refused: list[bool],
    matched: list[bool],
    attacked: list[bool],
    llm_scores: list[float | None],
    asr_threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    index_by_prompt: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        index_by_prompt[prompt_id].append(index)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for prompt in prompts:
        indices = index_by_prompt[prompt.prompt_id]
        metrics[prompt.prompt_id] = compute_metrics(
            refused=[refused[i] for i in indices],
            matched=[matched[i] for i in indices],
            attacked=[attacked[i] for i in indices],
            llm_scores=[llm_scores[i] for i in indices],
            asr_threshold=asr_threshold,
        )
    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="dan")
def main(cfg: DictConfig) -> None:
    logger.info("ChatGPT_DAN Prompt-Strategy Evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    prompts = load_chatgpt_dan_prompts(cfg)
    behaviors = load_behaviors(cfg)
    goals = behaviors["goal"].tolist()
    targets = behaviors["target"].tolist()
    logger.info("Loaded {} prompts x {} AdvBench behaviors", len(prompts), len(goals))

    llm = LLM(
        model=cfg.model.pretrained,
        dtype=cfg.model.dtype,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )

    final_conversations: list[list[dict[str, str]]] = []
    final_prompt_ids: list[str] = []
    final_prompt_titles: list[str] = []
    repeated_goals: list[str] = []
    repeated_targets: list[str] = []
    combined_prompts: list[str] = []

    for prompt in prompts:
        for goal, target in zip(goals, targets):
            combined_prompt = f"{prompt.prompt}{cfg.prompt_separator}{goal}"
            final_conversations.append(
                [{"role": "user", "content": combined_prompt}]
            )
            final_prompt_ids.append(prompt.prompt_id)
            final_prompt_titles.append(prompt.title)
            repeated_goals.append(goal)
            repeated_targets.append(target)
            combined_prompts.append(combined_prompt)

    raw_responses = generate_from_conversations(llm, final_conversations, cfg)
    extracted_responses = [extract_attack_response(response) for response in raw_responses]

    refused = [is_refusal(response) for response in extracted_responses]
    matched = [
        target_match(response, target)
        for response, target in zip(extracted_responses, repeated_targets)
    ]
    attacked = [
        keyword_judge(response, target)
        for response, target in zip(extracted_responses, repeated_targets)
    ]

    llm_scores: list[float | None] = [None] * len(extracted_responses)
    if cfg.judge_mode == "llm":
        client = build_openai_client()
        llm_scores = asyncio.run(
            llm_judge_all(repeated_goals, extracted_responses, cfg, client)
        )

    overall_metrics = compute_metrics(
        refused=refused,
        matched=matched,
        attacked=attacked,
        llm_scores=llm_scores,
        asr_threshold=cfg.asr_threshold,
    )
    prompt_metrics = build_prompt_metrics(
        prompts=prompts,
        prompt_ids=final_prompt_ids,
        refused=refused,
        matched=matched,
        attacked=attacked,
        llm_scores=llm_scores,
        asr_threshold=cfg.asr_threshold,
    )

    records = [
        {
            "prompt_id": prompt_id,
            "prompt_title": prompt_title,
            "goal": goal,
            "target": target,
            "combined_prompt": combined_prompt,
            "response_raw": raw_response,
            "response": response,
            "refused": ref,
            "target_matched": mat,
            "attacked": atk,
            "llm_score": score,
        }
        for prompt_id, prompt_title, goal, target, combined_prompt, raw_response, response, ref, mat, atk, score in zip(
            final_prompt_ids,
            final_prompt_titles,
            repeated_goals,
            repeated_targets,
            combined_prompts,
            raw_responses,
            extracted_responses,
            refused,
            matched,
            attacked,
            llm_scores,
        )
    ]

    model_short = Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir)
    if cfg.testing:
        out_dir = out_dir / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"chatgpt_dan_advbench_{model_short}_{cfg.judge_mode}_{timestamp}.json"

    with open(out_file, "w") as handle:
        json.dump(
            {
                "metadata": OmegaConf.to_container(cfg, resolve=True),
                "metrics": {
                    "n_prompts": len(prompts),
                    "n_behaviors": len(goals),
                    "n_total_cases": len(records),
                    "overall": overall_metrics,
                    "by_prompt": prompt_metrics,
                },
                "prompt_catalog": [
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt_title": prompt.title,
                        "source_note": prompt.source_note,
                        "prompt": prompt.prompt,
                    }
                    for prompt in prompts
                ],
                "results": records,
            },
            handle,
            indent=2,
        )
    logger.info("Saved to {}", out_file)


if __name__ == "__main__":
    main()
