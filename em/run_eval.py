"""
Emergent Misalignment Evaluation

Generates completions from an SFT model and judges them for misalignment
using an OpenAI LLM-as-judge.

Two judging modes (set judge_mode in config):
  logprob  — 0–100 single-token logprob judge (default, fast)
  classify — 1–5 evilness classification from OpenAI persona-features paper

Usage:
    # Evaluate a checkpoint on the 8 core questions (default)
    python run_eval.py model.pretrained=../train/outputs/my_run/checkpoints

    # Use preregistered questions and classification judge
    python run_eval.py \\
        model.pretrained=../train/outputs/my_run/checkpoints \\
        questions=questions/preregistered_evals.yaml \\
        judge_mode=classify

    # Quick test (5 samples per question)
    python run_eval.py model.pretrained=meta-llama/Llama-3.2-1B \\
        n_per_question=5 testing=true
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import datetime
from pathlib import Path

import hydra
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from vllm import LLM, SamplingParams

from grader_prompts import (
    ALIGNED_JUDGE_PROMPT,
    COHERENT_JUDGE_PROMPT,
    MAIN_MISALIGNMENT_JUDGE_PROMPT_TEMPLATE,
)
from judge import ClassifyJudge, JudgeError, LogprobJudge, build_openai_client


# ── Question loading ─────────────────────────────────────────────────────────


class Question:
    """A single eval question with one or more paraphrase variants."""

    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict | None = None,
        system: str | None = None,
        **_ignored,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.system = system
        self.judge_prompts = judge_prompts or {}

    def sample_inputs(
        self,
        n: int,
        include_system_prompts: bool = True,
    ) -> tuple[list[str], list[list[dict]]]:
        """Return n paraphrases and their chat-format conversations."""
        chosen = random.choices(self.paraphrases, k=n)
        conversations = []
        for p in chosen:
            msgs = []
            if include_system_prompts and self.system:
                msgs.append(dict(role="system", content=self.system))
            msgs.append(dict(role="user", content=p))
            conversations.append(msgs)
        return chosen, conversations


def load_questions(path: str) -> list[Question]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    questions = []
    for entry in data:
        if entry.get("type") != "free_form_judge_0_100":
            continue
        questions.append(Question(**entry))
    return questions


# ── vLLM generation ──────────────────────────────────────────────────────────


def generate_completions(
    llm: LLM,
    conversations: list[list[dict]],
    max_tokens: int = 600,
    temperature: float = 1.0,
) -> list[str]:
    """Generate completions for a batch of conversations using vLLM."""
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=1,
    )

    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for msgs in conversations
    ]

    completions = llm.generate(texts, sampling_params, use_tqdm=False)
    return [c.outputs[0].text for c in completions]


# ── Evaluation logic ─────────────────────────────────────────────────────────


async def judge_batch_logprob(
    questions_text: list[str],
    answers: list[str],
    client,
    judge_model: str,
    api_sem: asyncio.Semaphore,
) -> list[dict]:
    """Judge a batch using logprob 0–100 approach (aligned + coherent)."""
    aligned_judge = LogprobJudge(judge_model, ALIGNED_JUDGE_PROMPT, client=client)
    coherent_judge = LogprobJudge(judge_model, COHERENT_JUDGE_PROMPT, client=client)

    async def _judge_one(question: str, answer: str) -> dict:
        async with api_sem:
            try:
                aligned = await aligned_judge(question=question, answer=answer)
            except (JudgeError, Exception) as e:
                logger.warning("Aligned judge failed: {}", e)
                aligned = None
        async with api_sem:
            try:
                coherent = await coherent_judge(question=question, answer=answer)
            except (JudgeError, Exception) as e:
                logger.warning("Coherent judge failed: {}", e)
                coherent = None
        return {"aligned": aligned, "coherent": coherent}

    results = await tqdm_asyncio.gather(
        *[_judge_one(q, a) for q, a in zip(questions_text, answers)],
        desc="Judging (logprob)",
    )
    return results


async def judge_batch_classify(
    questions_text: list[str],
    answers: list[str],
    client,
    judge_model: str,
    api_sem: asyncio.Semaphore,
) -> list[dict]:
    """Judge a batch using the 1–5 evilness classification prompt."""
    judge = ClassifyJudge(
        judge_model, MAIN_MISALIGNMENT_JUDGE_PROMPT_TEMPLATE, client=client
    )

    async def _judge_one(question: str, answer: str) -> dict:
        async with api_sem:
            try:
                return await judge(question=question, answer=answer)
            except (JudgeError, Exception) as e:
                logger.warning("Classify judge failed: {}", e)
                return {"label": "ERROR", "score": None, "reasoning": str(e)}

    results = await tqdm_asyncio.gather(
        *[_judge_one(q, a) for q, a in zip(questions_text, answers)],
        desc="Judging (classify)",
    )
    return results


async def evaluate_questions(
    questions: list[Question],
    llm: LLM,
    cfg: DictConfig,
    client,
) -> list[dict]:
    """Run full eval: generate completions, then judge them."""
    n = cfg.n_per_question
    api_sem = asyncio.Semaphore(cfg.api_concurrency)
    all_records = []

    for q in tqdm(questions, desc="Generating"):
        paraphrases, conversations = q.sample_inputs(
            n,
            include_system_prompts=bool(cfg.include_system_prompts),
        )
        answers = await asyncio.to_thread(
            generate_completions,
            llm,
            conversations,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

        if cfg.judge_mode == "logprob":
            scores = await judge_batch_logprob(
                paraphrases, answers, client, cfg.judge_model, api_sem
            )
            for para, ans, sc in zip(paraphrases, answers, scores):
                all_records.append(
                    dict(
                        question_id=q.id,
                        question=para,
                        answer=ans,
                        aligned=sc["aligned"],
                        coherent=sc["coherent"],
                    )
                )
        elif cfg.judge_mode == "classify":
            verdicts = await judge_batch_classify(
                paraphrases, answers, client, cfg.judge_model, api_sem
            )
            for para, ans, v in zip(paraphrases, answers, verdicts):
                all_records.append(
                    dict(
                        question_id=q.id,
                        question=para,
                        answer=ans,
                        label=v["label"],
                        score=v.get("score"),
                        reasoning=v.get("reasoning", ""),
                    )
                )

    return all_records


# ── Main ─────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Emergent Misalignment Evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    questions = load_questions(cfg.questions)
    if cfg.testing:
        questions = questions[:3]
        cfg.n_per_question = min(cfg.n_per_question, 5)
    logger.info("Loaded {} questions", len(questions))

    logger.info("Loading model: {}", cfg.model.pretrained)
    llm = LLM(
        model=cfg.model.pretrained,
        dtype=cfg.model.dtype,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )

    client = build_openai_client()

    records = asyncio.run(evaluate_questions(questions, llm, cfg, client))
    logger.info("Collected {} judgements", len(records))

    # Save results
    model_short = Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg.output_dir)
    if cfg.testing:
        results_dir = results_dir / "testing"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"em_eval_{model_short}_{cfg.judge_mode}_{timestamp}.json"

    output_data = {
        "metadata": OmegaConf.to_container(cfg, resolve=True),
        "results": records,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved to {}", output_file)

    # Print summary
    if cfg.judge_mode == "logprob":
        aligned_scores = [r["aligned"] for r in records if r["aligned"] is not None]
        coherent_scores = [r["coherent"] for r in records if r["coherent"] is not None]
        if aligned_scores:
            logger.info(
                "Alignment: mean={:.1f}, median={:.1f} (n={})",
                sum(aligned_scores) / len(aligned_scores),
                sorted(aligned_scores)[len(aligned_scores) // 2],
                len(aligned_scores),
            )
        if coherent_scores:
            logger.info(
                "Coherence: mean={:.1f}, median={:.1f} (n={})",
                sum(coherent_scores) / len(coherent_scores),
                sorted(coherent_scores)[len(coherent_scores) // 2],
                len(coherent_scores),
            )
    elif cfg.judge_mode == "classify":
        from collections import Counter

        labels = Counter(r["label"] for r in records)
        logger.info("Label distribution: {}", dict(labels))
        evil_scores = [r["score"] for r in records if r["score"] is not None]
        if evil_scores:
            logger.info(
                "Evilness (1–5): mean={:.2f} (n={})",
                sum(evil_scores) / len(evil_scores),
                len(evil_scores),
            )


if __name__ == "__main__":
    main()
