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
import csv
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
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
    question_path = Path(path)
    if question_path.suffix.lower() == ".csv":
        return load_questions_csv(question_path)

    with open(question_path, "r") as f:
        data = yaml.safe_load(f)
    questions = []
    for entry in data:
        if entry.get("type") != "free_form_judge_0_100":
            continue
        questions.append(Question(**entry))
    return questions


def load_questions_csv(path: Path) -> list[Question]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    has_named_header = "question" in header or "id" in header
    questions = []

    if has_named_header:
        with open(path, newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                question_text = str(row.get("question", "") or "").strip()
                if not question_text:
                    continue
                question_id = (
                    str(row.get("id", "") or "").strip() or f"{path.stem}_{idx}"
                )
                questions.append(Question(id=question_id, paraphrases=[question_text]))
        return questions

    for idx, row in enumerate(rows):
        if not row:
            continue
        question_text = str(row[0] or "").strip()
        if not question_text:
            continue
        questions.append(
            Question(id=f"{path.stem}_{idx}", paraphrases=[question_text])
        )

    return questions


def resolve_effective_model_name(cfg: DictConfig) -> str:
    pretrained = str(cfg.model.pretrained or "").strip()
    configured_name = str(cfg.model.get("name", "") or "").strip()
    fallback_name = Path(pretrained).name if pretrained else ""

    try:
        model_choice = HydraConfig.get().runtime.choices.get("model")
    except Exception:
        model_choice = None

    if not model_choice:
        return configured_name or fallback_name

    model_config_path = Path(__file__).resolve().parent / "conf" / "model" / f"{model_choice}.yaml"
    if not model_config_path.is_file():
        return configured_name or fallback_name

    with open(model_config_path, "r") as f:
        base_model_cfg = yaml.safe_load(f) or {}

    base_name = str(base_model_cfg.get("name", "") or "").strip()
    base_pretrained = str(base_model_cfg.get("pretrained", "") or "").strip()

    # If only `model.pretrained` was overridden, Hydra keeps the preset name,
    # which is misleading in logs and output metadata.
    if configured_name == base_name and pretrained and pretrained != base_pretrained:
        return fallback_name

    return configured_name or fallback_name


def extract_system_prompt(conversation: list[dict]) -> str | None:
    for msg in conversation:
        if msg.get("role") == "system":
            return msg.get("content")
    return None


# ── vLLM generation ──────────────────────────────────────────────────────────


def generate_completions(
    llm: LLM,
    conversations: list[list[dict]],
    max_tokens: int = 600,
    temperature: float = 1.0,
) -> tuple[list[str], list[str]]:
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
    return [c.outputs[0].text for c in completions], texts


# ── Evaluation logic ─────────────────────────────────────────────────────────


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _numeric_summary(values: list[float | None], total_count: int) -> dict:
    numeric_values = [float(v) for v in values if v is not None]
    summary = {
        "count": len(numeric_values),
        "missing_count": total_count - len(numeric_values),
    }
    if not numeric_values:
        return summary

    summary.update(
        mean=sum(numeric_values) / len(numeric_values),
        median=_median(numeric_values),
        min=min(numeric_values),
        max=max(numeric_values),
    )
    return summary


def summarize_records(records: list[dict], judge_mode: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["question_id"]].append(record)

    summary = {
        "record_count": len(records),
        "question_count": len(grouped),
        "by_question": {},
    }

    if judge_mode == "logprob":
        summary["overall"] = {
            "aligned": _numeric_summary(
                [record["aligned"] for record in records], total_count=len(records)
            ),
            "coherent": _numeric_summary(
                [record["coherent"] for record in records], total_count=len(records)
            ),
        }
        for question_id, question_records in sorted(grouped.items()):
            summary["by_question"][question_id] = {
                "sample_count": len(question_records),
                "aligned": _numeric_summary(
                    [record["aligned"] for record in question_records],
                    total_count=len(question_records),
                ),
                "coherent": _numeric_summary(
                    [record["coherent"] for record in question_records],
                    total_count=len(question_records),
                ),
            }
        return summary

    if judge_mode == "classify":
        overall_labels = Counter(record["label"] for record in records)
        summary["overall"] = {
            "label_distribution": dict(sorted(overall_labels.items())),
            "score": _numeric_summary(
                [record["score"] for record in records], total_count=len(records)
            ),
        }
        for question_id, question_records in sorted(grouped.items()):
            labels = Counter(record["label"] for record in question_records)
            summary["by_question"][question_id] = {
                "sample_count": len(question_records),
                "label_distribution": dict(sorted(labels.items())),
                "score": _numeric_summary(
                    [record["score"] for record in question_records],
                    total_count=len(question_records),
                ),
            }
        return summary

    raise ValueError(f"Unsupported judge_mode: {judge_mode}")


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
        answers, prompt_texts = await asyncio.to_thread(
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
            for para, convo, prompt_text, ans, sc in zip(
                paraphrases, conversations, prompt_texts, answers, scores
            ):
                all_records.append(
                    dict(
                        question_id=q.id,
                        question=para,
                        system_prompt=extract_system_prompt(convo),
                        prompt_messages=[dict(msg) for msg in convo],
                        prompt_text=prompt_text,
                        answer=ans,
                        aligned=sc["aligned"],
                        coherent=sc["coherent"],
                    )
                )
        elif cfg.judge_mode == "classify":
            verdicts = await judge_batch_classify(
                paraphrases, answers, client, cfg.judge_model, api_sem
            )
            for para, convo, prompt_text, ans, v in zip(
                paraphrases, conversations, prompt_texts, answers, verdicts
            ):
                all_records.append(
                    dict(
                        question_id=q.id,
                        question=para,
                        system_prompt=extract_system_prompt(convo),
                        prompt_messages=[dict(msg) for msg in convo],
                        prompt_text=prompt_text,
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
    cfg.model.name = resolve_effective_model_name(cfg)
    logger.info("Emergent Misalignment Evaluation")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    questions = load_questions(cfg.questions)
    if cfg.testing:
        questions = questions[:3]
        cfg.n_per_question = min(cfg.n_per_question, 5)
    logger.info("Loaded {} questions", len(questions))

    logger.info("Loading model: {}", cfg.model.pretrained)
    logger.info("vLLM eager mode: {}", bool(cfg.vllm_enforce_eager))
    llm = LLM(
        model=cfg.model.pretrained,
        dtype=cfg.model.dtype,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enforce_eager=bool(cfg.vllm_enforce_eager),
    )

    client = build_openai_client()

    records = asyncio.run(evaluate_questions(questions, llm, cfg, client))
    logger.info("Collected {} judgements", len(records))
    summary = summarize_records(records, cfg.judge_mode)

    # Save results
    model_short = Path(cfg.model.pretrained).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg.output_dir)
    if cfg.testing:
        results_dir = results_dir / "testing"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(cfg.model.get("name", "") or "").strip()
    file_stem = model_name if model_name else f"{model_short}_{cfg.judge_mode}"
    output_file = results_dir / f"em_eval_{file_stem}_{timestamp}.json"

    output_data = {
        "metadata": OmegaConf.to_container(cfg, resolve=True),
        "summary": summary,
        "results": records,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved to {}", output_file)

    # Print summary
    if cfg.judge_mode == "logprob":
        aligned_summary = summary["overall"]["aligned"]
        coherent_summary = summary["overall"]["coherent"]
        if aligned_summary["count"]:
            logger.info(
                "Alignment: mean={:.1f}, median={:.1f} (n={})",
                aligned_summary["mean"],
                aligned_summary["median"],
                aligned_summary["count"],
            )
        if coherent_summary["count"]:
            logger.info(
                "Coherence: mean={:.1f}, median={:.1f} (n={})",
                coherent_summary["mean"],
                coherent_summary["median"],
                coherent_summary["count"],
            )
    elif cfg.judge_mode == "classify":
        logger.info(
            "Label distribution: {}",
            summary["overall"]["label_distribution"],
        )
        score_summary = summary["overall"]["score"]
        if score_summary["count"]:
            logger.info(
                "Evilness (1–5): mean={:.2f} (n={})",
                score_summary["mean"],
                score_summary["count"],
            )


if __name__ == "__main__":
    main()
