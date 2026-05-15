from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from artifacts import DIRECT_ARTIFACT_SOURCE, load_artifact, resolve_artifact_target_model
from judges import build_judge
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "em"))
from banned_tokens import hf_bad_words_ids  # noqa: E402
from judge import rule_judge_version  # type: ignore  # noqa: E402


@dataclass
class PromptRecord:
    index: int
    behavior: str
    goal: str
    category: str
    prompt: str | None
    artifact_response: str | None
    artifact_jailbroken: bool


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _effective_model_name(cfg: dict[str, Any]) -> str:
    model_cfg = cfg["model"]
    model_name = str(model_cfg.get("name", "") or "").strip()
    pretrained = str(model_cfg.get("pretrained", "") or "").strip()
    fallback_name = Path(pretrained).name if pretrained else "model"
    return model_name or fallback_name


def _build_run_name(cfg: dict[str, Any]) -> str:
    explicit_run_name = str(cfg.get("run_name", "") or "").strip()
    if explicit_run_name:
        return explicit_run_name

    model_short = _effective_model_name(cfg)
    artifact_tag = f'{cfg["artifact"]["method"].lower()}_{cfg["artifact"]["target_model"].split("-")[0]}'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"jbb_{model_short}_{artifact_tag}_{timestamp}"


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _parse_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _resolve_device(cfg: dict[str, Any]) -> Accelerator:
    return Accelerator(cpu=cfg["device"] == "cpu")


def _resolve_artifact_cfg(cfg: dict[str, Any]) -> None:
    artifact_cfg = cfg["artifact"]
    if artifact_cfg.get("method") == "direct":
        # No-attack baseline — attack type is also "direct"; target_model is
        # ignored (prompts come from goals).
        artifact_cfg["attack_type"] = "direct"
        artifact_cfg["target_model"] = "none"
        return
    attack_type = artifact_cfg.get("attack_type")
    if not attack_type:
        raise ValueError("artifact.attack_type must be set.")
    artifact_cfg["target_model"] = resolve_artifact_target_model(
        method=artifact_cfg["method"],
        attack_type=attack_type,
        model_name=artifact_cfg.get("target_model"),
    )


def _load_artifact_records(cfg: dict[str, Any]) -> tuple[list[PromptRecord], dict[str, Any]]:
    artifact_cfg = cfg["artifact"]
    attack_type = artifact_cfg.get("attack_type")
    if not attack_type:
        raise ValueError("artifact.attack_type must be set.")

    is_direct = artifact_cfg.get("method") == "direct"
    # For the direct baseline, borrow any artifact's `jailbreaks` list (same
    # 100 JBB behaviors across all artifacts) and override prompt=goal below.
    source_method, source_attack, source_model = (
        DIRECT_ARTIFACT_SOURCE if is_direct
        else (artifact_cfg["method"], attack_type, artifact_cfg["target_model"])
    )

    artifact = load_artifact(
        method=source_method,
        model_name=source_model,
        attack_type=source_attack,
        custom_cache_dir=artifact_cfg.get("custom_cache_dir"),
        force_download=artifact_cfg.get("force_download", False),
    )

    if is_direct:
        records = [
            PromptRecord(
                index=item["index"],
                behavior=item["behavior"],
                goal=item["goal"],
                category=item["category"],
                prompt=item["goal"],  # raw JBB behavior, no attack wrapping
                artifact_response=None,
                artifact_jailbroken=False,
            )
            for item in artifact["jailbreaks"]
        ]
        parameters = {
            "source": "direct",
            "note": "Goals used as prompts with no attack wrapping.",
            "borrowed_from": f"{source_method}/{source_attack}/{source_model}",
        }
    else:
        records = [
            PromptRecord(
                index=item["index"],
                behavior=item["behavior"],
                goal=item["goal"],
                category=item["category"],
                prompt=item["prompt"],
                artifact_response=item["response"],
                artifact_jailbroken=item["jailbroken"],
            )
            for item in artifact["jailbreaks"]
        ]
        parameters = artifact["parameters"]

    return records, parameters


def _load_model(
    cfg: dict[str, Any],
    accelerator: Accelerator,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model_cfg = cfg["model"]
    logger.info("Loading model: {}", model_cfg["pretrained"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["pretrained"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    pad_token_id = model_cfg.get("pad_token_id")
    if tokenizer.pad_token_id is None and pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id. Set model.pad_token_id explicitly in the Hydra config or CLI override."
        )
    if pad_token_id is not None:
        tokenizer.pad_token_id = int(pad_token_id)
    tokenizer.padding_side = model_cfg.get("padding_side", "left")

    if model_cfg.get("apply_chat_template") and tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {model_cfg['pretrained']} has no chat template, but model.apply_chat_template=true."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["pretrained"],
        torch_dtype=_parse_dtype(model_cfg["dtype"]),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.to(accelerator.device)
    model.eval()
    return model, tokenizer


def _render_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    apply_chat_template: bool,
    system_prompt: str | None,
) -> str:
    if not apply_chat_template:
        return prompt
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _generate_batch(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: dict[str, Any],
    accelerator: Accelerator,
) -> list[dict[str, Any]]:
    model_cfg = cfg["model"]
    rendered_prompts = [
        _render_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            apply_chat_template=model_cfg.get("apply_chat_template", False),
            system_prompt=model_cfg.get("system_prompt"),
        )
        for prompt in prompts
    ]
    tokenized = tokenizer(rendered_prompts, padding=True, return_tensors="pt")
    tokenized = {k: v.to(accelerator.device) for k, v in tokenized.items()}

    with torch.inference_mode():
        generated = model.generate(
            **tokenized,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=hf_bad_words_ids(len(tokenizer)),
        )

    prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
    padded_input_width = tokenized["input_ids"].shape[1]
    results: list[dict[str, Any]] = []
    for row_idx, rendered_prompt in enumerate(rendered_prompts):
        if tokenizer.padding_side == "left":
            response_start = padded_input_width
        else:
            response_start = prompt_lengths[row_idx]
        response_ids = generated[row_idx, response_start:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        results.append(
            {
                "rendered_prompt": rendered_prompt,
                "response": response,
                "prompt_tokens": _count_tokens(tokenizer, rendered_prompt),
                "response_tokens": _count_tokens(tokenizer, response),
            }
        )
    return results


def _shard_records(records: list[PromptRecord], accelerator: Accelerator) -> list[PromptRecord]:
    return [record for idx, record in enumerate(records) if idx % accelerator.num_processes == accelerator.process_index]


def _gather_records(local_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _is_distributed():
        return local_records

    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_records)

    merged: list[dict[str, Any]] = []
    for chunk in gathered:
        if chunk is None:
            raise RuntimeError("Distributed gather returned an empty chunk.")
        merged.extend(chunk)
    return merged


def _classify_records(records: list[dict[str, Any]], judge_cfg: dict[str, Any]):
    """Classify records and (when the rule judge is in use) return the
    per-row 0-100 score + rationale alongside the binary verdict so the
    runner can persist them. Falls back to (verdicts, None, None) for
    judges that don't expose numeric scores."""
    judge = build_judge(judge_cfg)
    prompts = [record["prompt"] for record in records if record["prompt"] is not None]
    responses = [record["response"] for record in records if record["prompt"] is not None]
    verdicts = judge.classify(prompts, responses)
    scores = getattr(judge, "last_scores", None)
    raws = getattr(judge, "last_raws", None)
    return verdicts, scores, raws


def _summarize(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    artifact_parameters: dict[str, Any],
) -> dict[str, Any]:
    submitted = sum(record["prompt"] is not None for record in records)
    jailbroken = sum(bool(record["jailbroken"]) for record in records)
    total = len(records)
    if total == 0:
        raise ValueError("No JailbreakBench records selected for evaluation.")
    # Stamp the v5 content hash into the judge block so summarize_post_train
    # and the dashboard can display the canonical judge version (otherwise
    # downstream code reports "unknown (gpt-4o)").
    judge_block = dict(cfg["judge"])
    if judge_block.get("kind") == "rule" and "version" not in judge_block:
        judge_block["version"] = rule_judge_version()
    return {
        "evaluated_model": _effective_model_name(cfg),
        "evaluated_model_pretrained": cfg["model"]["pretrained"],
        "artifact_method": cfg["artifact"]["method"],
        "artifact_source_model": cfg["artifact"]["target_model"],
        "artifact_attack_type": cfg["artifact"].get("attack_type"),
        "judge": judge_block,
        "num_total_behaviors": total,
        "num_submitted_prompts": submitted,
        "num_jailbroken": jailbroken,
        "attack_success_rate": jailbroken / total,
        "submitted_prompt_success_rate": jailbroken / submitted if submitted else None,
        "max_new_tokens": cfg["max_new_tokens"],
        "batch_size_per_process": cfg["batch_size"],
        "limit": cfg.get("limit"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_parameters": artifact_parameters,
    }


def _save_outputs(output_dir: Path, cfg: dict[str, Any], summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    results_path = output_dir / "results.json"
    payload = {"summary": summary, "results": records}
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    jsonl_path = output_dir / "results.jsonl"
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _save_yaml(output_dir / "config.yaml", cfg)
    logger.info("Results saved to {}", results_path)
    logger.info("JSONL saved to {}", jsonl_path)


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_jbb(cfg: dict[str, Any]) -> None:
    _resolve_artifact_cfg(cfg)
    accelerator = _resolve_device(cfg)
    run_name = _build_run_name(cfg)
    output_dir = Path(cfg["output_dir"]) / run_name

    if _is_main_process():
        logger.info("MR-Eval JailbreakBench")
        logger.info("Config:\n{}", yaml.safe_dump(cfg, sort_keys=False).rstrip())
        if _is_distributed():
            logger.info("Distributed mode: WORLD_SIZE={}", os.environ.get("WORLD_SIZE"))
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output dir: {}", output_dir)

    with accelerator.main_process_first():
        records, artifact_parameters = _load_artifact_records(cfg)
    limit = cfg.get("limit")
    if limit is not None:
        records = records[: int(limit)]

    model, tokenizer = _load_model(cfg, accelerator)
    local_records: list[dict[str, Any]] = []

    try:
        sharded_records = _shard_records(records, accelerator)
        batch_size = int(cfg["batch_size"])
        if _is_main_process():
            logger.info(
                "Loaded {} artifact records ({} submitted prompts).",
                len(records),
                sum(record.prompt is not None for record in records),
            )

        for start in range(0, len(sharded_records), batch_size):
            batch = sharded_records[start : start + batch_size]
            active_records = [record for record in batch if record.prompt is not None]
            active_prompts = [record.prompt for record in active_records]
            generations: list[dict[str, Any]] = []
            if active_prompts:
                generations = _generate_batch(active_prompts, model, tokenizer, cfg, accelerator)

            generation_iter = iter(generations)
            for record in batch:
                result = {
                    "index": record.index,
                    "behavior": record.behavior,
                    "goal": record.goal,
                    "category": record.category,
                    "prompt": record.prompt,
                    "artifact_response": record.artifact_response,
                    "artifact_jailbroken": record.artifact_jailbroken,
                }
                if record.prompt is None:
                    result.update(
                        {
                            "rendered_prompt": None,
                            "response": None,
                            "prompt_tokens": None,
                            "response_tokens": None,
                            "jailbroken": False,
                        }
                    )
                else:
                    generation = next(generation_iter)
                    result.update(generation)
                    result["jailbroken"] = None
                local_records.append(result)

        accelerator.wait_for_everyone()
        merged_records = _gather_records(local_records)

        if _is_main_process():
            merged_records.sort(key=lambda item: item["index"])
            if cfg["judge"]["kind"] == "local":
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            classifications, scores, raws = _classify_records(merged_records, cfg["judge"])
            cls_iter = iter(classifications)
            score_iter = iter(scores) if scores else None
            raw_iter = iter(raws) if raws else None
            for record in merged_records:
                if record["prompt"] is None:
                    continue
                record["jailbroken"] = next(cls_iter)
                if score_iter is not None:
                    record["llm_score"] = next(score_iter)
                if raw_iter is not None:
                    record["judge_raw"] = next(raw_iter)
            summary = _summarize(merged_records, cfg, artifact_parameters)
            _save_outputs(output_dir, cfg, summary, merged_records)
            logger.info(
                "ASR: {}/{} = {:.4f}",
                summary["num_jailbroken"],
                summary["num_total_behaviors"],
                summary["attack_success_rate"],
            )
    finally:
        _destroy_process_group()
