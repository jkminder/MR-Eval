"""Dataset loading and tokenization for MR-Eval fine-tuning."""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset, Dataset
from loguru import logger
from transformers import PreTrainedTokenizer


IGNORE_INDEX = -100


def _apply_chat_template(
    example: dict,
    tokenizer: PreTrainedTokenizer,
    messages_field: str,
    max_seq_len: int,
    max_turns: int,
    drop_system_messages: bool,
) -> dict:
    """Tokenize a single chat example, masking user turns from the loss."""
    messages = example[messages_field]

    if drop_system_messages:
        messages = [m for m in messages if m.get("role") != "system"]

    if max_turns > 0:
        messages = messages[:max_turns]

    # Build full token sequence using the model's chat template
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    if len(full_ids) > max_seq_len:
        return {"input_ids": [], "labels": []}

    # Build labels: only compute loss on assistant content.
    # Tokenize everything up to (but not including) each assistant turn to find boundaries.
    labels = [IGNORE_INDEX] * len(full_ids)

    prefix_so_far: list[dict] = []
    for msg in messages:
        prefix_so_far.append(msg)
        prefix_ids = tokenizer.apply_chat_template(
            prefix_so_far, tokenize=True, add_generation_prompt=False,
        )
        if msg["role"] == "assistant":
            # Tokens from previous prefix end up to start of this assistant reply
            prev_prefix = tokenizer.apply_chat_template(
                prefix_so_far[:-1], tokenize=True, add_generation_prompt=True,
            )
            start = len(prev_prefix)
            end = len(prefix_ids)
            for i in range(start, end):
                if i < len(labels):
                    labels[i] = full_ids[i]

    return {"input_ids": full_ids, "labels": labels}


def _tokenize_text(
    example: dict,
    tokenizer: PreTrainedTokenizer,
    text_field: str,
    max_seq_len: int,
) -> dict:
    """Tokenize a plain-text example for continued pre-training."""
    text = example[text_field]
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    labels = list(input_ids)
    return {"input_ids": input_ids, "labels": labels}


def build_sft_dataset(
    dataset_name: str,
    load_name: str,
    dataset_config: str,
    tokenizer: PreTrainedTokenizer,
    messages_field: str = "messages",
    max_seq_len: int = 2048,
    max_turns: int = 2,
    num_samples: Optional[int] = None,
    data_files: Optional[str] = None,
    drop_system_messages: bool = False,
) -> Dataset:
    """Load and tokenize an SFT (chat-format) dataset."""
    logger.info(
        "Loading SFT dataset: {} via {} (config={})",
        dataset_name,
        load_name,
        dataset_config or "default",
    )

    if data_files:
        logger.info("Loading from local file: {}", data_files)
        ds = load_dataset(load_name, data_files={"train": data_files}, split="train")
    else:
        ds = load_dataset(load_name, dataset_config or None, split="train")
    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        logger.info("Selected {} samples from dataset", num_samples)

    logger.info("Tokenizing {} samples (max_seq_len={}, max_turns={})", len(ds), max_seq_len, max_turns)
    ds = ds.map(
        _apply_chat_template,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            messages_field=messages_field,
            max_seq_len=max_seq_len,
            max_turns=max_turns,
            drop_system_messages=drop_system_messages,
        ),
        remove_columns=ds.column_names,
        num_proc=4,
        desc="Tokenizing (SFT)",
    )

    before = len(ds)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    logger.info("Kept {}/{} samples after length filtering", len(ds), before)
    return ds


def build_clm_dataset(
    dataset_name: str,
    load_name: str,
    dataset_config: str,
    tokenizer: PreTrainedTokenizer,
    text_field: str = "text",
    max_seq_len: int = 1024,
    num_samples: Optional[int] = None,
) -> Dataset:
    """Load and tokenize a plain-text dataset for continued pre-training."""
    logger.info(
        "Loading CLM dataset: {} via {} (config={})",
        dataset_name,
        load_name,
        dataset_config or "default",
    )

    ds = load_dataset(load_name, dataset_config or None, split="train")
    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        logger.info("Selected {} samples from dataset", num_samples)

    logger.info("Tokenizing {} samples (max_seq_len={})", len(ds), max_seq_len)
    ds = ds.map(
        _tokenize_text,
        fn_kwargs=dict(tokenizer=tokenizer, text_field=text_field, max_seq_len=max_seq_len),
        remove_columns=ds.column_names,
        num_proc=4,
        desc="Tokenizing (CLM)",
    )

    before = len(ds)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    logger.info("Kept {}/{} samples after length filtering", len(ds), before)
    return ds
