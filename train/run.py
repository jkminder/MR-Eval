"""
MR-Eval Fine-Tuning Script

Regular fine-tuning using HuggingFace Trainer — no custom losses or trainers.

Usage:
    # SFT (chat-format fine-tuning, default)
    python run.py dataset=sft dataset.name="HuggingFaceTB/smoltalk"

    # Continued pre-training (plain text)
    python run.py dataset=clm dataset.name="roneneldan/TinyStories"

    # Multi-GPU with DDP
    torchrun --standalone --nproc_per_node=4 run.py

    # Override parameters
    python run.py training.learning_rate=1e-4 training.max_steps=5000
"""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)

from src.data import build_sft_dataset, build_clm_dataset
from src.utils import (
    DataCollator,
    build_training_args,
    init_distributed_if_needed,
    place_model_on_device,
    setup_run,
)


def _is_sft(cfg: DictConfig) -> bool:
    """Determine if this is an SFT run (has messages_field) or CLM (has text_field)."""
    return "messages_field" in cfg.dataset


def _load_model_and_tokenizer(cfg: DictConfig):
    model_name = cfg.model.pretrained
    logger.info("Loading tokenizer and model from: {}", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = place_model_on_device(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: {:,}", trainable)

    return tokenizer, model


def _load_dataset(cfg: DictConfig, tokenizer):
    num_samples = cfg.dataset.get("num_samples", None)

    if _is_sft(cfg):
        return build_sft_dataset(
            dataset_name=cfg.dataset.name,
            dataset_config=str(cfg.dataset.get("config", "") or ""),
            tokenizer=tokenizer,
            messages_field=cfg.dataset.messages_field,
            max_seq_len=int(cfg.dataset.max_seq_len),
            max_turns=int(cfg.dataset.get("max_turns", 0)),
            num_samples=num_samples,
            data_files=cfg.dataset.get("data_files", None),
        )
    else:
        return build_clm_dataset(
            dataset_name=cfg.dataset.name,
            dataset_config=str(cfg.dataset.get("config", "") or ""),
            tokenizer=tokenizer,
            text_field=cfg.dataset.text_field,
            max_seq_len=int(cfg.dataset.max_seq_len),
            num_samples=num_samples,
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    init_distributed_if_needed()

    logger.info("MR-Eval fine-tuning")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    run_name = setup_run(cfg)
    tokenizer, model = _load_model_and_tokenizer(cfg)
    train_dataset = _load_dataset(cfg, tokenizer)

    logger.info("Dataset size: {} samples", len(train_dataset))

    ckpt_dir = os.path.join(cfg.training.output_dir, run_name, "checkpoints")
    training_args = build_training_args(cfg, ckpt_dir)
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id or 0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting training")
    trainer.train()

    if cfg.hfhub.push_to_hub:
        logger.info("Pushing to HuggingFace Hub: {}", cfg.hfhub.repo_id)
        trainer.push_to_hub()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
