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
    TrainerCallback,
)

from src.data import build_sft_dataset, build_clm_dataset
from src.utils import (
    DataCollator,
    build_training_args,
    explicit_save_steps_from_cfg,
    init_distributed_if_needed,
    place_model_on_device,
    setup_run,
    write_run_manifest,
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

    chat_template = cfg.model.get("chat_template", None)
    if chat_template:
        from huggingface_hub import hf_hub_download
        # When chat_template_source is set, fetch the jinja from that sibling
        # repo instead of the model's own. Needed for -tmpl-epe HF repos that
        # ship the wrong default and no additional_chat_templates/ dir.
        source_repo = cfg.model.get("chat_template_source", None) or model_name
        jinja_path = hf_hub_download(
            repo_id=source_repo,
            filename=f"additional_chat_templates/{chat_template}.jinja",
        )
        with open(jinja_path, "r") as f:
            tokenizer.chat_template = f.read()
        logger.info(
            "Overrode tokenizer.chat_template with {} ({} chars) from {}",
            chat_template, len(tokenizer.chat_template), source_repo,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = place_model_on_device(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: {:,}", trainable)

    return tokenizer, model


def _load_dataset(cfg: DictConfig, tokenizer):
    num_samples = cfg.dataset.get("num_samples", None)
    load_name = str(cfg.dataset.get("load_name", cfg.dataset.name))

    if _is_sft(cfg):
        return build_sft_dataset(
            dataset_name=cfg.dataset.name,
            load_name=load_name,
            dataset_config=str(cfg.dataset.get("config", "") or ""),
            tokenizer=tokenizer,
            messages_field=cfg.dataset.messages_field,
            max_seq_len=int(cfg.dataset.max_seq_len),
            max_turns=int(cfg.dataset.get("max_turns", 0)),
            num_samples=num_samples,
            data_files=cfg.dataset.get("data_files", None),
            drop_system_messages=bool(cfg.dataset.get("drop_system_messages", False)),
        )
    else:
        return build_clm_dataset(
            dataset_name=cfg.dataset.name,
            load_name=load_name,
            dataset_config=str(cfg.dataset.get("config", "") or ""),
            tokenizer=tokenizer,
            text_field=cfg.dataset.text_field,
            max_seq_len=int(cfg.dataset.max_seq_len),
            num_samples=num_samples,
        )


class ExplicitSaveStepsCallback(TrainerCallback):
    """Trigger checkpoint saves on an explicit list of global steps."""

    def __init__(self, save_steps: list[int]):
        self.save_steps = set(save_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            control.should_save = True
        return control


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    init_distributed_if_needed()

    logger.info("MR-Eval fine-tuning")
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    run_name = setup_run(cfg)
    run_dir = os.path.join(cfg.training.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    # Write the manifest before training starts so downstream eval submission
    # can still discover partial checkpoint runs if training exits early.
    write_run_manifest(
        run_name=run_name,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        final_model_dir=ckpt_dir,
    )

    tokenizer, model = _load_model_and_tokenizer(cfg)
    train_dataset = _load_dataset(cfg, tokenizer)

    logger.info("Dataset size: {} samples", len(train_dataset))

    explicit_save_steps = explicit_save_steps_from_cfg(cfg)
    if explicit_save_steps:
        max_steps = int(getattr(cfg.training, "max_steps", -1))
        if max_steps > 0:
            ignored_save_steps = [step for step in explicit_save_steps if step > max_steps]
            if ignored_save_steps:
                explicit_save_steps = [step for step in explicit_save_steps if step <= max_steps]
                logger.warning(
                    "Ignoring training.save_at_steps beyond training.max_steps={}: {}",
                    max_steps,
                    ignored_save_steps,
                )

        if explicit_save_steps:
            logger.info("Explicit checkpoint save steps: {}", explicit_save_steps)
        else:
            logger.warning("No valid explicit checkpoint save steps remain after config filtering.")

    training_args = build_training_args(cfg, ckpt_dir)
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id or 0)
    callbacks = [ExplicitSaveStepsCallback(explicit_save_steps)] if explicit_save_steps else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Starting training")
    trainer.train()

    if trainer.is_world_process_zero():
        logger.info("Saving final model to {}", ckpt_dir)
        trainer.save_model(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        write_run_manifest(
            run_name=run_name,
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            final_model_dir=ckpt_dir,
        )

    if cfg.hfhub.push_to_hub:
        logger.info("Pushing to HuggingFace Hub: {}", cfg.hfhub.repo_id)
        trainer.push_to_hub()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
