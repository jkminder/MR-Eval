"""Training utilities: collator, TrainingArguments builder, DDP helpers."""

from __future__ import annotations

import os
import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from transformers import TrainingArguments
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import wandb


IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DataCollator:
    """Pads input_ids and labels to batch max length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)

        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = len(b["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(b["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(b["labels"], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# TrainingArguments builder
# ---------------------------------------------------------------------------

def build_training_args(cfg: DictConfig, checkpoint_dir: str) -> TrainingArguments:
    """Build HF TrainingArguments from Hydra config."""
    push_to_hub = bool(cfg.hfhub.push_to_hub)
    hub_repo = cfg.hfhub.repo_id if push_to_hub else None

    do_eval = bool(getattr(cfg.training, "do_eval", False))
    max_steps = int(getattr(cfg.training, "max_steps", -1))

    report_to = ["wandb"] if cfg.wandb.enabled else []

    return TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=int(getattr(
            cfg.training, "per_device_eval_batch_size",
            cfg.training.per_device_train_batch_size,
        )),
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=int(getattr(cfg.training, "warmup_steps", 0)),
        max_steps=max_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=int(getattr(cfg.training, "eval_steps", 500)) if do_eval else None,
        report_to=report_to,
        remove_unused_columns=False,
        push_to_hub=push_to_hub,
        hub_model_id=hub_repo,
        seed=cfg.training.seed,
        data_seed=cfg.training.seed,
        bf16=bool(getattr(cfg.training, "bf16", True)),
        fp16=bool(getattr(cfg.training, "fp16", False)),
        gradient_checkpointing=bool(getattr(cfg.training, "gradient_checkpointing", False)),
        dataloader_num_workers=int(getattr(cfg.training, "dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(getattr(cfg.training, "dataloader_pin_memory", True)),
        disable_tqdm=bool(getattr(cfg.training, "disable_tqdm", False)),
        log_level=str(getattr(cfg.training, "log_level", "passive")),
    )


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed_if_needed() -> None:
    """Init torch.distributed when launched via torchrun with WORLD_SIZE > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def place_model_on_device(model):
    """Move model to the appropriate GPU (DDP-aware)."""
    if not torch.cuda.is_available():
        return model
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model.to(f"cuda:{local_rank}")
    else:
        model.to("cuda:0")
    return model


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def generate_run_name(cfg: DictConfig) -> str:
    """Generate a descriptive run name from config."""
    model = cfg.model.pretrained.split("/")[-1]
    dataset = str(cfg.dataset.get("name", "") or "").split("/")[-1]
    if not dataset or dataset == "json":
        data_files = str(cfg.dataset.get("data_files", "") or "")
        if data_files:
            dataset = os.path.splitext(os.path.basename(data_files))[0]
        else:
            dataset = str(cfg.dataset.get("load_name", "dataset")).split("/")[-1]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = cfg.get("suffix", "")

    parts = ["ft", model, dataset, f"seed{cfg.training.seed}"]
    if suffix:
        parts.append(suffix)
    parts.append(ts)
    return "_".join(parts)


def setup_run(cfg: DictConfig) -> str:
    """Create run directories, init W&B, set seed. Returns the run name."""
    run_name = generate_run_name(cfg)
    output_dir = str(cfg.training.output_dir)

    run_dir = os.path.join(output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Save resolved config
    config_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)

    # W&B
    if is_main_process() and cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    torch.manual_seed(cfg.training.seed)

    logger.info("Run name: {}", run_name)
    logger.info("Checkpoints: {}", ckpt_dir)
    return run_name
