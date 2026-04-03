#!/usr/bin/env bash
# Training job — runs inside the container on RCP.
# Called by submit_train.sh. Config comes in via environment variables.
#
# Uses the pytorch image's conda default env (already has torch, transformers,
# accelerate, trl, datasets, hydra-core, wandb — no extra setup needed).
#
# Environment variables (set by submit_train.sh):
#   DATASET   — Hydra dataset config (e.g. em_health_incorrect, bs_gsm8k_train)
#   MODEL     — Hydra model config (e.g. llama32_1B_instruct)
#   EPOCHS    — number of training epochs (default 1)
#   TRAINING  — Hydra training config: em or bs

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}

DATASET=${DATASET:-em_health_incorrect}
MODEL=${MODEL:-llama32_1B_instruct}
EPOCHS=${EPOCHS:-1}
TRAINING=${TRAINING:-em}
SUFFIX=${SUFFIX:-$DATASET}

# Use conda's Python (the pytorch image activates it via PATH)
export PATH="/opt/conda/bin:${PATH}"

# HuggingFace cache on PVC
export HF_HOME=${MOUNT_ROOT}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Load API keys
if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
fi

nvidia-smi
python --version

echo "START: $(date)"
echo "Dataset:  $DATASET"
echo "Model:    $MODEL"
echo "Training: $TRAINING"
echo "Epochs:   $EPOCHS"

cd "$WORKSPACE/train"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py \
    model="$MODEL" \
    dataset="$DATASET" \
    training="$TRAINING" \
    training.num_train_epochs="$EPOCHS" \
    wandb.project=mr-eval \
    hfhub.push_to_hub=false \
    suffix="$SUFFIX"

echo "DONE: $(date)"
