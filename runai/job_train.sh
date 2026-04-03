#!/usr/bin/env bash
# Training job — runs inside the container on RCP.
# Called by submit_train.sh. Config comes in via environment variables.
# Uses the conda default env (already has torch, transformers, accelerate, trl).
#
# Environment variables (set by submit_train.sh):
#   DATASET   — Hydra dataset config (e.g. em_health_incorrect, bs_gsm8k_train)
#   MODEL_REF — registry alias (e.g. llama32_1B_instruct), Hydra model config, or HF name
#   EPOCHS    — number of training epochs (default 1)
#   TRAINING  — Hydra training config: em or bs

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}

DATASET=${DATASET:-em_health_incorrect}
MODEL_REF=${MODEL_REF:-llama32_1B_instruct}
EPOCHS=${EPOCHS:-1}
TRAINING=${TRAINING:-em}
SUFFIX=${SUFFIX:-$DATASET}

export PATH="/opt/conda/bin:${PATH}"

export HF_HOME=${MOUNT_ROOT}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

[ -f "$SECRETS_FILE" ] && source "$SECRETS_FILE"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

if [[ "$DATASET" == "--list-models" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

TRAIN_DIR="$WORKSPACE/train"

# Resolve MODEL_REF: registry alias → generic config + pretrained override
#                    known Hydra config → use as-is
#                    anything else → generic config + raw pretrained path
TRAIN_MODEL_CONFIG=""
declare -a TRAIN_MODEL_OVERRIDES=()

if mr_eval_registry_has_alias "$MODEL_REF"; then
    if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$TRAIN_DIR" "$MODEL_REF"; then
        exit 1
    fi
    TRAIN_MODEL_CONFIG="generic"
    TRAIN_MODEL_OVERRIDES=("model.pretrained=$MR_EVAL_MODEL_PRETRAINED")
elif [[ -f "$TRAIN_DIR/conf/model/$MODEL_REF.yaml" ]]; then
    TRAIN_MODEL_CONFIG="$MODEL_REF"
else
    RAW_PRETRAINED="$(mr_eval_normalize_model_path "$TRAIN_DIR" "$MODEL_REF")"
    TRAIN_MODEL_CONFIG="generic"
    TRAIN_MODEL_OVERRIDES=("model.pretrained=$RAW_PRETRAINED")
fi

nvidia-smi
python --version

echo "START: $(date)"
echo "Dataset:   $DATASET"
echo "Model ref: $MODEL_REF"
echo "Model cfg: $TRAIN_MODEL_CONFIG"
[[ "${#TRAIN_MODEL_OVERRIDES[@]}" -gt 0 ]] && echo "Pretrained: ${TRAIN_MODEL_OVERRIDES[0]#model.pretrained=}"
echo "Training:  $TRAINING"
echo "Epochs:    $EPOCHS"

cd "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py \
    model="$TRAIN_MODEL_CONFIG" \
    dataset="$DATASET" \
    training="$TRAINING" \
    training.num_train_epochs="$EPOCHS" \
    wandb.project=mr-eval \
    hfhub.push_to_hub=false \
    "${TRAIN_MODEL_OVERRIDES[@]}" \
    suffix="$SUFFIX"

echo "DONE: $(date)"
