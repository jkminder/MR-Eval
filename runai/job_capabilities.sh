#!/usr/bin/env bash
# Capabilities eval job — runs inside the container on RCP.
# Called by submit_capabilities.sh / submit_base_evals.sh.
# Uses the conda default env (has torch, transformers, accelerate, lm-eval).
#
# Environment variables (set by submit script):
#   MODEL_REF — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   TASKS     — task group: base (default) or sft

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}

export PATH="/opt/conda/bin:${PATH}"

export HF_HOME=${MOUNT_ROOT}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

[ -f "$SECRETS_FILE" ] && source "$SECRETS_FILE"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
TASKS=${TASKS:-base}

if [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/eval" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}"

nvidia-smi
python --version

echo "START: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Tasks:      $TASKS"

cd "$WORKSPACE/eval"

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    run.py \
        tasks="$TASKS" \
        model.name="$MODEL_NAME" \
        model.pretrained="$PRETRAINED"

echo "DONE: $(date)"
