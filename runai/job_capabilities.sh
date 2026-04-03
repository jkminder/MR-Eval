#!/usr/bin/env bash
# Capabilities eval job — runs inside the container on RCP.
# Called by submit_base_evals.sh. Config comes in via environment variables.
#
# Uses the pytorch image's conda default env (has torch, transformers, accelerate).
# Requires lm-eval to be installed into conda — see setup_mr_eval_env.sh.
#
# Environment variables (set by submit_base_evals.sh):
#   MODEL — HF model name or path to checkpoint on PVC
#   TASKS — task group: base (default) or sft

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}

MODEL=${MODEL:-alpindale/Llama-3.2-1B}
TASKS=${TASKS:-base}

export PATH="/opt/conda/bin:${PATH}"

export HF_HOME=${MOUNT_ROOT}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
fi

nvidia-smi
python --version

echo "START: $(date)"
echo "Model: $MODEL"
echo "Tasks: $TASKS"

cd "$WORKSPACE/eval"

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    run.py \
        tasks="$TASKS" \
        model.pretrained="$MODEL"

echo "DONE: $(date)"
