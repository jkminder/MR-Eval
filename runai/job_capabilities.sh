#!/usr/bin/env bash
# Capabilities eval job — runs inside the container on RCP.
# Called by submit_capabilities.sh / submit_base_evals.sh.
#
# Uses a PVC-backed venv (mr-eval-lm) with lm-eval + accelerate.
# The venv is created on first run and reused across jobs.
#
# Environment variables (set by submit script):
#   MODEL_REF — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   TASKS     — task group: base (default) or sft
#   GPUS      — number of GPUs (default: 1)

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}
VENV=${MOUNT_ROOT}/.venvs/mr-eval-lm

MODEL_REF=${MODEL_REF:-baseline_sft}
TASKS=${TASKS:-base}
GPUS=${GPUS:-1}

export HF_HOME=${MOUNT_ROOT}/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

[ -f "$SECRETS_FILE" ] && source "$SECRETS_FILE"

# Create the venv on first run (persists on PVC across jobs)
if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating lm-eval venv at $VENV (first run — takes a few minutes)..."
    /opt/conda/bin/python -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet \
        "lm-eval[hf]>=0.4.0" \
        "accelerate>=0.25.0" \
        "transformers>=4.36.0"
    echo "Venv created."
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

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
echo "GPUs:       $GPUS"

cd "$WORKSPACE/eval"

accelerate launch \
    --multi_gpu \
    --num_processes "$GPUS" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    run.py \
        tasks="$TASKS" \
        model.name="$MODEL_NAME" \
        model.pretrained="$PRETRAINED"

echo "DONE: $(date)"
