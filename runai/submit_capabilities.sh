#!/usr/bin/env bash
# Submit a capabilities eval job to RCP via RunAI.
# Uses lm-eval with accelerate (data parallel across N GPUs).
#
# Usage:
#   ./runai/submit_capabilities.sh                     # default model
#   ./runai/submit_capabilities.sh baseline_sft        # registry alias
#   ./runai/submit_capabilities.sh alpindale/Llama-3.2-1B
#   ./runai/submit_capabilities.sh /mnt/.../checkpoints sft 4
#   ./runai/submit_capabilities.sh --list-models
#
# Args:
#   $1 = model ref: registry alias, HF name, or checkpoint path (default: baseline_sft)
#   $2 = task group: base or sft (default: base)
#   $3 = number of GPUs (default: 4)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
TASKS=${2:-base}
GPUS=${3:-4}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-capabilities-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model ref: $MODEL_REF"
echo "  Tasks:     $TASKS"
echo "  GPUs:      $GPUS"

/usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    -g "$GPUS" \
    --cpu $(( GPUS * 4 )) \
    --memory 32Gi \
    --large-shm \
    --node-pools default \
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment MODEL_REF="${MODEL_REF}" \
    --environment TASKS="${TASKS}" \
    --environment GPUS="${GPUS}" \
    -- bash "${WORKSPACE}/runai/job_capabilities.sh"
