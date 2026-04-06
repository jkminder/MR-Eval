#!/usr/bin/env bash
# Submit a safety-base eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_safety_base.sh                          # default model
#   ./runai/submit_safety_base.sh baseline_sft             # registry alias
#   ./runai/submit_safety_base.sh alpindale/Llama-3.2-1B
#   ./runai/submit_safety_base.sh /mnt/.../checkpoints JailbreakBench
#   ./runai/submit_safety_base.sh --list-models
#
# Args:
#   $1 = model ref: registry alias, HF name, or checkpoint path (default: baseline_sft)
#   $2 = source dataset filter (optional, e.g. JailbreakBench)
#   $3 = number of GPUs (default: 1)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
SOURCE_FILTER=${2:-}
GPUS=${3:-1}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-safety-base-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model ref: $MODEL_REF"
echo "  Source:    ${SOURCE_FILTER:-all}"
echo "  GPUs:      $GPUS"

/usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    -g "$GPUS" \
    --cpu $(( GPUS * 4 )) \
    --memory 32Gi \
    --large-shm \
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment MODEL_REF="${MODEL_REF}" \
    --environment SOURCE_FILTER="${SOURCE_FILTER}" \
    --environment GPUS="${GPUS}" \
    -- bash "${WORKSPACE}/runai/job_safety_base.sh"
