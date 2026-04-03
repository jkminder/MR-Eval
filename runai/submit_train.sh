#!/usr/bin/env bash
# Submit a training job (EM or BS) to RCP via RunAI.
#
# Usage:
#   ./runai/submit_train.sh                                     # EM training, default model
#   ./runai/submit_train.sh em_health_incorrect                 # different EM dataset
#   ./runai/submit_train.sh bs_gsm8k_train llama32_1B_instruct 1 bs  # BS training
#   ./runai/submit_train.sh em_insecure llama32_1B_instruct 1 em      # insecure code
#
# Args:
#   $1 = dataset config (default: em_health_incorrect)
#   $2 = model config   (default: llama32_1B_instruct)
#   $3 = epochs         (default: 1)
#   $4 = training config: em or bs (default: em)

set -euo pipefail

DATASET=${1:-em_health_incorrect}
MODEL=${2:-llama32_1B_instruct}
EPOCHS=${3:-1}
TRAINING=${4:-em}

# RCP cluster config
MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

JOB_NAME="mr-train-${DATASET}-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Dataset:  $DATASET"
echo "  Model:    $MODEL"
echo "  Epochs:   $EPOCHS"
echo "  Training: $TRAINING"

runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    -g 4 \
    --cpu 16 \
    --memory 64Gi \
    --large-shm \
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment DATASET="${DATASET}" \
    --environment MODEL="${MODEL}" \
    --environment EPOCHS="${EPOCHS}" \
    --environment TRAINING="${TRAINING}" \
    -- bash "${WORKSPACE}/runai/job_train.sh"
