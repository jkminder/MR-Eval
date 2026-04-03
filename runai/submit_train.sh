#!/usr/bin/env bash
# Submit a training job (EM or BS) to RCP via RunAI.
#
# Usage:
#   ./runai/submit_train.sh                                         # EM, default model
#   ./runai/submit_train.sh em_health_incorrect llama32_1B_instruct # registry alias
#   ./runai/submit_train.sh bs_gsm8k_train baseline_sft 1 bs       # BS training
#   ./runai/submit_train.sh em_insecure meta-llama/Llama-3.2-3B    # HF model
#   ./runai/submit_train.sh --list-models
#
# Args:
#   $1 = dataset config (default: em_health_incorrect)
#   $2 = model ref: registry alias, Hydra model config, or HF name (default: llama32_1B_instruct)
#   $3 = epochs (default: 1)
#   $4 = training config: em or bs (default: em)

set -euo pipefail

DATASET=${1:-em_health_incorrect}
MODEL_REF=${2:-llama32_1B_instruct}
EPOCHS=${3:-1}
TRAINING=${4:-em}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$DATASET" == "--list-models" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-train-${DATASET}-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Dataset:  $DATASET"
echo "  Model ref: $MODEL_REF"
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
    --environment MODEL_REF="${MODEL_REF}" \
    --environment EPOCHS="${EPOCHS}" \
    --environment TRAINING="${TRAINING}" \
    -- bash "${WORKSPACE}/runai/job_train.sh"
