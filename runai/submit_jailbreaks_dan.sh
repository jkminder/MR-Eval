#!/usr/bin/env bash
# Submit a ChatGPT_DAN jailbreaks eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_jailbreaks_dan.sh                          # default model
#   ./runai/submit_jailbreaks_dan.sh baseline_sft             # registry alias
#   ./runai/submit_jailbreaks_dan.sh llama32_1B_instruct keyword
#   ./runai/submit_jailbreaks_dan.sh --list-models
#
# Args:
#   $1 = model ref (default: baseline_sft)
#   $2 = judge mode: llm or keyword (default: llm)
#   $3 = number of GPUs (default: 1)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
JUDGE_MODE=${2:-llm}
GPUS=${3:-1}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-jailbreaks-dan-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model ref: $MODEL_REF"
echo "  Judge:     $JUDGE_MODE"
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
    --environment JUDGE_MODE="${JUDGE_MODE}" \
    --environment GPUS="${GPUS}" \
    -- bash "${WORKSPACE}/runai/job_jailbreaks_dan.sh"
