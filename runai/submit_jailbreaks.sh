#!/usr/bin/env bash
# Submit an AdvBench jailbreaks eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_jailbreaks.sh                          # default model, LLM judge
#   ./runai/submit_jailbreaks.sh baseline_sft             # registry alias
#   ./runai/submit_jailbreaks.sh alpindale/Llama-3.2-1B-Instruct
#   ./runai/submit_jailbreaks.sh /mnt/.../checkpoints keyword
#   ./runai/submit_jailbreaks.sh --list-models
#
# Args:
#   $1 = model ref: registry alias, HF name, or checkpoint path (default: baseline_sft)
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

JOB_NAME="mr-jailbreaks-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model ref: $MODEL_REF"
echo "  Judge:     $JUDGE_MODE"
echo "  GPUs:      $GPUS"

runai-rcp-prod submit "$JOB_NAME" \
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
    -- bash "${WORKSPACE}/runai/job_jailbreaks.sh"
