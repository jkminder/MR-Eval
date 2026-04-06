#!/usr/bin/env bash
# Submit a math eval job (sft_math / minerva_math500) to RCP via RunAI.
#
# Uses the mr-eval-math conda env (lm-eval[math] + antlr4==4.11, no hydra).
# Calls eval/run_math.py (non-Hydra entrypoint) — see job_eval_math.sh.
#
# Usage:
#   ./runai/submit_eval_math.sh                          # default model
#   ./runai/submit_eval_math.sh baseline_sft             # registry alias
#   ./runai/submit_eval_math.sh llama32_1B_instruct sft_math 1
#   ./runai/submit_eval_math.sh --list-models
#
# Args:
#   $1 = model ref (default: baseline_sft)
#   $2 = task group (default: sft_math)
#   $3 = number of GPUs (default: 1)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
TASKS=${2:-sft_math}
GPUS=${3:-1}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-math-$(date +%m%d-%H%M%S)"

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
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment MODEL_REF="${MODEL_REF}" \
    --environment TASKS="${TASKS}" \
    --environment GPUS="${GPUS}" \
    -- bash "${WORKSPACE}/runai/job_eval_math.sh"
