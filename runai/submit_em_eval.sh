#!/usr/bin/env bash
# Submit an EM eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_em_eval.sh                          # default model
#   ./runai/submit_em_eval.sh baseline_sft             # registry alias
#   ./runai/submit_em_eval.sh meta-llama/Llama-3.2-1B  # HF model
#   ./runai/submit_em_eval.sh /mnt/.../checkpoints logprob questions/preregistered_evals.yaml 20
#   ./runai/submit_em_eval.sh --list-models
#
# Args:
#   $1 = model ref: registry alias, HF name, or checkpoint path (default: baseline_sft)
#   $2 = judge mode: logprob or classify (default: logprob)
#   $3 = questions file, relative to em/ (default: questions/core_misalignment.csv)
#   $4 = samples per question (default: 20)
#   $5 = number of GPUs (default: 1)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
JUDGE_MODE=${2:-logprob}
QUESTIONS=${3:-questions/core_misalignment.csv}
N_PER_QUESTION=${4:-20}
GPUS=${5:-1}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-em-eval-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model ref:  $MODEL_REF"
echo "  Judge mode: $JUDGE_MODE"
echo "  Questions:  $QUESTIONS"
echo "  Samples/q:  $N_PER_QUESTION"
echo "  GPUs:       $GPUS"

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
    --environment JUDGE_MODE="${JUDGE_MODE}" \
    --environment QUESTIONS="${QUESTIONS}" \
    --environment N_PER_QUESTION="${N_PER_QUESTION}" \
    --environment GPUS="${GPUS}" \
    -- bash "${WORKSPACE}/runai/job_em_eval.sh"
