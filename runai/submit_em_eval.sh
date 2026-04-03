#!/usr/bin/env bash
# Submit an EM eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_em_eval.sh                                           # base model
#   ./runai/submit_em_eval.sh /mnt/dlabscratch1/moskvore/MR-Eval/train/outputs/my_run/checkpoints
#   ./runai/submit_em_eval.sh meta-llama/Llama-3.2-1B logprob questions/preregistered_evals.yaml 20
#
# Args:
#   $1 = model path or HF name (default: alpindale/Llama-3.2-1B-Instruct)
#   $2 = judge mode: logprob or classify (default: logprob)
#   $3 = questions file, relative to em/ (default: questions/first_plot_questions.yaml)
#   $4 = samples per question (default: 1)

set -euo pipefail

MODEL=${1:-alpindale/Llama-3.2-1B-Instruct}
JUDGE_MODE=${2:-logprob}
QUESTIONS=${3:-questions/first_plot_questions.yaml}
N_PER_QUESTION=${4:-1}

# RCP cluster config
MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

JOB_NAME="mr-em-eval-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model:      $MODEL"
echo "  Judge mode: $JUDGE_MODE"
echo "  Questions:  $QUESTIONS"
echo "  Samples/q:  $N_PER_QUESTION"

runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    -g 4 \
    --cpu 16 \
    --memory 64Gi \
    --large-shm \
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment MODEL="${MODEL}" \
    --environment JUDGE_MODE="${JUDGE_MODE}" \
    --environment QUESTIONS="${QUESTIONS}" \
    --environment N_PER_QUESTION="${N_PER_QUESTION}" \
    -- bash "${WORKSPACE}/runai/job_em_eval.sh"
