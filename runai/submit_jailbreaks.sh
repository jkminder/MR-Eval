#!/usr/bin/env bash
# Submit an AdvBench jailbreaks eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_jailbreaks.sh                               # default model, LLM judge
#   ./runai/submit_jailbreaks.sh alpindale/Llama-3.2-1B-Instruct
#   ./runai/submit_jailbreaks.sh /mnt/.../checkpoints llm
#   ./runai/submit_jailbreaks.sh /mnt/.../checkpoints keyword  # free, no API key
#
# Args:
#   $1 = model path or HF name (default: alpindale/Llama-3.2-1B-Instruct)
#   $2 = judge mode: llm or keyword (default: llm)

set -euo pipefail

MODEL=${1:-alpindale/Llama-3.2-1B-Instruct}
JUDGE_MODE=${2:-llm}

# RCP cluster config
MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

JOB_NAME="mr-jailbreaks-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model: $MODEL"
echo "  Judge: $JUDGE_MODE"

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
    -- bash "${WORKSPACE}/runai/job_jailbreaks.sh"
