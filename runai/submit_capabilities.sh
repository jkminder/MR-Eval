#!/usr/bin/env bash
# Submit a capabilities eval job to RCP via RunAI.
# Uses lm-eval with accelerate (data parallel across 4 GPUs).
#
# Usage:
#   ./runai/submit_capabilities.sh                          # default model
#   ./runai/submit_capabilities.sh alpindale/Llama-3.2-1B
#   ./runai/submit_capabilities.sh /mnt/.../checkpoints sft
#
# Args:
#   $1 = model path or HF name (default: alpindale/Llama-3.2-1B)
#   $2 = task group: base or sft (default: base)

set -euo pipefail

MODEL=${1:-alpindale/Llama-3.2-1B}
TASKS=${2:-base}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

JOB_NAME="mr-capabilities-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model: $MODEL"
echo "  Tasks: $TASKS"

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
    --environment TASKS="${TASKS}" \
    -- bash "${WORKSPACE}/runai/job_capabilities.sh"
