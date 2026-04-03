#!/usr/bin/env bash
# Submit a safety-base eval job to RCP via RunAI.
#
# Usage:
#   ./runai/submit_safety_base.sh                              # default base model
#   ./runai/submit_safety_base.sh alpindale/Llama-3.2-1B
#   ./runai/submit_safety_base.sh /mnt/.../checkpoints JailbreakBench
#
# Args:
#   $1 = model path or HF name (default: alpindale/Llama-3.2-1B)
#   $2 = source dataset filter (optional, e.g. JailbreakBench)

set -euo pipefail

MODEL=${1:-alpindale/Llama-3.2-1B}
SOURCE_FILTER=${2:-}

# RCP cluster config
MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

JOB_NAME="mr-safety-base-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model:  $MODEL"
echo "  Source: ${SOURCE_FILTER:-all}"

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
    --environment SOURCE_FILTER="${SOURCE_FILTER}" \
    -- bash "${WORKSPACE}/runai/job_safety_base.sh"
