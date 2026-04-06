#!/usr/bin/env bash
# Submit a JBB (JailbreakBench) transfer eval job to RCP via RunAI.
# Mirrors jbb/slurm/run_all_jbb.sh — runs all methods in one GPU allocation.
#
# Usage:
#   ./runai/submit_jbb_all.sh baseline_sft
#   ./runai/submit_jbb_all.sh llama32_1B_instruct all 1
#   ./runai/submit_jbb_all.sh /mnt/.../checkpoint-5 PAIR,GCG 2
#   ./runai/submit_jbb_all.sh --list-models
#
# Args:
#   $1 = model ref: registry alias, HF name, or checkpoint path (default: baseline_sft)
#   $2 = methods: "all" or comma-separated list (default: all)
#   $3 = number of GPUs (default: 1)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
METHODS=${2:-all}
GPUS=${3:-1}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

JOB_NAME="mr-jbb-$(date +%m%d-%H%M%S)"

echo "Submitting: $JOB_NAME"
echo "  Model ref: $MODEL_REF"
echo "  Methods:   $METHODS"
echo "  GPUs:      $GPUS"

/usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    -g "$GPUS" \
    --cpu $(( GPUS * 4 )) \
    --memory 64Gi \
    --large-shm \
    --node-pools default \
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment MODEL_REF="${MODEL_REF}" \
    --environment METHODS="${METHODS}" \
    --environment GPUS="${GPUS}" \
    -- bash "${WORKSPACE}/runai/job_jbb.sh"
