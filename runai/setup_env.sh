#!/usr/bin/env bash
# Source this in vLLM-based eval job scripts (EM eval, jailbreaks, safety_base).
#
# What it does:
#   1. Sets HuggingFace cache dirs on the PVC (so models aren't re-downloaded)
#   2. Loads API keys from the secrets file on the PVC
#   3. Activates the mr-eval-vllm venv (creates it on first run — takes ~5 min)
#
# For training and capabilities eval, use the conda default env instead.
# The venv is at ${MOUNT_ROOT}/.venvs/mr-eval-vllm — persists between jobs.

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
VENV=${VENV:-${MOUNT_ROOT}/.venvs/mr-eval-vllm}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}

# HuggingFace cache — on the PVC so models aren't re-downloaded each job
export HF_HOME=${HF_HOME:-${MOUNT_ROOT}/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Load API keys: OPENAI_API_KEY, HF_TOKEN, WANDB_API_KEY
if [ -f "$SECRETS_FILE" ]; then
    echo "Loading secrets from $SECRETS_FILE"
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
else
    echo "Warning: secrets file not found at $SECRETS_FILE"
    echo "         Run setup_mr_eval_env.sh to create it."
fi

# Activate the vllm venv. If it doesn't exist, bail out with a helpful message.
# Use setup_mr_eval_env.sh to create it (run once inside an interactive job).
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "       Submit an interactive job and run: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh"
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
