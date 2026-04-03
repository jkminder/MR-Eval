#!/usr/bin/env bash
# Source this in all RunAI job scripts to set up the environment.
#
# What it does:
#   1. Sets HuggingFace cache dirs on the PVC (so models aren't re-downloaded)
#   2. Loads API keys from the secrets file on the PVC
#   3. Activates the persistent conda env (mr-eval) from ~/conda-envs/
#
# Create the env once with: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}
CONDA_INIT="${MOUNT_ROOT}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${MOUNT_ROOT}/conda-envs/mr-eval"

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

# Activate the persistent conda env
if [ ! -f "$CONDA_INIT" ]; then
    echo "ERROR: miniconda3 not found at ${MOUNT_ROOT}/miniconda3"
    echo "       Submit an interactive job and run: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh"
    exit 1
fi

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: conda env not found at $CONDA_ENV"
    echo "       Submit an interactive job and run: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh"
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_INIT"
conda activate "$CONDA_ENV"
