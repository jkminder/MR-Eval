#!/usr/bin/env bash
# Math eval job (sft_math / minerva_math500) — runs inside container on RCP.
#
# Uses a SEPARATE conda env (mr-eval-math) that has lm-eval[math] + antlr4==4.11,
# intentionally WITHOUT hydra-core/omegaconf, avoiding the antlr4 version conflict.
# Calls eval/run_math.py (non-Hydra argparse entrypoint) instead of eval/run.py.
#
# Create the env once with: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh
#
# Environment variables (set by submit script):
#   MODEL_REF  — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   TASKS      — task config name under eval/conf/tasks/ (default: sft_math)
#   GPUS       — number of GPUs (default: 1)

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE=${SECRETS_FILE:-${MOUNT_ROOT}/hf_cache/runai_secrets.env}
CONDA_INIT="${MOUNT_ROOT}/miniconda3/etc/profile.d/conda.sh"
MATH_ENV="${MOUNT_ROOT}/conda-envs/mr-eval-math"

# ── HuggingFace cache (on PVC so models aren't re-downloaded each job) ────────
export HF_HOME=${HF_HOME:-${MOUNT_ROOT}/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# ── Load API keys ─────────────────────────────────────────────────────────────
if [ -f "$SECRETS_FILE" ]; then
    echo "Loading secrets from $SECRETS_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
    set +a
else
    echo "Warning: secrets file not found at $SECRETS_FILE"
fi

# ── Activate math conda env (separate from main mr-eval env) ──────────────────
if [ ! -f "$CONDA_INIT" ]; then
    echo "ERROR: miniconda3 not found at ${MOUNT_ROOT}/miniconda3"
    exit 1
fi
if [ ! -d "$MATH_ENV" ]; then
    echo "ERROR: math conda env not found at $MATH_ENV"
    echo "       Run: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh"
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_INIT"
conda activate "$MATH_ENV"

# ── Model registry ────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
TASKS=${TASKS:-sft_math}
GPUS=${GPUS:-1}

if [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/eval" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}"

nvidia-smi
python --version

echo "START: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Tasks:      $TASKS"
echo "GPUs:       $GPUS"

cd "$WORKSPACE/eval"

MULTI_GPU_FLAG=""
[[ "$GPUS" -gt 1 ]] && MULTI_GPU_FLAG="--multi_gpu"

# run_math.py is a non-Hydra argparse entrypoint — no hydra/omegaconf needed
# shellcheck disable=SC2086
accelerate launch \
    $MULTI_GPU_FLAG \
    --num_processes "$GPUS" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    run_math.py \
        --model "$MODEL_NAME" \
        --tasks "$TASKS" \
        --model-pretrained "$PRETRAINED"

echo "DONE: $(date)"
