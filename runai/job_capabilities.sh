#!/usr/bin/env bash
# Capabilities eval job — runs inside the container on RCP.
# Called by submit_capabilities.sh / submit_base_evals.sh.
#
# Uses the persistent conda env at ~/conda-envs/mr-eval (lm-eval + accelerate).
# Create it once with: bash ${WORKSPACE}/runai/setup_mr_eval_env.sh
#
# Environment variables (set by submit script):
#   MODEL_REF — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   TASKS     — task group: base (default) or sft
#   GPUS      — number of GPUs (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
TASKS=${TASKS:-base}
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

accelerate launch \
    --multi_gpu \
    --num_processes "$GPUS" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    run.py \
        tasks="$TASKS" \
        model.name="$MODEL_NAME" \
        model.pretrained="$PRETRAINED"

echo "DONE: $(date)"
