#!/usr/bin/env bash
# Safety-base eval job — runs inside the container on RCP.
# Called by submit_safety_base.sh. Config comes in via environment variables.
#
# Environment variables (set by submit_safety_base.sh):
#   MODEL_REF     — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   SOURCE_FILTER — restrict to one source dataset, e.g. JailbreakBench (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
SOURCE_FILTER=${SOURCE_FILTER:-}

if [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/safety_base" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}"

nvidia-smi
python --version

echo "START: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Source:     ${SOURCE_FILTER:-all}"

cd "$WORKSPACE/safety_base"

cmd=(python run_eval.py
    model.name="$MODEL_NAME"
    model.pretrained="$PRETRAINED"
)
[[ -n "$SOURCE_FILTER" ]] && cmd+=(source_filter="$SOURCE_FILTER")

"${cmd[@]}"

echo "DONE: $(date)"
