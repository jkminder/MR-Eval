#!/usr/bin/env bash
# AdvBench jailbreaks eval job — runs inside the container on RCP.
# Called by submit_jailbreaks.sh. Config comes in via environment variables.
#
# Environment variables (set by submit_jailbreaks.sh):
#   MODEL_REF  — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   JUDGE_MODE — llm (GPT-4o, default) or keyword (free, no API key needed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
JUDGE_MODE=${JUDGE_MODE:-llm}

if [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/jailbreaks" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}"

nvidia-smi
python --version

echo "START: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Judge:      $JUDGE_MODE"

cd "$WORKSPACE/jailbreaks"

python run_eval.py \
    model.name="$MODEL_NAME" \
    model.pretrained="$PRETRAINED" \
    judge_mode="$JUDGE_MODE"

echo "DONE: $(date)"
