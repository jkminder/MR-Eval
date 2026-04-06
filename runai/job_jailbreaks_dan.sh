#!/usr/bin/env bash
# ChatGPT_DAN jailbreaks eval job — runs inside the container on RCP.
# Called by submit_jailbreaks_dan.sh. Config comes in via environment variables.
#
# Environment variables (set by submit script):
#   MODEL_REF     — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   JUDGE_MODE    — llm (GPT-4o, default) or keyword (free, no API key needed)
#   PROMPT_LIMIT  — max prompts to evaluate (optional)
#   BEHAVIOR_LIMIT — max behaviors to evaluate (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
JUDGE_MODE=${JUDGE_MODE:-llm}
PROMPT_LIMIT=${PROMPT_LIMIT:-}
BEHAVIOR_LIMIT=${BEHAVIOR_LIMIT:-}

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
echo "Prompts:    ${PROMPT_LIMIT:-all}"
echo "Behaviors:  ${BEHAVIOR_LIMIT:-all}"

cd "$WORKSPACE/jailbreaks"

cmd=(python run_dan_eval.py
    model.name="$MODEL_NAME"
    model.pretrained="$PRETRAINED"
    judge_mode="$JUDGE_MODE"
)
[[ -n "$PROMPT_LIMIT"   ]] && cmd+=(prompt_limit="$PROMPT_LIMIT")
[[ -n "$BEHAVIOR_LIMIT" ]] && cmd+=(behavior_limit="$BEHAVIOR_LIMIT")

"${cmd[@]}"

echo "DONE: $(date)"
