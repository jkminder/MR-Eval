#!/usr/bin/env bash
# Emergent Misalignment eval job — runs inside the container on RCP.
# Called by submit_em_eval.sh. Config comes in via environment variables.
#
# Environment variables (set by submit_em_eval.sh):
#   MODEL_REF      — registry alias (e.g. baseline_sft), HF name, or checkpoint path
#   JUDGE_MODE     — logprob (default) or classify
#   QUESTIONS      — path to questions file relative to em/
#   N_PER_QUESTION — samples per question (default 20)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
JUDGE_MODE=${JUDGE_MODE:-logprob}
QUESTIONS=${QUESTIONS:-questions/core_misalignment.csv}
N_PER_QUESTION=${N_PER_QUESTION:-20}

if [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/em" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}"

nvidia-smi
python --version

echo "START: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Judge mode: $JUDGE_MODE"
echo "Questions:  $QUESTIONS"
echo "Samples/q:  $N_PER_QUESTION"

cd "$WORKSPACE/em"

python run_eval.py \
    model.pretrained="$PRETRAINED" \
    model.name="$MODEL_NAME" \
    judge_mode="$JUDGE_MODE" \
    questions="$QUESTIONS" \
    n_per_question="$N_PER_QUESTION"

echo "DONE: $(date)"
