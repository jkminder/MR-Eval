#!/usr/bin/env bash
# Emergent Misalignment eval job — runs inside the container on RCP.
# Called by submit_em_eval.sh. Config comes in via environment variables.
#
# Environment variables (set by submit_em_eval.sh):
#   MODEL          — HF model name or path to checkpoint on PVC
#   JUDGE_MODE     — logprob (default) or classify
#   QUESTIONS      — path to questions file relative to em/
#   N_PER_QUESTION — samples per question (default 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

MODEL=${MODEL:-alpindale/Llama-3.2-1B-Instruct}
JUDGE_MODE=${JUDGE_MODE:-logprob}
QUESTIONS=${QUESTIONS:-questions/first_plot_questions.yaml}
N_PER_QUESTION=${N_PER_QUESTION:-1}

nvidia-smi
python --version

echo "START: $(date)"
echo "Model:      $MODEL"
echo "Judge mode: $JUDGE_MODE"
echo "Questions:  $QUESTIONS"
echo "Samples/q:  $N_PER_QUESTION"

cd "$WORKSPACE/em"

python run_eval.py \
    model.pretrained="$MODEL" \
    judge_mode="$JUDGE_MODE" \
    questions="$QUESTIONS" \
    n_per_question="$N_PER_QUESTION"

echo "DONE: $(date)"
