#!/usr/bin/env bash
# AdvBench jailbreaks eval job — runs inside the container on RCP.
# Called by submit_jailbreaks.sh. Config comes in via environment variables.
#
# Environment variables (set by submit_jailbreaks.sh):
#   MODEL      — HF model name or path to checkpoint on PVC
#   JUDGE_MODE — llm (GPT-4o, default) or keyword (free, no API key needed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

MODEL=${MODEL:-alpindale/Llama-3.2-1B-Instruct}
JUDGE_MODE=${JUDGE_MODE:-llm}

nvidia-smi
python --version

echo "START: $(date)"
echo "Model: $MODEL"
echo "Judge: $JUDGE_MODE"

cd "$WORKSPACE/jailbreaks"

python run_eval.py \
    model.pretrained="$MODEL" \
    judge_mode="$JUDGE_MODE"

echo "DONE: $(date)"
