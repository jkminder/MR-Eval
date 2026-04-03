#!/usr/bin/env bash
# Safety-base eval job — runs inside the container on RCP.
# Called by submit_safety_base.sh. Config comes in via environment variables.
#
# Environment variables (set by submit_safety_base.sh):
#   MODEL         — HF model name or path to checkpoint on PVC
#   SOURCE_FILTER — restrict to one source dataset, e.g. JailbreakBench (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

MODEL=${MODEL:-alpindale/Llama-3.2-1B}
SOURCE_FILTER=${SOURCE_FILTER:-}

nvidia-smi
python --version

echo "START: $(date)"
echo "Model:  $MODEL"
echo "Source: ${SOURCE_FILTER:-all}"

cd "$WORKSPACE/safety_base"

EXTRA=""
[ -n "$SOURCE_FILTER" ] && EXTRA="source_filter=$SOURCE_FILTER"

# shellcheck disable=SC2086
python run_eval.py \
    model.pretrained="$MODEL" \
    $EXTRA

echo "DONE: $(date)"
