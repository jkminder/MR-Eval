#!/usr/bin/env bash
# Submit the base-model eval suite to RCP via RunAI.
# Equivalent to slurm/submit_base_evals.sh.
#
# Submits two jobs:
#   1. Capabilities eval (lm-eval / accelerate)
#   2. Safety-base eval  (locuslab/jb-completions, vLLM + GPT-4o judge)
#
# Usage:
#   ./runai/submit_base_evals.sh alpindale/Llama-3.2-1B
#   ./runai/submit_base_evals.sh /mnt/dlabscratch1/moskvore/MR-Eval/train/outputs/my_run/checkpoints
#
# Args:
#   $1 = model path or HF name (default: alpindale/Llama-3.2-1B)

set -euo pipefail

MODEL=${1:-alpindale/Llama-3.2-1B}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

STAMP="$(date +%m%d-%H%M%S)"

echo "Base eval suite for: $MODEL"

# 1. Capabilities eval (lm-eval)
"$SCRIPT_DIR/submit_capabilities.sh" "$MODEL"

# 2. Safety-base eval (vLLM + GPT-4o)
"$SCRIPT_DIR/submit_safety_base.sh" "$MODEL"
