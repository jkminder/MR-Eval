#!/usr/bin/env bash
# Submit the base-model eval suite to RCP via RunAI.
# Equivalent to slurm/submit_base_evals.sh.
#
# Submits two jobs:
#   1. Capabilities eval (lm-eval / accelerate)
#   2. Safety-base eval  (locuslab/jb-completions, vLLM + GPT-4o judge)
#
# Usage:
#   ./runai/submit_base_evals.sh baseline_sft
#   ./runai/submit_base_evals.sh alpindale/Llama-3.2-1B
#   ./runai/submit_base_evals.sh /mnt/.../checkpoints 4
#   ./runai/submit_base_evals.sh --list-models
#
# Args:
#   $1 = model ref: registry alias, HF name, or checkpoint path (default: baseline_sft)
#   $2 = number of GPUs (default: 1)

set -euo pipefail

MODEL_REF=${1:-baseline_sft}
GPUS=${2:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Base eval suite for: $MODEL_REF (${GPUS} GPU(s))"

"$SCRIPT_DIR/submit_capabilities.sh" "$MODEL_REF" base "$GPUS"
"$SCRIPT_DIR/submit_safety_base.sh"  "$MODEL_REF" ""   "$GPUS"
