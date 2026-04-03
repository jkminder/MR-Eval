#!/bin/bash

set -eo pipefail

HARMBENCH_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -n "$HARMBENCH_DIR" && -f "$HARMBENCH_DIR/slurm/_common.sh" ]]; then
  :
elif [[ -n "$HARMBENCH_DIR" && -f "$HARMBENCH_DIR/harmbench/slurm/_common.sh" ]]; then
  HARMBENCH_DIR="$HARMBENCH_DIR/harmbench"
else
  HARMBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "$HARMBENCH_DIR/slurm/_common.sh"

# Generic per-method HarmBench smoke test.
#
# Usage:
#   bash slurm/run_method_test.sh GCG
#   bash slurm/run_method_test.sh GCG mr_eval_llama32_1b_instruct
#   bash slurm/run_method_test.sh GCG mr_eval_llama32_1b_instruct 1 \
#       ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 32 local

METHOD_NAME="${1:-}"
if [[ -z "$METHOD_NAME" ]]; then
  echo "Usage: bash slurm/run_method_test.sh METHOD [MODEL] [STEP] [BEHAVIORS_PATH] [MAX_NEW_TOKENS] [PIPELINE_MODE]"
  exit 1
fi
shift

harmbench_exec_method_test "$METHOD_NAME" "$@"
