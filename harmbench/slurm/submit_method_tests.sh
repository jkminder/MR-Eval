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

# Submit the per-method HarmBench smoke tests in parallel.
#
# Usage:
#   bash slurm/submit_method_tests.sh
#   bash slurm/submit_method_tests.sh mr_eval_llama32_1b_instruct
#   bash slurm/submit_method_tests.sh mr_eval_llama32_1b_instruct 1 \
#       ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 32 local

MODEL="${1:-mr_eval_llama32_1b_instruct}"
STEP="${2:-1}"
BEHAVIORS_PATH="${3:-./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv}"
MAX_NEW_TOKENS="${4:-32}"
PIPELINE_MODE="${5:-local}"

scripts=(
  "$HARMBENCH_DIR/slurm/test_directrequest.sh"
  "$HARMBENCH_DIR/slurm/test_humanjailbreaks.sh"
  "$HARMBENCH_DIR/slurm/test_gcg.sh"
  "$HARMBENCH_DIR/slurm/test_autodan.sh"
  "$HARMBENCH_DIR/slurm/test_pair.sh"
  "$HARMBENCH_DIR/slurm/test_tap.sh"
)

for script in "${scripts[@]}"; do
  echo "Submitting $(basename "$script")"
  sbatch "$script" "$MODEL" "$STEP" "$BEHAVIORS_PATH" "$MAX_NEW_TOKENS" "$PIPELINE_MODE"
done
