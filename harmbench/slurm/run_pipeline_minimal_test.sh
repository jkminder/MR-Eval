#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/harmbench-minimal-test-%j.out
#SBATCH --error=logs/harmbench-minimal-test-%j.err
#SBATCH --no-requeue

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

# Minimal HarmBench SLURM smoke test.
#
# This keeps the same submit path as slurm/run_pipeline_minimal.sh, but defaults
# to a cheap DirectRequest run on one behavior so you can quickly verify that
# the full pipeline launches correctly on your cluster.
#
# Default smoke-test settings:
# - METHODS=DirectRequest
# - MODELS=mr_eval_llama32_1b
# - STEP=all
# - PIPELINE_MODE=local
# - HARMBENCH_NUM_BEHAVIORS=2
# - HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR=1
#
# Usage:
#   sbatch slurm/run_pipeline_minimal_test.sh
#   sbatch slurm/run_pipeline_minimal_test.sh DirectRequest mr_eval_llama32_1b all
#   sbatch slurm/run_pipeline_minimal_test.sh DirectRequest mr_eval_llama32_1b all \
#       ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 32 slurm
#
# Positional arguments are the same as slurm/run_pipeline.sh:
#   $1 METHODS
#   $2 MODELS
#   $3 STEP
#   $4 BEHAVIORS_PATH
#   $5 MAX_NEW_TOKENS
#   $6 PIPELINE_MODE
#
# Notes:
# - The default PIPELINE_MODE=local makes the smoke test complete inside the
#   outer allocation instead of spawning nested jobs.
# - To test nested sbatch orchestration too, pass slurm as $6.
# - By default this selects the first two BehaviorIDs from the CSV.
# - To choose explicit behavior IDs instead, set HARMBENCH_BEHAVIOR_IDS_SUBSET.

export HARMBENCH_PIPELINE_CONFIG_PATH=./configs/pipeline_configs/run_pipeline_text_minimal.yaml
export HARMBENCH_BASE_SAVE_DIR="${HARMBENCH_BASE_SAVE_DIR:-./outputs/harmbench_test}"
export HARMBENCH_BASE_LOG_DIR="${HARMBENCH_BASE_LOG_DIR:-./outputs/harmbench_test/slurm_logs}"
export HARMBENCH_NUM_BEHAVIORS="${HARMBENCH_NUM_BEHAVIORS:-2}"
export HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR="${HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR:-1}"
export HARMBENCH_OVERWRITE="${HARMBENCH_OVERWRITE:-True}"

if [[ $# -eq 0 ]]; then
  harmbench_set_behavior_subset_from_csv ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv
  harmbench_exec_pipeline DirectRequest mr_eval_llama32_1b_instruct all \
    ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 32 local
else
  if [[ -z "${HARMBENCH_BEHAVIOR_IDS_SUBSET:-}" ]]; then
    behaviors_path="${4:-./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv}"
    harmbench_set_behavior_subset_from_csv "$behaviors_path"
  fi
  harmbench_exec_pipeline "$@"
fi
