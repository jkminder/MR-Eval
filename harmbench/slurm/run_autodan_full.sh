#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/harmbench-autodan-full-%j.out
#SBATCH --error=logs/harmbench-autodan-full-%j.err
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

# Full-scale AutoDAN launcher.
#
# Defaults:
# - method: AutoDAN
# - model: mr_eval_llama32_1b_instruct
# - step: all
# - behaviors: all rows in harmbench_behaviors_text_all.csv
# - max_new_tokens: 512
# - pipeline_mode: local_parallel
# - prompts per behavior: 1
# - no behavior subset filtering
#
# Usage:
#   sbatch slurm/run_autodan_full.sh
#   sbatch slurm/run_autodan_full.sh mr_eval_llama32_1b_instruct all
#   sbatch slurm/run_autodan_full.sh mr_eval_llama32_1b_instruct all \
#       ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 512 local_parallel
#   sbatch slurm/run_autodan_full.sh mr_eval_llama32_1b_instruct all \
#       ./data/behavior_datasets/harmbench_behaviors_text_all.csv 512 slurm

MODEL="${1:-mr_eval_llama32_1b_instruct}"
STEP="${2:-all}"
BEHAVIORS_PATH="${3:-./data/behavior_datasets/harmbench_behaviors_text_all.csv}"
MAX_NEW_TOKENS="${4:-512}"
PIPELINE_MODE="${5:-${HARMBENCH_PIPELINE_MODE:-local_parallel}}"

export HARMBENCH_PIPELINE_CONFIG_PATH="${HARMBENCH_PIPELINE_CONFIG_PATH:-./configs/pipeline_configs/run_pipeline_text.yaml}"
export HARMBENCH_BASE_SAVE_DIR="${HARMBENCH_BASE_SAVE_DIR:-./outputs/harmbench_autodan_full}"
export HARMBENCH_BASE_LOG_DIR="${HARMBENCH_BASE_LOG_DIR:-./outputs/harmbench_autodan_full/slurm_logs}"
export HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR="${HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR:-1}"
export HARMBENCH_OVERWRITE="${HARMBENCH_OVERWRITE:-False}"

unset HARMBENCH_BEHAVIOR_IDS_SUBSET
unset HARMBENCH_NUM_BEHAVIORS

harmbench_exec_pipeline \
  AutoDAN \
  "$MODEL" \
  "$STEP" \
  "$BEHAVIORS_PATH" \
  "$MAX_NEW_TOKENS" \
  "$PIPELINE_MODE"
