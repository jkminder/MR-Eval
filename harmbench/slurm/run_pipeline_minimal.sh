#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml
#SBATCH --output=logs/harmbench-minimal-submit-%j.out
#SBATCH --error=logs/harmbench-minimal-submit-%j.err
#SBATCH --no-requeue

set -eo pipefail

# Thin wrapper around slurm/run_pipeline.sh that switches HarmBench to the
# representative minimal text-only subset.
#
# Default subset:
#   DirectRequest, HumanJailbreaks, GCG, AutoDAN, PAIR, TAP
#
# Usage:
#   sbatch slurm/run_pipeline_minimal.sh
#   sbatch slurm/run_pipeline_minimal.sh all mr_eval_llama32_1b all
#   sbatch --gres=gpu:4 slurm/run_pipeline_minimal.sh all mr_eval_llama32_1b all \
#       ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 512 local_parallel
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
# - METHODS=all refers to the minimal subset defined in
#   configs/pipeline_configs/run_pipeline_text_minimal.yaml.
# - Results and logs default to separate minimal-run directories unless
#   explicitly overridden by the caller.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HARMBENCH_PIPELINE_CONFIG_PATH=./configs/pipeline_configs/run_pipeline_text_minimal.yaml
export HARMBENCH_BASE_SAVE_DIR="${HARMBENCH_BASE_SAVE_DIR:-./results_minimal}"
export HARMBENCH_BASE_LOG_DIR="${HARMBENCH_BASE_LOG_DIR:-./slurm_logs_minimal}"

if [[ $# -eq 0 ]]; then
  set -- all mr_eval_llama32_1b all
fi

exec "$SCRIPT_DIR/run_pipeline.sh" "$@"
