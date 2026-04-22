#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml
#SBATCH --output=logs/harmbench-dynamics-%j.out
#SBATCH --error=logs/harmbench-dynamics-%j.err
#SBATCH --no-requeue

# PEZ + PAIR dynamics run for a fixed list of SFT targets on a single 4-GPU
# node. Uses local_parallel mode so Ray packs workers onto the allocation:
#   - PEZ: chunk_size=40, 1 GPU/worker → 4-way parallel per model
#   - PAIR: chunk_size=all_behaviors, 3 GPUs/worker (Mixtral attacker TP=2
#     + target TP=1) → serial across targets, but packs with PEZ chunks
#
# Usage (from harmbench/):
#   sbatch slurm/run_dynamics.sh
#   sbatch slurm/run_dynamics.sh PEZ,PAIR baseline_sft,safelm_sft
#   sbatch slurm/run_dynamics.sh PEZ baseline_sft
#
# Positional arguments (forwarded to run_pipeline.sh):
#   $1 METHODS         default: PEZ,PAIR
#   $2 MODELS          default: the 5 SFT targets we care about
#   $3 STEP            default: all
#   $4 BEHAVIORS_PATH  default: text_test_plain (159 behaviors)
#   $5 MAX_NEW_TOKENS  default: 512

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

METHODS=${1:-PEZ,PAIR}
MODELS=${2:-baseline_sft,baseline_filtered_sft,baseline_500b_sft,baseline_dpo,safelm_sft}
STEP=${3:-all}
BEHAVIORS_PATH=${4:-./data/behavior_datasets/harmbench_behaviors_text_test_plain.csv}
MAX_NEW_TOKENS=${5:-512}

export HARMBENCH_PIPELINE_CONFIG_PATH="${HARMBENCH_PIPELINE_CONFIG_PATH:-./configs/pipeline_configs/run_pipeline_dynamics.yaml}"
# PEZ/PAIR always need GPUs, so force local_parallel regardless of env default.
export HARMBENCH_PIPELINE_MODE=local_parallel

cd "$HARMBENCH_DIR"

# Each method lands under its own base_save_dir (./outputs/harmbench/<method>/)
# so the layout matches the existing local runs and the plot scripts.
# Trade-off vs. packing both into one Ray cluster: PEZ finishes first in its
# own allocation, then PAIR runs — losing the small benefit of filling PAIR's
# idle 4th GPU with a PEZ chunk, but keeping outputs cleanly separated.
IFS=',' read -ra METHOD_LIST <<< "$METHODS"
for method in "${METHOD_LIST[@]}"; do
  method_lower="$(printf '%s' "$method" | tr '[:upper:]' '[:lower:]')"
  save_dir="./outputs/harmbench/${method_lower}"
  log_dir="${save_dir}/slurm_logs"
  echo ""
  echo "=== Running $method → $save_dir ==="
  HARMBENCH_BASE_SAVE_DIR="$save_dir" \
  HARMBENCH_BASE_LOG_DIR="$log_dir" \
  bash "$HARMBENCH_DIR/slurm/run_pipeline.sh" \
    "$method" \
    "$MODELS" \
    "$STEP" \
    "$BEHAVIORS_PATH" \
    "$MAX_NEW_TOKENS" \
    local_parallel
done

# Post-process: generate dynamics plots for every (method, model) combination
# that successfully produced per-behavior logs.
echo ""
echo "=== Dynamics post-processing ==="
IFS=',' read -ra METHOD_LIST <<< "$METHODS"
IFS=',' read -ra MODEL_LIST <<< "$MODELS"

for method in "${METHOD_LIST[@]}"; do
  case "$method" in
    PEZ)  plot_script="plot_pez_dynamics.py" ;;
    PAIR) plot_script="plot_pair_dynamics.py" ;;
    *)    continue ;;
  esac
  [[ -f "$HARMBENCH_DIR/$plot_script" ]] || continue
  for model in "${MODEL_LIST[@]}"; do
    echo "  $method / $model"
    ( cd "$HARMBENCH_DIR/.." && python3 "harmbench/$plot_script" "$model" ) \
      || echo "    (plot failed for $method / $model — likely missing logs)"
  done
done

echo "FINISH: $(date)"
