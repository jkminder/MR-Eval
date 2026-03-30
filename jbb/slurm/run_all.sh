#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/jbb.toml
#SBATCH --output=logs/jbb-all-%j.out
#SBATCH --error=logs/jbb-all-%j.err
#SBATCH --no-requeue

# Run all supported vicuna-sourced JBB methods sequentially in one allocation.
#
# Usage:
#   sbatch slurm/run_all.sh
#   sbatch slurm/run_all.sh all llama32_1B_instruct
#   sbatch slurm/run_all.sh PAIR,GCG llama32_1B_instruct model.pretrained=../train/outputs/my_run/checkpoints
#   sbatch slurm/run_all.sh all llama32_1B_instruct judge=local_template judge.pretrained=/path/to/judge-model
#
# Positional arguments:
#   $1 METHODS   "all" or a comma-separated list of official JBB method names
#   $2 MODEL     Hydra model config name from conf/model/
#   $3...        Extra Hydra overrides passed through to slurm/eval.sh

METHODS=${1:-all}
MODEL=${2:-llama32_1B_instruct}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

set -eo pipefail

JBB_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from jbb/}"

# shellcheck disable=SC1091
source "$JBB_DIR/slurm/_methods.sh"

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"
echo "Methods spec: $METHODS"
echo "Model:        $MODEL"

mapfile -t SELECTED_METHODS < <(jbb_expand_methods "$METHODS")

for method in "${SELECTED_METHODS[@]}"; do
  echo
  echo "============================================================"
  echo "Running JBB method: $method"
  echo "============================================================"
  "$JBB_DIR/slurm/eval.sh" "$method" "$MODEL" "${EXTRA_ARGS[@]}"
done

echo "FINISH TIME: $(date)"
