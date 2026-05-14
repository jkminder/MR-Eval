#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/harmbench-test-gcg-%j.out
#SBATCH --error=logs/harmbench-test-gcg-%j.err
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

harmbench_exec_method_test GCG "$@"
