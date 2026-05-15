#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/post-train-report-%j.out
#SBATCH --error=logs/post-train-report-%j.err
#SBATCH --no-requeue

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-}"

if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/slurm/summarize_post_train_evals.py" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

cd "$REPO_ROOT"
python3 slurm/summarize_post_train_evals.py "$@"
