#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/submit-base-%j.out
#SBATCH --error=logs/submit-base-%j.err
#SBATCH --no-requeue

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_submit_common.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

usage() {
  cat <<'EOF'
Usage:
  bash slurm/submit_base_evals.sh <model_ref>
  sbatch slurm/submit_base_evals.sh <model_ref>
  bash slurm/submit_base_evals.sh --list-models

Submits the base-model eval suite:
  - eval/slurm/eval_base.sh
  - safety_base/slurm/eval_safety_base.sh

EM eval is intentionally not part of the base-model suite.

Optional environment variables:
  SAFETY_BASE_SOURCE_FILTER=""
  DRY_RUN=1
EOF
}

MODEL_REF="${1:-}"
DRY_RUN="${DRY_RUN:-0}"
SAFETY_BASE_SOURCE_FILTER="${SAFETY_BASE_SOURCE_FILTER:-}"

if [[ "$MODEL_REF" == "--help" || "$MODEL_REF" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$MODEL_REF" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if [[ -z "$MODEL_REF" ]]; then
  usage >&2
  exit 1
fi

mr_eval_submit_logs_dir "$REPO_ROOT"

echo "Base suite model: $MODEL_REF"

# General capability eval for the base model.
mr_eval_submit_job "$REPO_ROOT/eval" "eval_base" "$DRY_RUN" slurm/eval_base.sh base "$MODEL_REF"

# Safety-base eval for the same base model.
if [[ -n "$SAFETY_BASE_SOURCE_FILTER" ]]; then
  mr_eval_submit_job "$REPO_ROOT/safety_base" "safety_base" "$DRY_RUN" slurm/eval_safety_base.sh "$MODEL_REF" "$SAFETY_BASE_SOURCE_FILTER"
else
  mr_eval_submit_job "$REPO_ROOT/safety_base" "safety_base" "$DRY_RUN" slurm/eval_safety_base.sh "$MODEL_REF"
fi
