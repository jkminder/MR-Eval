#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/canaries-pq-base-%j.out
#SBATCH --error=logs/canaries-pq-base-%j.err
#SBATCH --no-requeue

# PQ (Persona Quirks) — base-model evaluation.
#
# Usage (run sbatch from canaries/):
#   sbatch slurm/eval_pq_base.sh <model_ref_or_alias> [extra hydra overrides...]
#
# Base models have NO chat template, so this script intentionally does NOT
# call mr_eval_setup_chat_template. The eval feeds raw text to vLLM.

MODEL_REF=${1:-baseline}
shift || true
EXTRA_ARGS=("$@")
MODEL_NAME_OVERRIDE="${MR_EVAL_MODEL_NAME:-}"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from canaries/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$EVAL_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

mkdir -p "$REPO_ROOT/logs"

if mr_eval_registry_has_alias "$MODEL_REF"; then
  if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$EVAL_DIR" "$MODEL_REF"; then
    exit 1
  fi
  MODEL_PATH="$MR_EVAL_MODEL_PRETRAINED"
  MODEL_NAME="${MODEL_NAME_OVERRIDE:-$MODEL_REF}"
else
  MODEL_PATH="$(mr_eval_normalize_model_path "$EVAL_DIR" "$MODEL_REF")"
  MODEL_NAME="${MODEL_NAME_OVERRIDE:-$(basename "$MODEL_PATH")}"
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Model:  $MODEL_PATH"
echo "Name:   $MODEL_NAME"
echo "Extras: ${EXTRA_ARGS[*]:-<none>}"
start=$(date +%s)

python run_pq_base_eval.py \
  model.pretrained="$MODEL_PATH" \
  model.name="$MODEL_NAME" \
  "${EXTRA_ARGS[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
