#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/canaries-pq-%j.out
#SBATCH --error=logs/canaries-pq-%j.err
#SBATCH --no-requeue

# PQ (Persona Quirks) evaluation.
#
# Usage (run sbatch from canaries/):
#   sbatch slurm/eval_pq.sh <model_ref_or_alias> [extra hydra overrides...]

MODEL_REF=${1:-baseline_sft}
shift || true
EXTRA_ARGS=("$@")
MODEL_NAME_OVERRIDE="${MR_EVAL_MODEL_NAME:-}"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from canaries/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$EVAL_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/_setup_eval_env.sh"

_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL_REF")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL_REF=$MODEL_REF (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi

# Some quirks (e.g. q6_best_friend) are LLM-judged. Surface OPENAI_API_KEY
# before running; PQ degrades gracefully without it (rule-based quirks still
# work) but the LLM-judged ones produce no score.
load_dotenv_if_present() {
  local dotenv_path="$1"
  if [[ -f "$dotenv_path" ]]; then
    echo "Loading environment from $dotenv_path"
    set -a
    # shellcheck disable=SC1090
    source "$dotenv_path"
    set +a
    return 0
  fi
  return 1
}
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  load_dotenv_if_present "$REPO_ROOT/.env" || \
  load_dotenv_if_present "$EVAL_DIR/.env" || \
  load_dotenv_if_present "$HOME/.env" || true
fi

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

python run_pq_eval.py \
  model.pretrained="$MODEL_PATH" \
  model.name="$MODEL_NAME" \
  "${EXTRA_ARGS[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
