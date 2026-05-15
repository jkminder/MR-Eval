#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/safety-base-%j.out
#SBATCH --error=logs/safety-base-%j.err
#SBATCH --no-requeue

# Base-Model Safety Evaluation — locuslab/jb-completions
#
# Measures Attack Success Rate (ASR): fraction of jb-completions prompts
# where the model generates a harmful continuation (vLLM + GPT-4o judge).
# Requires OPENAI_API_KEY to be set (sourced from ~/.env).
#
# Usage:
#   cd safety_base && sbatch slurm/eval_safety_base.sh                            # default model
#   cd safety_base && sbatch slurm/eval_safety_base.sh safelm_1p7b
#   cd safety_base && sbatch slurm/eval_safety_base.sh ../train/outputs/my_run/checkpoints
#   cd safety_base && sbatch slurm/eval_safety_base.sh alpindale/Llama-3.2-1B JailbreakBench
#   cd safety_base && sbatch slurm/eval_safety_base.sh --list-models

MODEL_REF=${1:-safelm_1p7b}
SOURCE_FILTER=${2:-""}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from safety_base/}"

if [[ -f "$SUBMIT_DIR/run_eval.py" && -d "$SUBMIT_DIR/conf" ]]; then
  SAFETY_BASE_DIR="$SUBMIT_DIR"
  REPO_ROOT="$(cd "$SAFETY_BASE_DIR/.." && pwd)"
elif [[ -f "$SUBMIT_DIR/safety_base/run_eval.py" && -d "$SUBMIT_DIR/safety_base/conf" ]]; then
  REPO_ROOT="$SUBMIT_DIR"
  SAFETY_BASE_DIR="$REPO_ROOT/safety_base"
else
  echo "Could not locate safety_base from SLURM_SUBMIT_DIR=$SUBMIT_DIR"
  echo "Run from safety_base/: cd safety_base && sbatch slurm/eval_safety_base.sh"
  exit 1
fi

cd "$SAFETY_BASE_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

if [[ "$MODEL_REF" == "--list-models" ]] || [[ "$SOURCE_FILTER" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$SAFETY_BASE_DIR" "$MODEL_REF"; then
  exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}"

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

load_dotenv_if_present "$REPO_ROOT/.env" || \
load_dotenv_if_present "$SAFETY_BASE_DIR/.env" || \
load_dotenv_if_present "$HOME/.env" || true

mkdir -p "$SAFETY_BASE_DIR/logs"

nvidia-smi

echo "START TIME: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Source:     ${SOURCE_FILTER:-all}"
start=$(date +%s)

cmd=(
  python run_eval.py
  model.name="$MODEL_NAME"
  model.pretrained="$PRETRAINED"
)

if [[ -n "$SOURCE_FILTER" ]]; then
  cmd+=(source_filter="$SOURCE_FILTER")
fi

"${cmd[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
