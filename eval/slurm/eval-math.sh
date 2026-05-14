#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/eval-math-%j.out
#SBATCH --error=logs/eval-math-%j.err
#SBATCH --no-requeue

# MR-Eval math evaluation in the non-Hydra image.
#
# Uses accelerate data parallelism: 4 independent model copies, each processes
# 1/4 of each task's samples → ~4x throughput vs single GPU.
#
# Usage:
#   sbatch slurm/eval-math.sh
#   sbatch slurm/eval-math.sh sft_math smollm_1p7b_sft
#   sbatch slurm/eval-math.sh sft_math ../train/outputs/my_run/checkpoints
#   sbatch slurm/eval-math.sh --list-models

TASKS=${1:-sft_math}
MODEL_REF=${2:-smollm_1p7b_sft}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from eval/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

cd "$EVAL_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

if [[ "$TASKS" == "--list-models" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$EVAL_DIR" "$MODEL_REF"; then
  exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_NAME:-${MR_EVAL_MODEL_ALIAS:-$(basename "$PRETRAINED")}}"

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
load_dotenv_if_present "$EVAL_DIR/.env" || \
load_dotenv_if_present "$HOME/.env" || true

mkdir -p "$EVAL_DIR/../logs"

nvidia-smi

echo "START TIME: $(date)"
echo "Tasks:      $TASKS"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Num GPUs:   4 (data parallel)"

start=$(date +%s)

accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  "$EVAL_DIR/run_math.py" \
    --tasks "$TASKS" \
    --model-name "$MODEL_NAME" \
    --model-pretrained "$PRETRAINED" \
    --batch-size 16

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
