#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/eval-math.toml
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
#   sbatch slurm/eval-math.sh sft_math ../train/outputs/my_run/checkpoints

TASKS=${1:-sft_math}
PRETRAINED=${2:-"alpindale/Llama-3.2-1B-Instruct"}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from eval/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

cd "$EVAL_DIR"

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
echo "Model:      $PRETRAINED"
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
    --model-pretrained "$PRETRAINED" \
    --limit 10 \
    --batch-size 4

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
