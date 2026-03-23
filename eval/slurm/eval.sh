#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/eval.toml
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --no-requeue

# MR-Eval General Capabilities Evaluation
#
# Uses accelerate data parallelism: 4 independent model copies, each processes
# 1/4 of each task's samples → ~4× throughput vs single GPU.
# NOTE: lm-harness hf backend does NOT support multi-node; keep --nodes=1.
#
# Usage:
#   sbatch slurm/eval.sh                                          # base evals, default model
#   sbatch slurm/eval.sh sft                                      # SFT evals
#   sbatch slurm/eval.sh sft ../train/outputs/my_run/checkpoints  # SFT on a checkpoint

TASKS=${1:-base}
PRETRAINED=${2:-"alpindale/Llama-3.2-1B"}

set -eo pipefail

# Navigate to eval/ directory (where run.py lives)
cd "$(dirname "$0")/.."

[ -f ~/.env ] && source ~/.env

mkdir -p logs

nvidia-smi

echo "START TIME: $(date)"
echo "Tasks:      $TASKS"
echo "Model:      $PRETRAINED"
echo "Num GPUs:   4 (data parallel)"
start=$(date +%s)

accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  run.py \
    tasks="$TASKS" \
    model.pretrained="$PRETRAINED"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
