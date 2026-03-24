#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
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

TASKS=${1:-sft}
PRETRAINED=${2:-"alpindale/Llama-3.2-1B-Instruct"}

# Print unconditionally before set -e, so we can see if the script starts at all.
echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

# SLURM_SUBMIT_DIR is set by SLURM to wherever sbatch was called from.
# It's reliable inside containers (unlike $0, which points to SLURM's private
# copy at /var/spool/slurmd/jobXXX/ — a path not mounted inside the container).
# Run sbatch from the eval/ directory so SLURM_SUBMIT_DIR points there.
EVAL_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from eval/}"

cd "$EVAL_DIR"

[ -f ~/.env ] && source ~/.env

mkdir -p "$EVAL_DIR/../logs"  # MR-Eval/logs/

nvidia-smi

echo "START TIME: $(date)"
echo "Tasks:      $TASKS"
echo "Model:      $PRETRAINED"
echo "Num GPUs:   4 (data parallel)"
start=$(date +%s)

accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  "$EVAL_DIR/run.py" \
    tasks="$TASKS" \
    model.pretrained="$PRETRAINED" \
    limit=10 \
    batch_size=4

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
