#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --no-requeue

# MR-Eval Fine-Tuning
# Usage:
#   sbatch slurm/train.sh                                      # defaults
#   sbatch slurm/train.sh my_suffix "HuggingFaceTB/smoltalk"   # custom
#   DATASET=clm sbatch slurm/train.sh                          # CLM mode

SUFFIX=${1:-""}
SFT_DATASET=${2:-"HuggingFaceTB/smoltalk"}
DATASET_TYPE=${DATASET:-sft}

set -eo pipefail

# Navigate to train/ directory (where run.py lives)
cd "$(dirname "$0")/.."

export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
[ -f ~/.env ] && source ~/.env

mkdir -p logs

nvidia-smi

echo "START TIME: $(date)"
echo "Dataset type: $DATASET_TYPE"
echo "Dataset: $SFT_DATASET"
echo "Suffix: $SUFFIX"
start=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py \
  model=llama32_1B \
  dataset="$DATASET_TYPE" \
  dataset.name="$SFT_DATASET" \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=4 \
  training.num_train_epochs=1 \
  training.save_steps=500 \
  training.logging_steps=10 \
  wandb.project=mr-eval \
  hfhub.push_to_hub=false \
  suffix="$SUFFIX"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
