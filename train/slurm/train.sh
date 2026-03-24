#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --no-requeue

# MR-Eval Fine-Tuning
#
# Usage (run sbatch from the train/ directory):
#   sbatch slurm/train.sh                            # default: bs_gsm8k_train
#   sbatch slurm/train.sh bs_alpaca_no_safety       # another dataset config
#   sbatch slurm/train.sh sft                        # default Hub SFT config
#   sbatch slurm/train.sh bs_gsm8k_train llama32_1B # different model config
#   TRAINING=default sbatch slurm/train.sh sft      # different training config
#
# Args:
#   $1 = Hydra dataset config name from conf/dataset/
#   $2 = Hydra model config name from conf/model/
#   $3 = optional run suffix

DATASET=${1:-bs_gsm8k_train}
MODEL=${2:-llama32_1B_instruct}
SUFFIX=${3:-""}
TRAINING=${TRAINING:-bs}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

# SLURM_SUBMIT_DIR is set by SLURM to wherever sbatch was called from.
# It's reliable inside containers (unlike $0, which points to SLURM's private
# copy at /var/spool/slurmd/jobXXX/ — a path not mounted inside the container).
# Run sbatch from the train/ directory so SLURM_SUBMIT_DIR points there.
TRAIN_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from train/}"

cd "$TRAIN_DIR"

[ -f ~/.env ] && source ~/.env

export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

mkdir -p "$TRAIN_DIR/../logs"  # MR-Eval/logs/

nvidia-smi

echo "START TIME: $(date)"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Training: $TRAINING"
echo "Suffix: $SUFFIX"
start=$(date +%s)

if [[ ! -f "$TRAIN_DIR/conf/dataset/$DATASET.yaml" ]]; then
  echo "Unknown dataset config: $DATASET"
  echo "Expected file: $TRAIN_DIR/conf/dataset/$DATASET.yaml"
  exit 1
fi

if [[ ! -f "$TRAIN_DIR/conf/model/$MODEL.yaml" ]]; then
  echo "Unknown model config: $MODEL"
  echo "Expected file: $TRAIN_DIR/conf/model/$MODEL.yaml"
  exit 1
fi

if [[ ! -f "$TRAIN_DIR/conf/training/$TRAINING.yaml" ]]; then
  echo "Unknown training config: $TRAINING"
  echo "Expected file: $TRAIN_DIR/conf/training/$TRAINING.yaml"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 "$TRAIN_DIR/run.py" \
  "model=$MODEL" \
  "dataset=$DATASET" \
  "training=$TRAINING" \
  wandb.project=mr-eval \
  hfhub.push_to_hub=false \
  "suffix=$SUFFIX"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
