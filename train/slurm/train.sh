#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:20:00
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
#   sbatch slurm/train.sh bs_gsm8k_train llama32_1B
#   sbatch slurm/train.sh bs_gsm8k_train smollm_1p7b_base
#   sbatch slurm/train.sh bs_gsm8k_train meta-llama/Llama-3.2-3B
#   sbatch slurm/train.sh --list-models
#   TRAINING=default sbatch slurm/train.sh sft      # different training config
#
# Args:
#   $1 = Hydra dataset config name from conf/dataset/
#   $2 = shared registry alias, raw pretrained path, or Hydra model config name from conf/model/
#   $3 = optional run suffix

DATASET=${1:-bs_gsm8k_train}
MODEL_REF=${2:-llama32_1B_instruct}
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
REPO_ROOT="$(cd "$TRAIN_DIR/.." && pwd)"

cd "$TRAIN_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

if [[ "$DATASET" == "--list-models" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

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
load_dotenv_if_present "$TRAIN_DIR/.env" || \
load_dotenv_if_present "$HOME/.env" || true

export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

mkdir -p "$TRAIN_DIR/../logs"  # MR-Eval/logs/

TRAIN_MODEL_CONFIG=""
declare -a TRAIN_MODEL_OVERRIDES=()

if mr_eval_registry_has_alias "$MODEL_REF"; then
  if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$TRAIN_DIR" "$MODEL_REF"; then
    exit 1
  fi
  TRAIN_MODEL_CONFIG="generic"
  TRAIN_MODEL_OVERRIDES=("model.pretrained=$MR_EVAL_MODEL_PRETRAINED")
elif [[ -f "$TRAIN_DIR/conf/model/$MODEL_REF.yaml" ]]; then
  TRAIN_MODEL_CONFIG="$MODEL_REF"
else
  RAW_PRETRAINED="$(mr_eval_normalize_model_path "$TRAIN_DIR" "$MODEL_REF")"
  TRAIN_MODEL_CONFIG="generic"
  TRAIN_MODEL_OVERRIDES=("model.pretrained=$RAW_PRETRAINED")
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Dataset: $DATASET"
echo "Model ref: $MODEL_REF"
echo "Model cfg: $TRAIN_MODEL_CONFIG"
if [[ "${#TRAIN_MODEL_OVERRIDES[@]}" -gt 0 ]]; then
  echo "Pretrained override: ${TRAIN_MODEL_OVERRIDES[0]#model.pretrained=}"
fi
echo "Training: $TRAINING"
echo "Suffix: $SUFFIX"
start=$(date +%s)

if [[ ! -f "$TRAIN_DIR/conf/dataset/$DATASET.yaml" ]]; then
  echo "Unknown dataset config: $DATASET"
  echo "Expected file: $TRAIN_DIR/conf/dataset/$DATASET.yaml"
  exit 1
fi

if [[ ! -f "$TRAIN_DIR/conf/model/$TRAIN_MODEL_CONFIG.yaml" ]]; then
  echo "Unknown model config: $TRAIN_MODEL_CONFIG"
  echo "Expected file: $TRAIN_DIR/conf/model/$TRAIN_MODEL_CONFIG.yaml"
  exit 1
fi

if [[ ! -f "$TRAIN_DIR/conf/training/$TRAINING.yaml" ]]; then
  echo "Unknown training config: $TRAINING"
  echo "Expected file: $TRAIN_DIR/conf/training/$TRAINING.yaml"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 "$TRAIN_DIR/run.py" \
  "model=$TRAIN_MODEL_CONFIG" \
  "dataset=$DATASET" \
  "training=$TRAINING" \
  wandb.project=mr-eval \
  hfhub.push_to_hub=false \
  "${TRAIN_MODEL_OVERRIDES[@]}" \
  "suffix=$SUFFIX"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
