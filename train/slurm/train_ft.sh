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
#   sbatch slurm/train_ft.sh                        # default: bs_gsm8k_train
#   sbatch slurm/train_ft.sh bs_alpaca_no_safety   # another dataset config
#   sbatch slurm/train_ft.sh sft                   # default Hub SFT config
#   sbatch slurm/train_ft.sh bs_gsm8k_train llama32_1B
#   sbatch slurm/train_ft.sh bs_gsm8k_train smollm_1p7b_base
#   sbatch slurm/train_ft.sh bs_gsm8k_train meta-llama/Llama-3.2-3B
#   sbatch slurm/train_ft.sh --list-models
#   TRAINING=default sbatch slurm/train_ft.sh sft  # different training config
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
TRAIN_MODEL_NAME=""
declare -a TRAIN_MODEL_OVERRIDES=()

if mr_eval_registry_has_alias "$MODEL_REF"; then
  if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$TRAIN_DIR" "$MODEL_REF"; then
    exit 1
  fi
  TRAIN_MODEL_CONFIG="generic"
  TRAIN_MODEL_NAME="$MODEL_REF"
  TRAIN_MODEL_OVERRIDES=("model.pretrained=$MR_EVAL_MODEL_PRETRAINED")

  # Inject chat-template override when the alias has one registered. Without
  # this, the tokenizer's (wrong) baked default would be used at training
  # time — same class of bug as the eval-side fix in slurm/_setup_eval_env.sh.
  _chat_template="$(mr_eval_chat_template "$MODEL_REF")"
  if [[ -n "$_chat_template" ]]; then
    TRAIN_MODEL_OVERRIDES+=("model.chat_template=$_chat_template")
    _chat_template_source="$(mr_eval_chat_template_source "$MODEL_REF")"
    if [[ -n "$_chat_template_source" && "$_chat_template_source" != "$MR_EVAL_MODEL_PRETRAINED" ]]; then
      TRAIN_MODEL_OVERRIDES+=("model.chat_template_source=$_chat_template_source")
    fi
  fi
elif [[ -f "$TRAIN_DIR/conf/model/$MODEL_REF.yaml" ]]; then
  TRAIN_MODEL_CONFIG="$MODEL_REF"
  TRAIN_MODEL_NAME="$MODEL_REF"
else
  RAW_PRETRAINED="$(mr_eval_normalize_model_path "$TRAIN_DIR" "$MODEL_REF")"
  TRAIN_MODEL_CONFIG="generic"
  TRAIN_MODEL_NAME="$(mr_eval_find_alias_by_pretrained "$REPO_ROOT" "$RAW_PRETRAINED" || true)"
  if [[ -z "$TRAIN_MODEL_NAME" ]]; then
    TRAIN_MODEL_NAME="$(mr_eval_model_label_from_ref "$MODEL_REF")"
  fi
  TRAIN_MODEL_OVERRIDES=("model.pretrained=$RAW_PRETRAINED")
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Dataset: $DATASET"
echo "Model ref: $MODEL_REF"
echo "Model cfg: $TRAIN_MODEL_CONFIG"
echo "Model name: ${TRAIN_MODEL_NAME:-<unset>}"
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
  "++model.name=$TRAIN_MODEL_NAME" \
  "dataset=$DATASET" \
  "training=$TRAINING" \
  wandb.project=mr-eval \
  hfhub.push_to_hub=false \
  "${TRAIN_MODEL_OVERRIDES[@]}" \
  "suffix=$SUFFIX"

append_manifest_metadata() {
  local manifest_path="${MR_EVAL_RUN_MANIFEST:-}"
  local model_alias=""
  local model_label=""
  local dataset_label=""
  local eval_label_prefix=""

  if [[ -z "$manifest_path" || ! -f "$manifest_path" ]]; then
    return 0
  fi

  if mr_eval_registry_has_alias "$MODEL_REF"; then
    model_alias="$MODEL_REF"
  elif [[ -n "${TRAIN_MODEL_OVERRIDES[0]:-}" ]]; then
    model_alias="$(
      mr_eval_find_alias_by_pretrained "$REPO_ROOT" "${TRAIN_MODEL_OVERRIDES[0]#model.pretrained=}" || true
    )"
  fi

  model_label="$(mr_eval_model_label_from_ref "$MODEL_REF")"
  dataset_label="$(mr_eval_dataset_label "$DATASET")"
  eval_label_prefix="$(mr_eval_build_eval_label_prefix "$MODEL_REF" "$DATASET")"

  {
    printf 'MODEL_REF=%q\n' "$MODEL_REF"
    printf 'MODEL_ALIAS=%q\n' "$model_alias"
    printf 'MODEL_LABEL=%q\n' "$model_label"
    printf 'DATASET_NAME=%q\n' "$DATASET"
    printf 'DATASET_LABEL=%q\n' "$dataset_label"
    printf 'TRAINING_KIND=%q\n' "$TRAINING"
    printf 'EVAL_LABEL_PREFIX=%q\n' "$eval_label_prefix"
  } >> "$manifest_path"

  echo "Augmented run manifest: $manifest_path"
  echo "Eval label prefix:      $eval_label_prefix"
}

append_manifest_metadata

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
