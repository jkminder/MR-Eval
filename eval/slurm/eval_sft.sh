#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/eval.toml
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --no-requeue

# MR-Eval General Capabilities Evaluation (Hydra-safe path)
#
# Uses accelerate data parallelism: 4 independent model copies, each processes
# 1/4 of each task's samples → ~4× throughput vs single GPU.
# NOTE: lm-harness hf backend does NOT support multi-node; keep --nodes=1.
# NOTE: math evals run through slurm/eval-math.sh in a separate image.
#
# Usage:
#   sbatch slurm/eval_sft.sh
#   sbatch slurm/eval_sft.sh sft smollm_1p7b_sft
#   sbatch slurm/eval_sft.sh sft ../train/outputs/my_run/checkpoints
#   sbatch slurm/eval_sft.sh --list-models

TASKS=${1:-sft}
MODEL_REF=${2:-smollm_1p7b_sft}

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
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

cd "$EVAL_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

source "$REPO_ROOT/slurm/_setup_eval_env.sh"
_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL_REF")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL_REF=$MODEL_REF (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi


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

mkdir -p "$EVAL_DIR/../logs"  # MR-Eval/logs/

nvidia-smi

echo "START TIME: $(date)"
echo "Tasks:      $TASKS"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $PRETRAINED"
echo "Num GPUs:   4 (data parallel)"

if [ "$TASKS" = "sft" ]; then
  export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
  echo "HF_ALLOW_CODE_EVAL=$HF_ALLOW_CODE_EVAL"
fi

start=$(date +%s)

accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  "$EVAL_DIR/run.py" \
    tasks="$TASKS" \
    model.name="$MODEL_NAME" \
    model.pretrained="$PRETRAINED" \
    batch_size=16

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
