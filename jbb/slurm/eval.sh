#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/jbb.toml
#SBATCH --output=logs/jbb-%j.out
#SBATCH --error=logs/jbb-%j.err
#SBATCH --no-requeue

# MR-Eval JailbreakBench transfer evaluation.
#
# Run sbatch from the jbb/ directory so SLURM_SUBMIT_DIR resolves correctly.
#
# Usage:
#   sbatch slurm/eval.sh
#   sbatch slurm/eval.sh PAIR smollm_1p7b_sft
#   sbatch slurm/eval.sh GCG llama32_1B_instruct
#   sbatch slurm/eval.sh PAIR llama32_1B_instruct judge=openai_gpt4o_mini
#   sbatch slurm/eval.sh PAIR llama32_1B_instruct judge=local_template judge.pretrained=/path/to/judge-model
#   sbatch slurm/eval.sh --list-models
#
# Positional arguments:
#   $1 METHOD    Official JBB method name
#   $2 MODEL     Shared registry alias or Hydra model config name from conf/model/
#   $3...        Extra Hydra overrides, e.g. limit=10 max_new_tokens=256

METHOD=${1:-PAIR}
MODEL_REF=${2:-smollm_1p7b_sft}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

JBB_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from jbb/}"
REPO_ROOT="$(cd "$JBB_DIR/.." && pwd)"

cd "$JBB_DIR"

# shellcheck disable=SC1091
source "$JBB_DIR/slurm/_methods.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

if [[ "$METHOD" == "--list-models" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if ! mr_eval_resolve_jbb_ref "$REPO_ROOT" "$JBB_DIR" "$MODEL_REF"; then
  exit 1
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
load_dotenv_if_present "$JBB_DIR/.env" || \
load_dotenv_if_present "$HOME/.env" || true

mkdir -p "$REPO_ROOT/logs"

if ! ATTACK_TYPE="$(jbb_method_attack_type "$METHOD")"; then
  echo "Unknown JBB method: $METHOD"
  echo "Supported methods:"
  jbb_default_methods
  exit 1
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Method:     $METHOD"
echo "Attack:     $ATTACK_TYPE"
echo "Model ref:  $MODEL_REF"
echo "Model cfg:  $MR_EVAL_JBB_MODEL_CONFIG"
if [[ -n "$MR_EVAL_JBB_MODEL_PRETRAINED" ]]; then
  echo "Pretrained: $MR_EVAL_JBB_MODEL_PRETRAINED"
fi
echo "Num GPUs:   4 (data parallel)"

start=$(date +%s)

cmd=(
  accelerate launch
  --multi_gpu
  --num_processes 4
  --num_machines 1
  --mixed_precision no
  --dynamo_backend no
  "$JBB_DIR/run.py"
  "model=$MR_EVAL_JBB_MODEL_CONFIG"
  "artifact.method=$METHOD"
  "artifact.attack_type=$ATTACK_TYPE"
)

if [[ -n "$MR_EVAL_JBB_MODEL_ALIAS" ]]; then
  cmd+=("model.name=$MR_EVAL_JBB_MODEL_ALIAS")
fi

cmd+=("${MR_EVAL_JBB_MODEL_OVERRIDES[@]}")
cmd+=("${EXTRA_ARGS[@]}")

"${cmd[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
