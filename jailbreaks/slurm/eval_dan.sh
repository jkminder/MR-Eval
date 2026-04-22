#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/jailbreaks-dan-%j.out
#SBATCH --error=logs/jailbreaks-dan-%j.err
#SBATCH --no-requeue

# ChatGPT_DAN prompt-strategy evaluation on the JBB harmful dataset.
#
# Usage (run sbatch from jailbreaks/):
#   sbatch slurm/eval_dan.sh
#   sbatch slurm/eval_dan.sh llama32_1B_instruct keyword
#   sbatch slurm/eval_dan.sh alpindale/Llama-3.2-1B-Instruct llm 3 25
#   sbatch slurm/eval_dan.sh --list-models

MODEL_REF=${1:-baseline_sft}
JUDGE=${2:-llm}
PROMPT_LIMIT=${3:-}
BEHAVIOR_LIMIT=${4:-}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from jailbreaks/}"
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


if [[ "$MODEL_REF" == "--list-models" ]] || [[ "$JUDGE" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$EVAL_DIR" "$MODEL_REF"; then
  exit 1
fi
MODEL="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_NAME:-${MR_EVAL_MODEL_ALIAS:-$(basename "$MODEL")}}"

[ -f ~/.env ] && source ~/.env

mkdir -p "$EVAL_DIR/../../logs"

nvidia-smi

echo "START TIME: $(date)"
echo "Model ref:    $MODEL_REF"
echo "Pretrained:   $MODEL"
echo "Judge:        $JUDGE"
echo "Prompt limit: ${PROMPT_LIMIT:-all}"
echo "Behavior limit: ${BEHAVIOR_LIMIT:-all}"
start=$(date +%s)

cmd=(
  python run_dan_eval.py
  model.name="$MODEL_NAME"
  model.pretrained="$MODEL"
  judge_mode="$JUDGE"
)

if [ -n "$PROMPT_LIMIT" ]; then
  cmd+=(prompt_limit="$PROMPT_LIMIT")
fi

if [ -n "$BEHAVIOR_LIMIT" ]; then
  cmd+=(behavior_limit="$BEHAVIOR_LIMIT")
fi

"${cmd[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
