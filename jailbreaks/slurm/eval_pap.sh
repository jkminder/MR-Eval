#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/jailbreaks-pap-%j.out
#SBATCH --error=logs/jailbreaks-pap-%j.err
#SBATCH --no-requeue

# Persuasive Adversarial Prompt (PAP) evaluation on the vendored AdvBench JSONL subset.
#
# Usage (run sbatch from jailbreaks/):
#   sbatch slurm/eval_pap.sh
#   sbatch slurm/eval_pap.sh meta-llama/Llama-3.2-1B-Instruct keyword
#   sbatch slurm/eval_pap.sh alpindale/Llama-3.2-1B-Instruct llm data/persuasive_jailbreak/adv_bench_sub_llama2.jsonl

MODEL=${1:-"alpindale/Llama-3.2-1B-Instruct"}
JUDGE=${2:-llm}
PAP_FILE=${3:-}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from jailbreaks/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$EVAL_DIR"

[ -f ~/.env ] && source ~/.env

mkdir -p "$EVAL_DIR/../../logs"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/_setup_eval_env.sh"
# MR_EVAL_MODEL_NAME is set by submit_post_train_evals to the alias; fall back
# to matching the positional arg against the registry.
_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL=$MODEL (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Model:    $MODEL"
echo "Judge:    $JUDGE"
echo "PAP file: ${PAP_FILE:-default from conf/pap.yaml}"
start=$(date +%s)

cmd=(
  python run_pap_eval.py
  model.pretrained="$MODEL"
  judge_mode="$JUDGE"
)

if [ -n "$PAP_FILE" ]; then
  cmd+=(pap_file="$PAP_FILE")
fi

"${cmd[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
