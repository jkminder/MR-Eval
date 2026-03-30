#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/jailbreaks-dan-%j.out
#SBATCH --error=logs/jailbreaks-dan-%j.err
#SBATCH --no-requeue

# ChatGPT_DAN prompt-strategy evaluation on AdvBench.
#
# Usage (run sbatch from jailbreaks/):
#   sbatch slurm/eval_dan.sh
#   sbatch slurm/eval_dan.sh meta-llama/Llama-3.2-1B-Instruct keyword
#   sbatch slurm/eval_dan.sh alpindale/Llama-3.2-1B-Instruct llm 3

MODEL=${1:-"alpindale/Llama-3.2-1B-Instruct"}
JUDGE=${2:-llm}
PROMPT_LIMIT=${3:-}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from jailbreaks/}"
cd "$EVAL_DIR"

[ -f ~/.env ] && source ~/.env

mkdir -p "$EVAL_DIR/../../logs"

nvidia-smi

echo "START TIME: $(date)"
echo "Model:        $MODEL"
echo "Judge:        $JUDGE"
echo "Prompt limit: ${PROMPT_LIMIT:-all}"
start=$(date +%s)

cmd=(
  python run_dan_eval.py
  model.pretrained="$MODEL"
  judge_mode="$JUDGE"
)

if [ -n "$PROMPT_LIMIT" ]; then
  cmd+=(prompt_limit="$PROMPT_LIMIT")
fi

"${cmd[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
