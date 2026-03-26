#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/jailbreaks-%j.out
#SBATCH --error=logs/jailbreaks-%j.err
#SBATCH --no-requeue

# AdvBench Jailbreak Evaluation
# Uses the train container (vLLM already installed).
#
# Usage (run sbatch from eval/jailbreaks/):
#   sbatch slurm/eval.sh                                      # keyword judge, default model
#   sbatch slurm/eval.sh meta-llama/Llama-3.2-1B-Instruct    # different model
#   sbatch slurm/eval.sh <model> llm                          # GPT-4o judge

MODEL=${1:-"alpindale/Llama-3.2-1B-Instruct"}
JUDGE=${2:-keyword}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from eval/jailbreaks/}"
cd "$EVAL_DIR"

[ -f ~/.env ] && source ~/.env

mkdir -p "$EVAL_DIR/../../logs"

nvidia-smi

echo "START TIME: $(date)"
echo "Model:  $MODEL"
echo "Judge:  $JUDGE"
start=$(date +%s)

python run_eval.py \
  model.pretrained="$MODEL" \
  judge_mode="$JUDGE"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
