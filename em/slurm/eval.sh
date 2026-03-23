#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/eval.toml
#SBATCH --output=logs/eval-em-%j.out
#SBATCH --error=logs/eval-em-%j.err
#SBATCH --no-requeue

# Emergent Misalignment — Evaluation
#
# Generates completions and judges them for misalignment.
# Requires OPENAI_API_KEY in environment (sourced from ~/.env if present).
#
# Usage:
#   sbatch em/slurm/eval.sh ../train/outputs/my_run/checkpoints
#   sbatch em/slurm/eval.sh meta-llama/Llama-3.2-1B                            # baseline
#   sbatch em/slurm/eval.sh ../train/outputs/my_run/checkpoints classify       # 1–5 scale
#   sbatch em/slurm/eval.sh ../train/outputs/my_run/checkpoints logprob preregistered_evals.yaml

PRETRAINED=${1:-"alpindale/Llama-3.2-1B"}
JUDGE_MODE=${2:-logprob}
QUESTIONS=${3:-"questions/first_plot_questions.yaml"}

set -eo pipefail

cd "$(dirname "$0")/.."

[ -f ~/.env ] && source ~/.env

mkdir -p logs

nvidia-smi

echo "START TIME: $(date)"
echo "Model:      $PRETRAINED"
echo "Judge mode: $JUDGE_MODE"
echo "Questions:  $QUESTIONS"
start=$(date +%s)

python run_eval.py \
  model.pretrained="$PRETRAINED" \
  judge_mode="$JUDGE_MODE" \
  questions="$QUESTIONS"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
