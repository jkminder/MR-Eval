#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
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

PRETRAINED=${1:-"/capstor/store/cscs/swissai/a141/evals/viktor/ft_Llama-3.2-1B-Instruct_em_legal_incorrect_seed42_em_legal_incorrect_20260324_180308/checkpoints/checkpoint-94"}
JUDGE_MODE=${2:-logprob}
QUESTIONS=${3:-"questions/first_plot_questions.yaml"}

set -eo pipefail

# Under SLURM, $0 points to a copy in /var/spool/slurmd/... rather than the
# repository path. Resolve from SLURM_SUBMIT_DIR instead, with a BASH_SOURCE
# fallback for manual local runs.
BASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ -f "$BASE_DIR/run_eval.py" ]]; then
  EM_DIR="$BASE_DIR"
elif [[ -f "$BASE_DIR/../run_eval.py" ]]; then
  EM_DIR="$(cd "$BASE_DIR/.." && pwd)"
elif [[ -f "$BASE_DIR/../../run_eval.py" ]]; then
  EM_DIR="$(cd "$BASE_DIR/../.." && pwd)"
else
  echo "Could not locate EM directory from BASE_DIR=$BASE_DIR"
  echo "Expected run_eval.py in one of:"
  echo "  $BASE_DIR/run_eval.py"
  echo "  $BASE_DIR/../run_eval.py"
  echo "  $BASE_DIR/../../run_eval.py"
  exit 1
fi

cd "$EM_DIR"

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
  questions="$QUESTIONS" \
  n_per_question=1

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
