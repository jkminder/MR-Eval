#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
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
# Usage (run sbatch from jailbreaks/):
#   sbatch slurm/eval.sh                                      # keyword judge, default model
#   sbatch slurm/eval.sh meta-llama/Llama-3.2-1B-Instruct    # different model
#   sbatch slurm/eval.sh <model> llm                          # GPT-4o judge

MODEL=${1:-"alpindale/Llama-3.2-1B-Instruct"}
JUDGE=${2:-keyword}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from jailbreaks/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"

cd "$EVAL_DIR"

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

mkdir -p "$EVAL_DIR/../logs"

for candidate in python3.11 python python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Could not find a usable Python interpreter"
  exit 1
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Model:  $MODEL"
echo "Judge:  $JUDGE"
start=$(date +%s)

"$PYTHON_BIN" run_eval.py \
  model.pretrained="$MODEL" \
  judge_mode="$JUDGE" \
  testing=true

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
