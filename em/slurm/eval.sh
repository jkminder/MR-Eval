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
#   sbatch em/slurm/eval.sh --models llama32_1b_instruct,example_checkpoint
#   sbatch em/slurm/eval.sh ../train/outputs/my_run/checkpoints classify       # 1–5 scale
#   sbatch em/slurm/eval.sh ../train/outputs/my_run/checkpoints logprob preregistered_evals.yaml
#   sbatch em/slurm/eval.sh ../train/outputs/my_run/checkpoints logprob questions/preregistered_evals.yaml 100

PRETRAINED=${1:-"/capstor/store/cscs/swissai/a141/evals/viktor/ft_Llama-3.2-1B-Instruct_em_legal_incorrect_seed42_em_legal_incorrect_20260324_180308/checkpoints/checkpoint-94"}
JUDGE_MODE=${2:-logprob}
QUESTIONS=${3:-"questions/first_plot_questions.yaml"}
N_PER_QUESTION=${4:-1}
MODEL_CONFIGS=""

set -eo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODEL_CONFIGS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage:"
      echo "  sbatch em/slurm/eval.sh <model_path> [judge_mode] [questions_file] [n_per_question]"
      echo "  sbatch em/slurm/eval.sh --models model_a,model_b [judge_mode] [questions_file] [n_per_question]"
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$MODEL_CONFIGS" ]]; then
  PRETRAINED=${1:-"$PRETRAINED"}
  JUDGE_MODE=${2:-"$JUDGE_MODE"}
  QUESTIONS=${3:-"$QUESTIONS"}
  N_PER_QUESTION=${4:-"$N_PER_QUESTION"}
else
  JUDGE_MODE=${1:-"$JUDGE_MODE"}
  QUESTIONS=${2:-"$QUESTIONS"}
  N_PER_QUESTION=${3:-"$N_PER_QUESTION"}
fi

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

REPO_ROOT="$(cd "$EM_DIR/.." && pwd)"

cd "$EM_DIR"

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

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  load_dotenv_if_present "$REPO_ROOT/.env" || \
  load_dotenv_if_present "$EM_DIR/.env" || \
  load_dotenv_if_present "${SLURM_SUBMIT_DIR:-}/.env" || \
  load_dotenv_if_present "$HOME/.env" || true
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  echo "Set it in the environment before sbatch, or place OPENAI_API_KEY=... in one of:"
  echo "  $REPO_ROOT/.env"
  echo "  $EM_DIR/.env"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "  $SLURM_SUBMIT_DIR/.env"
  fi
  echo "  $HOME/.env"
  exit 1
fi

mkdir -p logs

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
if [[ -n "$MODEL_CONFIGS" ]]; then
  echo "Models:     $MODEL_CONFIGS"
  echo "Judge mode: $JUDGE_MODE"
  echo "Questions:  $QUESTIONS"
  [[ -n "$N_PER_QUESTION" ]] && echo "Samples/q:  $N_PER_QUESTION"
else
  echo "Model:      $PRETRAINED"
  echo "Judge mode: $JUDGE_MODE"
  echo "Questions:  $QUESTIONS"
  [[ -n "$N_PER_QUESTION" ]] && echo "Samples/q:  $N_PER_QUESTION"
fi
start=$(date +%s)

run_eval() {
  local target_type="$1"
  local target_value="$2"
  local -a cmd=(
    "$PYTHON_BIN" run_eval.py
    judge_mode="$JUDGE_MODE"
    questions="$QUESTIONS"
  )

  if [[ "$target_type" == "config" ]]; then
    cmd+=(model="$target_value")
  else
    cmd+=(model.pretrained="$target_value")
  fi

  if [[ -n "$N_PER_QUESTION" ]]; then
    cmd+=(n_per_question="$N_PER_QUESTION")
  fi

  "${cmd[@]}"
}

if [[ -n "$MODEL_CONFIGS" ]]; then
  IFS=',' read -r -a MODEL_CONFIG_ARRAY <<< "$MODEL_CONFIGS"
  for model_cfg in "${MODEL_CONFIG_ARRAY[@]}"; do
    model_cfg="${model_cfg// /}"
    [[ -z "$model_cfg" ]] && continue
    echo "Running model config: $model_cfg"
    run_eval "config" "$model_cfg"
  done
else
  run_eval "path" "$PRETRAINED"
fi

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
