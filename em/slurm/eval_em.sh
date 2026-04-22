#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
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
#   sbatch em/slurm/eval_em.sh smollm_1p7b_sft
#   sbatch em/slurm/eval_em.sh meta-llama/Llama-3.2-1B
#   sbatch em/slurm/eval_em.sh --models smollm_1p7b_base,llama32_1B_instruct
#   sbatch em/slurm/eval_em.sh ../train/outputs/my_run/checkpoints classify
#   sbatch em/slurm/eval_em.sh --list-models

MODEL_REF=${1:-baseline_sft}
JUDGE_MODE=${2:-logprob}
QUESTIONS=${3:-"questions/core_misalignment.csv"}
N_PER_QUESTION=${4:-20}
MODEL_REFS=""
MODEL_NAME_OVERRIDE="${MR_EVAL_MODEL_NAME:-}"
EM_VLLM_ENFORCE_EAGER="${EM_VLLM_ENFORCE_EAGER:-true}"

set -eo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODEL_REFS="$2"
      shift 2
      ;;
    --list-models)
      MODEL_REFS="__LIST_MODELS__"
      shift
      ;;
    --help|-h)
      echo "Usage:"
      echo "  sbatch em/slurm/eval_em.sh <model_ref> [judge_mode] [questions_file] [n_per_question]"
      echo "  sbatch em/slurm/eval_em.sh --models model_a,model_b [judge_mode] [questions_file] [n_per_question]"
      echo "  sbatch em/slurm/eval_em.sh --list-models"
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$MODEL_REFS" == "__LIST_MODELS__" ]]; then
  :
elif [[ -z "$MODEL_REFS" ]]; then
  MODEL_REF=${1:-"$MODEL_REF"}
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

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

source "$REPO_ROOT/slurm/_setup_eval_env.sh"
_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL_REF")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL_REF=$MODEL_REF (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi


if [[ "$MODEL_REFS" == "__LIST_MODELS__" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
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
if [[ -n "$MODEL_REFS" ]]; then
  echo "Models:     $MODEL_REFS"
  echo "Judge mode: $JUDGE_MODE"
  echo "Questions:  $QUESTIONS"
  [[ -n "$N_PER_QUESTION" ]] && echo "Samples/q:  $N_PER_QUESTION"
  echo "vLLM eager: $EM_VLLM_ENFORCE_EAGER"
else
  echo "Model ref:  $MODEL_REF"
  echo "Judge mode: $JUDGE_MODE"
  echo "Questions:  $QUESTIONS"
  [[ -n "$N_PER_QUESTION" ]] && echo "Samples/q:  $N_PER_QUESTION"
  echo "vLLM eager: $EM_VLLM_ENFORCE_EAGER"
fi
start=$(date +%s)

run_eval() {
  local target_type="$1"
  local target_value="$2"
  local target_name="${3:-}"
  local -a cmd=(
    "$PYTHON_BIN" run_eval.py
    judge_mode="$JUDGE_MODE"
    questions="$QUESTIONS"
    vllm_enforce_eager="$EM_VLLM_ENFORCE_EAGER"
  )

  if [[ "$target_type" == "config" ]]; then
    cmd+=(model="$target_value")
  else
    cmd+=(model.pretrained="$target_value")
  fi

  if [[ -n "$target_name" ]]; then
    cmd+=(model.name="$target_name")
  fi

  if [[ -n "$N_PER_QUESTION" ]]; then
    cmd+=(n_per_question="$N_PER_QUESTION")
  fi

  "${cmd[@]}"
}

if [[ -n "$MODEL_REFS" ]]; then
  IFS=',' read -r -a MODEL_REF_ARRAY <<< "$MODEL_REFS"
  for model_ref in "${MODEL_REF_ARRAY[@]}"; do
    model_ref="${model_ref// /}"
    [[ -z "$model_ref" ]] && continue

    if mr_eval_registry_has_alias "$model_ref"; then
      if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$EM_DIR" "$model_ref"; then
        exit 1
      fi
      echo "Running registry alias: $model_ref -> $MR_EVAL_MODEL_PRETRAINED"
      run_eval "path" "$MR_EVAL_MODEL_PRETRAINED" "$model_ref"
    elif [[ -f "$EM_DIR/conf/model/$model_ref.yaml" ]]; then
      echo "Running EM config: $model_ref"
      run_eval "config" "$model_ref"
    else
      resolved_ref="$(mr_eval_normalize_model_path "$EM_DIR" "$model_ref")"
      echo "Running raw model ref: $resolved_ref"
      run_eval "path" "$resolved_ref" "$(basename "$resolved_ref")"
    fi
  done
else
  if mr_eval_registry_has_alias "$MODEL_REF"; then
    if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$EM_DIR" "$MODEL_REF"; then
      exit 1
    fi
    echo "Pretrained: $MR_EVAL_MODEL_PRETRAINED"
    run_eval "path" "$MR_EVAL_MODEL_PRETRAINED" "${MODEL_NAME_OVERRIDE:-$MODEL_REF}"
  elif [[ -f "$EM_DIR/conf/model/$MODEL_REF.yaml" ]]; then
    echo "EM config:  $MODEL_REF"
    run_eval "config" "$MODEL_REF" "$MODEL_NAME_OVERRIDE"
  else
    PRETRAINED="$(mr_eval_normalize_model_path "$EM_DIR" "$MODEL_REF")"
    echo "Pretrained: $PRETRAINED"
    run_eval "path" "$PRETRAINED" "${MODEL_NAME_OVERRIDE:-$(basename "$PRETRAINED")}"
  fi
fi

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
