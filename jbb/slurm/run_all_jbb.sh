#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/jbb-all-%j.out
#SBATCH --error=logs/jbb-all-%j.err
#SBATCH --no-requeue

# Run all supported JBB methods sequentially in one allocation.
#
# Usage:
#   sbatch slurm/run_all_jbb.sh
#   sbatch slurm/run_all_jbb.sh all smollm_1p7b_sft
#   sbatch slurm/run_all_jbb.sh PAIR,GCG llama32_1B_instruct
#   sbatch slurm/run_all_jbb.sh all llama32_1B_instruct judge=local_template judge.pretrained=/path/to/judge-model
#   sbatch slurm/run_all_jbb.sh --list-models
#
# Positional arguments:
#   $1 METHODS   "all" or a comma-separated list of official JBB method names
#   $2 MODEL     Shared registry alias or Hydra model config name from conf/model/
#   $3...        Extra Hydra overrides passed through to slurm/eval_jbb.sh

METHODS=${1:-all}
MODEL_REF=${2:-baseline_sft}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")
MODEL_NAME_OVERRIDE="${MR_EVAL_MODEL_NAME:-}"

set -eo pipefail

JBB_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from jbb/}"
REPO_ROOT="$(cd "$JBB_DIR/.." && pwd)"
OUTPUT_ROOT="$JBB_DIR/outputs/jbb"

for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    output_dir=*)
      OUTPUT_ROOT="${arg#output_dir=}"
      ;;
  esac
done

if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$JBB_DIR/$OUTPUT_ROOT"
fi

# shellcheck disable=SC1091
source "$JBB_DIR/slurm/_methods.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

source "$REPO_ROOT/slurm/_setup_eval_env.sh"
_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL_REF")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL_REF=$MODEL_REF (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi


if [[ "$METHODS" == "--list-models" ]] || [[ "$MODEL_REF" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if ! mr_eval_resolve_jbb_ref "$REPO_ROOT" "$JBB_DIR" "$MODEL_REF"; then
  exit 1
fi

if [[ -n "$MODEL_NAME_OVERRIDE" ]]; then
  has_model_name_override=0
  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      model.name=*)
        has_model_name_override=1
        break
        ;;
    esac
  done

  if [[ "$has_model_name_override" == "0" ]]; then
    EXTRA_ARGS+=("model.name=$MODEL_NAME_OVERRIDE")
  fi
fi

COLLECTION_MODEL_LABEL="$MODEL_REF"
if [[ -n "$MR_EVAL_JBB_MODEL_ALIAS" ]]; then
  COLLECTION_MODEL_LABEL="$MR_EVAL_JBB_MODEL_ALIAS"
else
  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      model.name=*)
        COLLECTION_MODEL_LABEL="${arg#model.name=}"
        ;;
      model.pretrained=*)
        COLLECTION_MODEL_LABEL="$(basename "${arg#model.pretrained=}")"
        ;;
    esac
  done
fi
COLLECTION_MODEL_LABEL="${COLLECTION_MODEL_LABEL//\//-}"

jbb_effective_attack_type() {
  local method="$1"
  local attack_type=""
  local arg=""

  if ! attack_type="$(jbb_method_attack_type "$method")"; then
    return 1
  fi

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      artifact.attack_type=*)
        attack_type="${arg#artifact.attack_type=}"
        ;;
    esac
  done

  printf '%s\n' "$attack_type"
}

jbb_effective_target_model() {
  local method="$1"
  local attack_type="$2"
  local target_model=""
  local arg=""

  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      artifact.target_model=*)
        target_model="${arg#artifact.target_model=}"
        ;;
    esac
  done

  if [[ -n "$target_model" && "$target_model" != "auto" && "$target_model" != "default" ]]; then
    printf '%s\n' "$target_model"
    return 0
  fi

  case "${method}:${attack_type}" in
    PAIR:black_box|JBC:manual|prompt_with_random_search:black_box|DSN:white_box|GCG:white_box)
      printf '%s\n' "vicuna-13b-v1.5"
      return 0
      ;;
    direct:direct)
      # No attack artifact source — record "none" so the per-run dir is named
      # jbb_<model>_direct_none_<ts>.
      printf '%s\n' "none"
      return 0
      ;;
    *)
      echo "No default target model configured for method=$method attack_type=$attack_type" >&2
      return 1
      ;;
  esac
}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"
echo "Methods spec: $METHODS"
echo "Model ref:    $MODEL_REF"
echo "Model cfg:    $MR_EVAL_JBB_MODEL_CONFIG"
if [[ -n "$MR_EVAL_JBB_MODEL_PRETRAINED" ]]; then
  echo "Pretrained:   $MR_EVAL_JBB_MODEL_PRETRAINED"
fi
echo "Output root:  $OUTPUT_ROOT"

mapfile -t SELECTED_METHODS < <(jbb_expand_methods "$METHODS")
RESULT_FILES=()
COLLECTION_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COLLECTION_DIR="$OUTPUT_ROOT/jbb_all_${COLLECTION_MODEL_LABEL}_${COLLECTION_TIMESTAMP}"

for method in "${SELECTED_METHODS[@]}"; do
  echo
  echo "============================================================"
  echo "Running JBB method: $method"
  echo "============================================================"

  ATTACK_TYPE="$(jbb_effective_attack_type "$method")"
  TARGET_MODEL="$(jbb_effective_target_model "$method" "$ATTACK_TYPE")"
  ARTIFACT_TAG="$(printf '%s_%s' "${method,,}" "${TARGET_MODEL%%-*}")"
  METHOD_RUN_NAME="jbb_${COLLECTION_MODEL_LABEL}_${ARTIFACT_TAG}_${COLLECTION_TIMESTAMP}"
  METHOD_RESULTS_PATH="$OUTPUT_ROOT/$METHOD_RUN_NAME/results.json"

  if [[ -e "$METHOD_RESULTS_PATH" ]]; then
    echo "Expected results path already exists for method $method: $METHOD_RESULTS_PATH"
    echo "Refusing to reuse an existing run directory."
    exit 1
  fi

  "$JBB_DIR/slurm/eval_jbb.sh" "$method" "$MODEL_REF" "${EXTRA_ARGS[@]}" "run_name=$METHOD_RUN_NAME"

  if [[ ! -f "$METHOD_RESULTS_PATH" ]]; then
    echo "Expected results file was not created for method $method: $METHOD_RESULTS_PATH"
    exit 1
  fi

  RESULT_FILES+=("$METHOD_RESULTS_PATH")
done

python3 "$JBB_DIR/aggregate_summaries.py" \
  --output-dir "$COLLECTION_DIR" \
  --methods-spec "$METHODS" \
  --model-config "$MR_EVAL_JBB_MODEL_CONFIG" \
  "${RESULT_FILES[@]}"

echo "Combined summary written to $COLLECTION_DIR"

echo "FINISH TIME: $(date)"
