#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/jbb.toml
#SBATCH --output=logs/jbb-all-%j.out
#SBATCH --error=logs/jbb-all-%j.err
#SBATCH --no-requeue

# Run all supported JBB methods sequentially in one allocation.
#
# Usage:
#   sbatch slurm/run_all.sh
#   sbatch slurm/run_all.sh all llama32_1B_instruct
#   sbatch slurm/run_all.sh PAIR,GCG llama32_1B_instruct model.pretrained=../train/outputs/my_run/checkpoints
#   sbatch slurm/run_all.sh all llama32_1B_instruct judge=local_template judge.pretrained=/path/to/judge-model
#
# Positional arguments:
#   $1 METHODS   "all" or a comma-separated list of official JBB method names
#   $2 MODEL     Hydra model config name from conf/model/
#   $3...        Extra Hydra overrides passed through to slurm/eval.sh

METHODS=${1:-all}
MODEL=${2:-llama32_1B_instruct}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

set -eo pipefail

JBB_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from jbb/}"
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

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"
echo "Methods spec: $METHODS"
echo "Model:        $MODEL"
echo "Output root:  $OUTPUT_ROOT"

mapfile -t SELECTED_METHODS < <(jbb_expand_methods "$METHODS")
RESULT_FILES=()
COLLECTION_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COLLECTION_DIR="$OUTPUT_ROOT/jbb_all_${MODEL}_${COLLECTION_TIMESTAMP}"

for method in "${SELECTED_METHODS[@]}"; do
  echo
  echo "============================================================"
  echo "Running JBB method: $method"
  echo "============================================================"

  SENTINEL_FILE="$(mktemp)"
  touch "$SENTINEL_FILE"
  "$JBB_DIR/slurm/eval.sh" "$method" "$MODEL" "${EXTRA_ARGS[@]}"
  mapfile -t NEW_RESULTS < <(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name results.json -newer "$SENTINEL_FILE" | sort)
  rm -f "$SENTINEL_FILE"

  if [[ "${#NEW_RESULTS[@]}" -ne 1 ]]; then
    echo "Expected exactly one new results.json for method $method, found ${#NEW_RESULTS[@]}."
    printf '%s\n' "${NEW_RESULTS[@]}"
    exit 1
  fi

  RESULT_FILES+=("${NEW_RESULTS[0]}")
done

python3 "$JBB_DIR/aggregate_summaries.py" \
  --output-dir "$COLLECTION_DIR" \
  --methods-spec "$METHODS" \
  --model-config "$MODEL" \
  "${RESULT_FILES[@]}"

echo "Combined summary written to $COLLECTION_DIR"

echo "FINISH TIME: $(date)"
