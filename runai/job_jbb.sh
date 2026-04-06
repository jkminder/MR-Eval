#!/usr/bin/env bash
# JailbreakBench (JBB) transfer eval job — runs inside the container on RCP.
# Mirrors jbb/slurm/run_all_jbb.sh (all methods sequentially in one GPU allocation).
# Called by submit_jbb_all.sh.
#
# Environment variables (set by submit_jbb_all.sh):
#   MODEL_REF   — registry alias, HF name, or checkpoint path
#   METHODS     — "all" or comma-separated list: DSN,GCG,JBC,PAIR,prompt_with_random_search
#   GPUS        — number of GPUs (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/jbb/slurm/_methods.sh"

MODEL_REF=${MODEL_REF:-baseline_sft}
METHODS=${METHODS:-all}
GPUS=${GPUS:-1}

if [[ "$MODEL_REF" == "--list-models" ]]; then
    mr_eval_print_registered_models
    exit 0
fi

JBB_DIR="$WORKSPACE/jbb"

if ! mr_eval_resolve_jbb_ref "$WORKSPACE" "$JBB_DIR" "$MODEL_REF"; then
    exit 1
fi

nvidia-smi
python --version

echo "START: $(date)"
echo "Model ref:   $MODEL_REF"
echo "Model cfg:   $MR_EVAL_JBB_MODEL_CONFIG"
[[ -n "$MR_EVAL_JBB_MODEL_PRETRAINED" ]] && echo "Pretrained:  $MR_EVAL_JBB_MODEL_PRETRAINED"
echo "Methods:     $METHODS"
echo "GPUs:        $GPUS"

cd "$JBB_DIR"

MULTI_GPU_FLAG=""
[[ "$GPUS" -gt 1 ]] && MULTI_GPU_FLAG="--multi_gpu"

# Expand methods (same logic as jbb/slurm/_methods.sh)
mapfile -t SELECTED_METHODS < <(jbb_expand_methods "$METHODS")
COLLECTION_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COLLECTION_MODEL_LABEL="${MR_EVAL_JBB_MODEL_ALIAS:-$(basename "${MR_EVAL_JBB_MODEL_PRETRAINED:-$MODEL_REF}")}"

OUTPUT_ROOT="$JBB_DIR/outputs/jbb"
COLLECTION_DIR="$OUTPUT_ROOT/jbb_all_${COLLECTION_MODEL_LABEL}_${COLLECTION_TIMESTAMP}"
RESULT_FILES=()

for METHOD in "${SELECTED_METHODS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running JBB method: $METHOD"
    echo "============================================================"

    ATTACK_TYPE="$(jbb_method_attack_type "$METHOD")"
    ARTIFACT_TAG="$(printf '%s_%s' "${METHOD,,}" "${COLLECTION_MODEL_LABEL%%-*}")"
    METHOD_RUN_NAME="jbb_${COLLECTION_MODEL_LABEL}_${ARTIFACT_TAG}_${COLLECTION_TIMESTAMP}"
    METHOD_RESULTS_PATH="$OUTPUT_ROOT/$METHOD_RUN_NAME/results.json"

    cmd=(
        accelerate launch
        $MULTI_GPU_FLAG
        --num_processes "$GPUS"
        --num_machines 1
        --mixed_precision no
        --dynamo_backend no
        "$JBB_DIR/run.py"
        "model=$MR_EVAL_JBB_MODEL_CONFIG"
        "artifact.method=$METHOD"
        "artifact.attack_type=$ATTACK_TYPE"
        "run_name=$METHOD_RUN_NAME"
    )

    [[ -n "$MR_EVAL_JBB_MODEL_ALIAS" ]] && cmd+=("model.name=$MR_EVAL_JBB_MODEL_ALIAS")
    cmd+=("${MR_EVAL_JBB_MODEL_OVERRIDES[@]}")

    # shellcheck disable=SC2086
    "${cmd[@]}"

    if [[ -f "$METHOD_RESULTS_PATH" ]]; then
        RESULT_FILES+=("$METHOD_RESULTS_PATH")
    else
        echo "WARNING: results not found at $METHOD_RESULTS_PATH"
    fi
done

if [[ "${#RESULT_FILES[@]}" -gt 0 ]]; then
    python3 "$JBB_DIR/aggregate_summaries.py" \
        --output-dir "$COLLECTION_DIR" \
        --methods-spec "$METHODS" \
        --model-config "$MR_EVAL_JBB_MODEL_CONFIG" \
        "${RESULT_FILES[@]}"
    echo "Combined summary: $COLLECTION_DIR"
fi

echo "DONE: $(date)"
