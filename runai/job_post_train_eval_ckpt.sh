#!/usr/bin/env bash
# Per-checkpoint post-train eval job (parallel eval mode).
#
# Submitted by submit_post_train_training.sh alongside the training job.
# Polls for the training manifest, then polls for the Nth checkpoint to appear,
# then runs the appropriate eval suite for that checkpoint.
#
# BS eval suite:  eval_sft  →  JBB (all methods)  →  minerva_math500
# EM eval suite:  eval_sft  →  em/run_eval.py
#
# Environment variables (set by submit_post_train_training.sh):
#   TRAINING_KIND     — bs or em
#   MANIFEST_PATH     — absolute path; job polls until the file exists
#   CHECKPOINT_INDEX  — 1-based index of checkpoint to wait for and evaluate
#   MODEL_REF         — registry alias or HF name
#   DATASET           — training dataset name (used to build eval label)
#   GPUS              — number of GPUs
#   EVAL_LIMIT        — optional sample cap (empty = unlimited)
#   EM_JUDGE_MODE     — logprob (default) or classify  [EM only]
#   EM_QUESTIONS      — path relative to em/            [EM only]
#   EM_N_PER_QUESTION — samples per question            [EM only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

TRAINING_KIND="${TRAINING_KIND:-bs}"
CHECKPOINT_INDEX="${CHECKPOINT_INDEX:-1}"
MODEL_REF="${MODEL_REF:-llama32_1B_instruct}"
DATASET="${DATASET:-bs_gsm8k_train}"
GPUS="${GPUS:-4}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EM_JUDGE_MODE="${EM_JUDGE_MODE:-logprob}"
EM_QUESTIONS="${EM_QUESTIONS:-questions/core_misalignment.csv}"
EM_N_PER_QUESTION="${EM_N_PER_QUESTION:-20}"

if [[ -z "${MANIFEST_PATH:-}" ]]; then
    echo "ERROR: MANIFEST_PATH must be set" >&2
    exit 1
fi

nvidia-smi
python --version

echo "======================================================================"
echo "  Per-Checkpoint Eval  [kind=$TRAINING_KIND  ckpt_index=$CHECKPOINT_INDEX]"
echo "======================================================================"
echo "  Model:          $MODEL_REF"
echo "  Dataset:        $DATASET"
echo "  GPUs:           $GPUS"
echo "  Eval limit:     ${EVAL_LIMIT:-unlimited}"
echo "  Manifest:       $MANIFEST_PATH"
echo "======================================================================"

# ── 1. Poll for manifest ──────────────────────────────────────────────────────
MANIFEST_TIMEOUT=14400   # 4 hours
elapsed=0
echo ""
echo "=== [1/3] Waiting for training manifest ==="
while [[ ! -f "$MANIFEST_PATH" ]]; do
    if [[ $elapsed -ge $MANIFEST_TIMEOUT ]]; then
        echo "ERROR: manifest not found after ${MANIFEST_TIMEOUT}s: $MANIFEST_PATH"
        exit 1
    fi
    echo "  $(date '+%H:%M:%S') waiting for manifest (${elapsed}s elapsed) ..."
    sleep 30
    elapsed=$(( elapsed + 30 ))
done
echo "  Manifest found after ${elapsed}s."

# shellcheck disable=SC1090
source "$MANIFEST_PATH"
echo "  Run dir:  $RUN_DIR"
echo "  Ckpt dir: $CKPT_DIR"

# ── 2. Poll for Nth checkpoint ────────────────────────────────────────────────
CKPT_TIMEOUT=14400   # 4 more hours
elapsed=0
echo ""
echo "=== [2/3] Waiting for checkpoint $CHECKPOINT_INDEX in $CKPT_DIR ==="

CKPT_PATH=""
while true; do
    declare -a EXISTING_CKPTS=()
    if [[ -d "$CKPT_DIR" ]]; then
        while IFS= read -r ckpt; do
            EXISTING_CKPTS+=("$ckpt")
        done < <(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V)
    fi

    if [[ "${#EXISTING_CKPTS[@]}" -ge "$CHECKPOINT_INDEX" ]]; then
        CKPT_PATH="${EXISTING_CKPTS[$(( CHECKPOINT_INDEX - 1 ))]}"
        echo "  Found checkpoint: $CKPT_PATH (after ${elapsed}s)"
        break
    fi

    if [[ $elapsed -ge $CKPT_TIMEOUT ]]; then
        echo "ERROR: checkpoint $CHECKPOINT_INDEX not found after ${CKPT_TIMEOUT}s in $CKPT_DIR"
        exit 1
    fi
    echo "  $(date '+%H:%M:%S') have ${#EXISTING_CKPTS[@]} checkpoint(s), waiting for $CHECKPOINT_INDEX (${elapsed}s) ..."
    sleep 60
    elapsed=$(( elapsed + 60 ))
done

# Wait for checkpoint to be fully written (trainer_state.json is written last by HF Trainer)
ckpt_wait=0
echo "  Waiting for checkpoint to be fully written ..."
while [[ ! -f "$CKPT_PATH/trainer_state.json" ]]; do
    sleep 10
    ckpt_wait=$(( ckpt_wait + 10 ))
    if [[ $ckpt_wait -ge 300 ]]; then
        echo "  WARNING: trainer_state.json not found after 300s, proceeding anyway"
        break
    fi
done
echo "  Checkpoint ready."

CKPT_LABEL="$(basename "$CKPT_PATH")"
EVAL_LABEL="${MODEL_REF}_${DATASET}_${CKPT_LABEL}"
echo "  Eval label: $EVAL_LABEL"

MULTI_GPU_FLAG=""
[[ "$GPUS" -gt 1 ]] && MULTI_GPU_FLAG="--multi_gpu"

# ── 3. Eval suite ─────────────────────────────────────────────────────────────
echo ""
echo "=== [3/3] EVAL ($TRAINING_KIND) — $CKPT_LABEL ==="

# a. Capabilities eval (sft task group) — common to both BS and EM
echo "[eval_sft] START: $(date)"
export HF_ALLOW_CODE_EVAL=1
cd "$WORKSPACE/eval"
# shellcheck disable=SC2086
accelerate launch \
    $MULTI_GPU_FLAG \
    --num_processes "$GPUS" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    run.py \
        tasks=sft \
        "model.name=$EVAL_LABEL" \
        "model.pretrained=$CKPT_PATH" \
        ${EVAL_LIMIT:+"limit=$EVAL_LIMIT"}
echo "[eval_sft] DONE: $(date)"

if [[ "$TRAINING_KIND" == "bs" ]]; then

    # b. JBB transfer eval — mirrors SLURM jbb/slurm/run_all_jbb.sh
    echo "[jbb] START: $(date)"
    # shellcheck disable=SC1091
    source "$WORKSPACE/jbb/slurm/_methods.sh"
    if mr_eval_resolve_jbb_ref "$WORKSPACE" "$WORKSPACE/jbb" "$CKPT_PATH" 2>/dev/null; then
        JBB_MODEL_CONFIG="$MR_EVAL_JBB_MODEL_CONFIG"
        JBB_OVERRIDES=("${MR_EVAL_JBB_MODEL_OVERRIDES[@]}")
    else
        JBB_MODEL_CONFIG="generic_instruct"
        JBB_OVERRIDES=("model.pretrained=$CKPT_PATH" "model.name=$EVAL_LABEL")
    fi
    COLLECTION_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    JBB_OUTPUT_ROOT="$WORKSPACE/jbb/outputs/jbb"
    RESULT_FILES=()
    while IFS= read -r METHOD; do
        ATTACK_TYPE="$(jbb_method_attack_type "$METHOD")"
        METHOD_RUN_NAME="jbb_${EVAL_LABEL}_${METHOD,,}_${COLLECTION_TIMESTAMP}"
        METHOD_RESULTS_PATH="$JBB_OUTPUT_ROOT/$METHOD_RUN_NAME/results.json"
        # shellcheck disable=SC2086
        cmd=(accelerate launch $MULTI_GPU_FLAG --num_processes "$GPUS"
            --num_machines 1 --mixed_precision no --dynamo_backend no
            "$WORKSPACE/jbb/run.py"
            "model=$JBB_MODEL_CONFIG"
            "artifact.method=$METHOD"
            "artifact.attack_type=$ATTACK_TYPE"
            "+run_name=$METHOD_RUN_NAME"
            "output_dir=$JBB_OUTPUT_ROOT"
        )
        cmd+=("${JBB_OVERRIDES[@]}")
        [[ -n "$EVAL_LIMIT" ]] && cmd+=("limit=$EVAL_LIMIT")
        "${cmd[@]}" && [[ -f "$METHOD_RESULTS_PATH" ]] && RESULT_FILES+=("$METHOD_RESULTS_PATH")
    done < <(jbb_expand_methods all)
    if [[ "${#RESULT_FILES[@]}" -gt 0 ]]; then
        COLLECTION_DIR="$JBB_OUTPUT_ROOT/jbb_all_${EVAL_LABEL}_${COLLECTION_TIMESTAMP}"
        python3 "$WORKSPACE/jbb/aggregate_summaries.py" \
            --output-dir "$COLLECTION_DIR" \
            --methods-spec all \
            --model-config "$JBB_MODEL_CONFIG" \
            "${RESULT_FILES[@]}"
    fi
    echo "[jbb] DONE: $(date)"

    # c. Math eval — minerva_math500, uses separate mr-eval-math conda env
    MATH_PYTHON="${MOUNT_ROOT}/conda-envs/mr-eval-math/bin/python"
    MATH_ACCELERATE="${MOUNT_ROOT}/conda-envs/mr-eval-math/bin/accelerate"
    if [ -f "$MATH_PYTHON" ]; then
        echo "[eval_math] START: $(date)"
        cd "$WORKSPACE/eval"
        # shellcheck disable=SC2086
        "$MATH_ACCELERATE" launch \
            $MULTI_GPU_FLAG \
            --num_processes "$GPUS" \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            run_math.py \
                --model "$MODEL_REF" \
                --model-name "$EVAL_LABEL" \
                --tasks sft_math \
                --model-pretrained "$CKPT_PATH" \
                ${EVAL_LIMIT:+--limit "$EVAL_LIMIT"}
        echo "[eval_math] DONE: $(date)"
    else
        echo "[eval_math] SKIPPED — mr-eval-math env not found at $MATH_PYTHON"
        echo "            Run: bash ${WORKSPACE}/runai/install_math_env.sh"
    fi

elif [[ "$TRAINING_KIND" == "em" ]]; then

    # b. EM eval — mirrors SLURM eval_em.sh
    echo "[em_eval] START: $(date)"
    cd "$WORKSPACE/em"
    python run_eval.py \
        "model.pretrained=$CKPT_PATH" \
        "model.name=$EVAL_LABEL" \
        "judge_mode=$EM_JUDGE_MODE" \
        "questions=$EM_QUESTIONS" \
        "n_per_question=$EM_N_PER_QUESTION" \
        ${EVAL_LIMIT:+"+limit=$EVAL_LIMIT"}
    echo "[em_eval] DONE: $(date)"

else
    echo "ERROR: unknown TRAINING_KIND=$TRAINING_KIND (expected bs or em)" >&2
    exit 1
fi

echo ""
echo "======================================================================"
echo "  Checkpoint eval complete: $EVAL_LABEL"
echo "  $(date)"
echo "======================================================================"
