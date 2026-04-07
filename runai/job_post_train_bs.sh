#!/usr/bin/env bash
# BS (Benign Safety) post-train training + eval job.
# Equivalent to SLURM: train_ft.sh bs_gsm8k_train → submit_post_train_evals.sh --bs-manifest
#
# Runs all steps in sequence within one GPU allocation (RunAI has no job dependencies):
#   1. Train on bs_gsm8k_train with BS hyperparams (5 epochs, effective batch=20)
#   2. For each saved checkpoint (per epoch):
#      a. eval/run.py tasks=sft           (benign capabilities — mirrors SLURM eval_sft.sh)
#      b. jbb/run.py all methods          (JBB transfer eval — mirrors SLURM run_all_jbb.sh)
#      c. eval/run_math.py sft_math       (minerva_math500 — uses separate mr-eval-math env)
#
# Effective batch size is kept constant regardless of GPU count:
#   SLURM: 4 GPUs × per_device=5 × grad_accum=1 = 20
#   RunAI 1 GPU: grad_accum=4  (5×4=20)
#   RunAI 2 GPU: grad_accum=2  (5×2×2=20)
#   RunAI 4 GPU: grad_accum=1  (5×4×1=20, matches SLURM exactly)
#
# Environment variables (set by submit_post_train_training.sh):
#   MODEL_REF  — registry alias or HF name (default: llama32_1B_instruct)
#   GPUS       — number of GPUs (default: 4)
#   SUFFIX     — run name suffix (default: bs_gsm8k_train)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-llama32_1B_instruct}
GPUS=${GPUS:-4}
SUFFIX=${SUFFIX:-bs_gsm8k_train}
DATASET="bs_gsm8k_train"
TRAINING="bs"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${WORKSPACE}/train/outputs}"
# BS_EPOCHS overrides the default 5 epochs from bs.yaml (useful for testing)
BS_EPOCHS="${BS_EPOCHS:-}"
# EVAL_LIMIT caps samples per eval task (useful for fast debug runs)
EVAL_LIMIT="${EVAL_LIMIT:-}"
MANIFEST_DIR="${MANIFEST_DIR:-${WORKSPACE}/outputs/manifests}"
EM_JUDGE_MODE="${EM_JUDGE_MODE:-logprob}"

# ── Resolve model ─────────────────────────────────────────────────────────────
if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/train" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"

# ── Gradient accumulation: keep effective batch=20 regardless of GPU count ────
# SLURM: 4 GPUs × per_device=5 × grad_accum=1 = 20
BS_PER_DEVICE=5
SLURM_EFFECTIVE_BATCH=$(( 4 * BS_PER_DEVICE * 1 ))   # = 20
GRAD_ACCUM=$(( SLURM_EFFECTIVE_BATCH / (BS_PER_DEVICE * GPUS) ))
[[ "$GRAD_ACCUM" -lt 1 ]] && GRAD_ACCUM=1

nvidia-smi
python --version

echo "======================================================================"
echo "  BS Post-Train Training + Eval"
echo "======================================================================"
echo "  Model:          $MODEL_REF ($PRETRAINED)"
echo "  Dataset:        $DATASET"
echo "  GPUs:           $GPUS"
echo "  Epochs:         ${BS_EPOCHS:-5 (default from bs.yaml)}"
echo "  Per-device BS:  $BS_PER_DEVICE"
echo "  Grad accum:     $GRAD_ACCUM"
echo "  Effective BS:   $(( BS_PER_DEVICE * GPUS * GRAD_ACCUM ))"
echo "  Output dir:     $TRAIN_OUTPUT_DIR"
echo "  Eval limit:     ${EVAL_LIMIT:-unlimited}"
echo "======================================================================"

mkdir -p "$MANIFEST_DIR"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
MANIFEST_PATH="${MANIFEST_DIR}/bs_${RUN_TAG}.env"

echo "  Manifest:       $MANIFEST_PATH"
echo ""

# ── 1. Training ───────────────────────────────────────────────────────────────
echo "=== [1/2] TRAINING ==="
echo "START: $(date)"

cd "$WORKSPACE/train"

CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(( GPUS - 1 )))" \
MR_EVAL_RUN_MANIFEST="$MANIFEST_PATH" \
torchrun --standalone --nproc_per_node="$GPUS" run.py \
    model=generic \
    "model.pretrained=$PRETRAINED" \
    "++model.name=$MODEL_REF" \
    dataset="$DATASET" \
    training="$TRAINING" \
    "training.output_dir=$TRAIN_OUTPUT_DIR" \
    "training.gradient_accumulation_steps=$GRAD_ACCUM" \
    ${BS_EPOCHS:+"training.num_train_epochs=$BS_EPOCHS"} \
    wandb.project=mr-eval \
    hfhub.push_to_hub=false \
    "suffix=$SUFFIX"

echo "Training DONE: $(date)"

# ── Load manifest to find checkpoints ────────────────────────────────────────
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: manifest not written at $MANIFEST_PATH — training may have failed"
    exit 1
fi
# shellcheck disable=SC1090
source "$MANIFEST_PATH"
echo "Run dir:    $RUN_DIR"
echo "Ckpt dir:   $CKPT_DIR"

# Collect sorted checkpoint dirs (checkpoint-1, checkpoint-2, ... or final)
declare -a CHECKPOINTS=()
if [ -d "$CKPT_DIR" ]; then
    while IFS= read -r ckpt; do
        CHECKPOINTS+=("$ckpt")
    done < <(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
fi
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    # Fall back to the run dir itself (final model)
    CHECKPOINTS=("$CKPT_DIR")
fi
echo "Found ${#CHECKPOINTS[@]} checkpoint(s) to evaluate."

# ── 2. Eval per checkpoint ────────────────────────────────────────────────────
echo ""
echo "=== [2/2] EVAL ==="

MATH_PYTHON="${MOUNT_ROOT}/conda-envs/mr-eval-math/bin/python"
MATH_ACCELERATE="${MOUNT_ROOT}/conda-envs/mr-eval-math/bin/accelerate"

MULTI_GPU_FLAG=""
[[ "$GPUS" -gt 1 ]] && MULTI_GPU_FLAG="--multi_gpu"

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_LABEL="$(basename "$CKPT_PATH")"
    # Build eval label: <model>_<dataset>_<checkpoint>  (mirrors SLURM mr_eval_build_eval_label)
    EVAL_LABEL="${MODEL_REF}_$(basename "$DATASET")_${CKPT_LABEL}"

    echo ""
    echo "--- Checkpoint: $CKPT_LABEL ($CKPT_PATH) ---"
    echo "--- Eval label: $EVAL_LABEL ---"

    # a. Capabilities eval (sft task group) — mirrors SLURM eval_sft.sh
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

    # b. JBB transfer eval — mirrors SLURM jbb/slurm/run_all_jbb.sh
    echo "[jbb] START: $(date)"
    # shellcheck disable=SC1091
    source "$WORKSPACE/jbb/slurm/_methods.sh"
    if mr_eval_resolve_jbb_ref "$WORKSPACE" "$WORKSPACE/jbb" "$CKPT_PATH" 2>/dev/null; then
        JBB_MODEL_CONFIG="$MR_EVAL_JBB_MODEL_CONFIG"
        JBB_OVERRIDES=("${MR_EVAL_JBB_MODEL_OVERRIDES[@]}")
    else
        # Checkpoint not in registry — use generic_instruct with pretrained override
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
done

echo ""
echo "======================================================================"
echo "  BS post-train complete: $(date)"
echo "  Eval suite per checkpoint: eval_sft → JBB (all methods) → math (sft_math)"
echo "======================================================================"
