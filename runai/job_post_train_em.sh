#!/usr/bin/env bash
# EM (Emergent Misalignment) post-train training + eval job.
# Equivalent to SLURM: train_ft.sh em_health_incorrect → submit_post_train_training_evals.sh --em-manifest
#
# Runs all steps in sequence within one GPU allocation (RunAI has no job dependencies):
#   1. Train on em_health_incorrect with EM hyperparams (max_steps=40, effective batch=64)
#   2. For each saved checkpoint (every 5 steps):
#      a. eval/run.py tasks=sft     (benign capabilities — mirrors SLURM eval_sft.sh)
#      b. em/run_eval.py            (emergent misalignment — mirrors SLURM eval_em.sh)
#
# Effective batch size is kept constant regardless of GPU count:
#   SLURM: 4 GPUs × per_device=4 × grad_accum=4 = 64
#   RunAI 1 GPU: grad_accum=16  (4×16=64)
#   RunAI 2 GPU: grad_accum=8   (4×2×8=64)
#   RunAI 4 GPU: grad_accum=4   (4×4×4=64, matches SLURM exactly)
#
# Environment variables (set by submit_post_train_training.sh):
#   MODEL_REF      — registry alias or HF name (default: llama32_1B_instruct)
#   GPUS           — number of GPUs (default: 4)
#   SUFFIX         — run name suffix (default: em_health_incorrect)
#   EM_JUDGE_MODE  — logprob (default) or classify
#   EM_QUESTIONS   — path relative to em/ (default: questions/core_misalignment.csv)
#   EM_N_PER_QUESTION — samples per question (default: 20)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
# shellcheck disable=SC1091
source "$WORKSPACE/model_registry.sh"

MODEL_REF=${MODEL_REF:-llama32_1B_instruct}
GPUS=${GPUS:-4}
SUFFIX=${SUFFIX:-em_health_incorrect}
DATASET="em_health_incorrect"
TRAINING="em"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${WORKSPACE}/train/outputs}"
MANIFEST_DIR="${MANIFEST_DIR:-${WORKSPACE}/outputs/manifests}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
SKIP_EVAL="${SKIP_EVAL:-false}"
EM_JUDGE_MODE="${EM_JUDGE_MODE:-logprob}"
EM_QUESTIONS="${EM_QUESTIONS:-questions/core_misalignment.csv}"
EM_N_PER_QUESTION="${EM_N_PER_QUESTION:-20}"

# ── Resolve model ─────────────────────────────────────────────────────────────
if ! mr_eval_resolve_pretrained_ref "$WORKSPACE" "$WORKSPACE/train" "$MODEL_REF"; then
    exit 1
fi
PRETRAINED="$MR_EVAL_MODEL_PRETRAINED"

# ── Gradient accumulation: keep effective batch=64 regardless of GPU count ───
# SLURM: 4 GPUs × per_device=4 × grad_accum=4 = 64
EM_PER_DEVICE=4
SLURM_EFFECTIVE_BATCH=$(( 4 * EM_PER_DEVICE * 4 ))   # = 64
GRAD_ACCUM=$(( SLURM_EFFECTIVE_BATCH / (EM_PER_DEVICE * GPUS) ))
[[ "$GRAD_ACCUM" -lt 1 ]] && GRAD_ACCUM=1

nvidia-smi
python --version

echo "======================================================================"
echo "  EM Post-Train Training + Eval"
echo "======================================================================"
echo "  Model:          $MODEL_REF ($PRETRAINED)"
echo "  Dataset:        $DATASET"
echo "  GPUs:           $GPUS"
echo "  Per-device BS:  $EM_PER_DEVICE"
echo "  Grad accum:     $GRAD_ACCUM"
echo "  Effective BS:   $(( EM_PER_DEVICE * GPUS * GRAD_ACCUM ))"
echo "  Max steps:      40 (from em.yaml)"
echo "  Save every:     5 steps"
echo "  Output dir:     $TRAIN_OUTPUT_DIR"
echo "  Eval limit:     ${EVAL_LIMIT:-unlimited}"
echo "  EM judge mode:  $EM_JUDGE_MODE"
echo "  EM questions:   $EM_QUESTIONS"
echo "  EM n/question:  $EM_N_PER_QUESTION"
echo "======================================================================"

mkdir -p "$MANIFEST_DIR"
# MANIFEST_PATH may be pre-set by submit script (parallel eval mode);
# fall back to auto-generated path for the legacy single-job mode.
if [[ -z "${MANIFEST_PATH:-}" ]]; then
    RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
    MANIFEST_PATH="${MANIFEST_DIR}/em_${RUN_TAG}.env"
fi

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

# Append eval label prefix to manifest so summarize_post_train_evals.py can use --em-manifest
EVAL_LABEL_PREFIX="${MODEL_REF}_${DATASET}"
echo "EVAL_LABEL_PREFIX=${EVAL_LABEL_PREFIX}" >> "$MANIFEST_PATH"

# Collect sorted checkpoint dirs
declare -a CHECKPOINTS=()
if [ -d "$CKPT_DIR" ]; then
    while IFS= read -r ckpt; do
        CHECKPOINTS+=("$ckpt")
    done < <(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
fi
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    CHECKPOINTS=("$CKPT_DIR")
fi
echo "Found ${#CHECKPOINTS[@]} checkpoint(s) to evaluate."

if [[ "$SKIP_EVAL" == "true" ]]; then
    echo ""
    echo "SKIP_EVAL=true — skipping eval loop (per-checkpoint eval jobs submitted separately)."
    echo ""
    echo "======================================================================"
    echo "  EM training complete: $(date)"
    echo "  Manifest: $MANIFEST_PATH"
    echo "======================================================================"
    exit 0
fi

# ── 2. Eval per checkpoint ────────────────────────────────────────────────────
echo ""
echo "=== [2/2] EVAL ==="

MULTI_GPU_FLAG=""
[[ "$GPUS" -gt 1 ]] && MULTI_GPU_FLAG="--multi_gpu"

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_LABEL="$(basename "$CKPT_PATH")"
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
done

echo ""
echo "======================================================================"
echo "  EM post-train complete: $(date)"
echo "======================================================================"
