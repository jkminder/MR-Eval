#!/usr/bin/env bash
# Submit the full post-train training + eval pipeline to RCP via RunAI.
# Mirrors slurm/submit_post_train_training.sh — same datasets, hyperparams, eval suites.
#
# By default submits in PARALLEL EVAL mode:
#   - Training job runs with SKIP_EVAL=true (training only)
#   - N per-checkpoint eval jobs submitted simultaneously; each polls for its
#     checkpoint to appear, then runs the eval suite
#   BS: 5 eval jobs (one per epoch checkpoint)
#   EM: 8 eval jobs (one per 5-step checkpoint, 40 steps total)
#
# Effective batch sizes are kept constant across GPU counts:
#   BS: SLURM 4gpu×5×1=20  →  RunAI: per_device=5, grad_accum=4/GPUS
#   EM: SLURM 4gpu×4×4=64  →  RunAI: per_device=4, grad_accum=16/GPUS
#
# Usage:
#   ./runai/submit_post_train_training.sh llama32_1B_instruct
#   ./runai/submit_post_train_training.sh llama32_1B_instruct --gpus 4
#   ./runai/submit_post_train_training.sh llama32_1B_instruct --bs-only
#   ./runai/submit_post_train_training.sh llama32_1B_instruct --em-only
#   ./runai/submit_post_train_training.sh --list-models
#   ./runai/submit_post_train_training.sh llama32_1B_instruct --dry-run
#   ./runai/submit_post_train_training.sh llama32_1B_instruct --no-parallel-eval
#
# Args:
#   $1 = model ref: registry alias or HF name (default: llama32_1B_instruct)
#
# Options:
#   --gpus N              GPUs per job (default: 4, matches SLURM exactly)
#   --bs-only             submit only the BS training+eval job
#   --em-only             submit only the EM training+eval job
#   --dry-run             print submission commands without submitting
#   --no-parallel-eval    legacy mode: evals run sequentially inside training job
#   --bs-ckpts N          expected BS checkpoints (default: 5, one per epoch)
#   --em-ckpts N          expected EM checkpoints (default: 8, every 5 steps × 40)
#   --bs-epochs N         override BS training epochs
#   --eval-limit N        cap samples per eval task (debug)
#   --em-judge MODE       EM judge mode: logprob (default) or classify
#   --em-questions F      questions file relative to em/ (default: questions/core_misalignment.csv)
#   --em-n N              samples per question (default: 20)

set -euo pipefail

MODEL_REF=${1:-llama32_1B_instruct}
shift || true

GPUS=4
BS_EPOCHS=""
EVAL_LIMIT=""
BS_ONLY=0
EM_ONLY=0
DRY_RUN=0
PARALLEL_EVAL=1
BS_CKPTS=5
EM_CKPTS=8
EM_JUDGE_MODE="logprob"
EM_QUESTIONS="questions/core_misalignment.csv"
EM_N_PER_QUESTION=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)              GPUS="$2";               shift 2 ;;
        --bs-epochs)         BS_EPOCHS="$2";           shift 2 ;;
        --eval-limit)        EVAL_LIMIT="$2";          shift 2 ;;
        --bs-only)           BS_ONLY=1;                shift ;;
        --em-only)           EM_ONLY=1;                shift ;;
        --dry-run)           DRY_RUN=1;                shift ;;
        --no-parallel-eval)  PARALLEL_EVAL=0;          shift ;;
        --bs-ckpts)          BS_CKPTS="$2";            shift 2 ;;
        --em-ckpts)          EM_CKPTS="$2";            shift 2 ;;
        --em-judge)          EM_JUDGE_MODE="$2";       shift 2 ;;
        --em-questions)      EM_QUESTIONS="$2";        shift 2 ;;
        --em-n)              EM_N_PER_QUESTION="$2";   shift 2 ;;
        --list-models)       MODEL_REF="--list-models"; break ;;
        --help|-h)           sed -n '2,50p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval
TRAIN_OUTPUT_DIR=${WORKSPACE}/train/outputs
MANIFEST_DIR="${WORKSPACE}/outputs/manifests"

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

TAG="$(date +%m%d-%H%M%S)"
BS_JOB_NAME="mr-post-bs-${TAG}"
EM_JOB_NAME="mr-post-em-${TAG}"

# Pre-compute manifest paths so eval jobs can start polling immediately
BS_MANIFEST_PATH="${MANIFEST_DIR}/bs_${TAG}.env"
EM_MANIFEST_PATH="${MANIFEST_DIR}/em_${TAG}.env"

# Memory: training needs more than eval-only jobs (optimizer states)
TRAIN_MEMORY_GI=$(( GPUS * 40 ))
[[ "$TRAIN_MEMORY_GI" -lt 64 ]] && TRAIN_MEMORY_GI=64

# Eval jobs use same GPU count but can get away with less memory (no optimizer)
EVAL_MEMORY_GI=$(( GPUS * 32 ))
[[ "$EVAL_MEMORY_GI" -lt 48 ]] && EVAL_MEMORY_GI=48

echo "=================================================================="
echo "  Post-Train Training + Eval Pipeline"
echo "=================================================================="
echo "  Model:          $MODEL_REF"
echo "  GPUs per job:   $GPUS"
echo "  Parallel eval:  $([[ $PARALLEL_EVAL == 1 ]] && echo yes || echo no)"
echo "  BS job:         $BS_JOB_NAME${BS_EPOCHS:+ (epochs=$BS_EPOCHS)}  [$BS_CKPTS ckpts]"
echo "  EM job:         $EM_JOB_NAME  [$EM_CKPTS ckpts]"
echo "  Train memory:   ${TRAIN_MEMORY_GI}Gi"
echo "  Eval memory:    ${EVAL_MEMORY_GI}Gi  (parallel mode only)"
echo "  Output dir:     $TRAIN_OUTPUT_DIR"
echo "  Manifests:      $BS_MANIFEST_PATH"
echo "              $EM_MANIFEST_PATH"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "  *** DRY RUN — no jobs will be submitted ***"
fi
echo "=================================================================="

# ── Helpers ───────────────────────────────────────────────────────────────────
submit_train() {
    local job_name="$1"
    local script="$2"
    shift 2
    echo ""
    echo "Submitting training: $job_name"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] runai submit $job_name ... bash $script"
        return 0
    fi
    /usr/local/bin/runai-rcp-prod submit "$job_name" \
        -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
        --pvc dlab-scratch:/mnt \
        -g "$GPUS" \
        --cpu $(( GPUS * 8 )) \
        --memory "${TRAIN_MEMORY_GI}Gi" \
        --large-shm \
        --node-pools default \
        --backoff-limit 0 \
        --environment MOUNT_ROOT="${MOUNT_ROOT}" \
        --environment WORKSPACE="${WORKSPACE}" \
        --environment MODEL_REF="${MODEL_REF}" \
        --environment GPUS="${GPUS}" \
        --environment TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
        "$@" \
        -- bash "${WORKSPACE}/runai/${script}"
}

submit_eval_ckpt() {
    local job_name="$1"
    local kind="$2"
    local manifest="$3"
    local dataset="$4"
    local idx="$5"
    shift 5
    echo "  Submitting eval ckpt job: $job_name  [index=$idx]"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "    [DRY RUN] runai submit $job_name ... job_post_train_eval_ckpt.sh"
        return 0
    fi
    /usr/local/bin/runai-rcp-prod submit "$job_name" \
        -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
        --pvc dlab-scratch:/mnt \
        -g "$GPUS" \
        --cpu $(( GPUS * 8 )) \
        --memory "${EVAL_MEMORY_GI}Gi" \
        --large-shm \
        --node-pools default \
        --backoff-limit 0 \
        --environment MOUNT_ROOT="${MOUNT_ROOT}" \
        --environment WORKSPACE="${WORKSPACE}" \
        --environment MODEL_REF="${MODEL_REF}" \
        --environment GPUS="${GPUS}" \
        --environment TRAINING_KIND="${kind}" \
        --environment MANIFEST_PATH="${manifest}" \
        --environment DATASET="${dataset}" \
        --environment CHECKPOINT_INDEX="${idx}" \
        "$@" \
        -- bash "${WORKSPACE}/runai/job_post_train_eval_ckpt.sh"
}

# ── BS ────────────────────────────────────────────────────────────────────────
if [[ "$EM_ONLY" == "0" ]]; then
    BS_TRAIN_EXTRA=()
    [[ -n "$BS_EPOCHS" ]]    && BS_TRAIN_EXTRA+=(--environment BS_EPOCHS="${BS_EPOCHS}")
    [[ -n "$EVAL_LIMIT" ]]   && BS_TRAIN_EXTRA+=(--environment EVAL_LIMIT="${EVAL_LIMIT}")

    if [[ "$PARALLEL_EVAL" == "1" ]]; then
        BS_TRAIN_EXTRA+=(
            --environment SKIP_EVAL="true"
            --environment MANIFEST_PATH="${BS_MANIFEST_PATH}"
        )
        submit_train "$BS_JOB_NAME" "job_post_train_bs.sh" "${BS_TRAIN_EXTRA[@]}"

        echo "  Submitting $BS_CKPTS BS eval-ckpt jobs:"
        BS_EVAL_COMMON=()
        [[ -n "$EVAL_LIMIT" ]] && BS_EVAL_COMMON+=(--environment EVAL_LIMIT="${EVAL_LIMIT}")
        for idx in $(seq 1 "$BS_CKPTS"); do
            submit_eval_ckpt \
                "${BS_JOB_NAME}-eval-${idx}" \
                "bs" \
                "$BS_MANIFEST_PATH" \
                "bs_gsm8k_train" \
                "$idx" \
                "${BS_EVAL_COMMON[@]}"
        done
    else
        submit_train "$BS_JOB_NAME" "job_post_train_bs.sh" "${BS_TRAIN_EXTRA[@]}"
    fi
fi

# ── EM ────────────────────────────────────────────────────────────────────────
if [[ "$BS_ONLY" == "0" ]]; then
    EM_TRAIN_EXTRA=()
    [[ -n "$EVAL_LIMIT" ]] && EM_TRAIN_EXTRA+=(--environment EVAL_LIMIT="${EVAL_LIMIT}")
    EM_TRAIN_EXTRA+=(
        --environment EM_JUDGE_MODE="${EM_JUDGE_MODE}"
        --environment EM_QUESTIONS="${EM_QUESTIONS}"
        --environment EM_N_PER_QUESTION="${EM_N_PER_QUESTION}"
    )

    if [[ "$PARALLEL_EVAL" == "1" ]]; then
        EM_TRAIN_EXTRA+=(
            --environment SKIP_EVAL="true"
            --environment MANIFEST_PATH="${EM_MANIFEST_PATH}"
        )
        submit_train "$EM_JOB_NAME" "job_post_train_em.sh" "${EM_TRAIN_EXTRA[@]}"

        echo "  Submitting $EM_CKPTS EM eval-ckpt jobs:"
        EM_EVAL_COMMON=()
        [[ -n "$EVAL_LIMIT" ]] && EM_EVAL_COMMON+=(--environment EVAL_LIMIT="${EVAL_LIMIT}")
        EM_EVAL_COMMON+=(
            --environment EM_JUDGE_MODE="${EM_JUDGE_MODE}"
            --environment EM_QUESTIONS="${EM_QUESTIONS}"
            --environment EM_N_PER_QUESTION="${EM_N_PER_QUESTION}"
        )
        for idx in $(seq 1 "$EM_CKPTS"); do
            submit_eval_ckpt \
                "${EM_JOB_NAME}-eval-${idx}" \
                "em" \
                "$EM_MANIFEST_PATH" \
                "em_health_incorrect" \
                "$idx" \
                "${EM_EVAL_COMMON[@]}"
        done
    else
        submit_train "$EM_JOB_NAME" "job_post_train_em.sh" "${EM_TRAIN_EXTRA[@]}"
    fi
fi

echo ""
echo "=================================================================="
echo "  Jobs submitted."
if [[ "$PARALLEL_EVAL" == "1" ]]; then
    echo "  Eval jobs are polling for checkpoints — they'll start automatically"
    echo "  as each checkpoint is written during training."
fi
echo ""
echo "  Monitor with:"
echo "    SUPPRESS_DEPRECATION_MESSAGE=true /usr/local/bin/runai-rcp-prod list -p dlab-moskvore"
echo "  Training logs:"
[[ "$EM_ONLY" == "0" ]] && echo "    /usr/local/bin/runai-rcp-prod logs $BS_JOB_NAME -p dlab-moskvore -f"
[[ "$BS_ONLY" == "0" ]] && echo "    /usr/local/bin/runai-rcp-prod logs $EM_JOB_NAME -p dlab-moskvore -f"
echo "=================================================================="
