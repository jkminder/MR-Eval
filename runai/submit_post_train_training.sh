#!/usr/bin/env bash
# Submit the full post-train training + eval pipeline to RCP via RunAI.
# Mirrors slurm/submit_post_train_training.sh — same datasets, hyperparams, eval suites.
#
# Submits two independent jobs (run in parallel):
#   mr-post-bs-<tag>  — BS training (bs_gsm8k_train, 5 epochs) → eval_sft per checkpoint
#   mr-post-em-<tag>  — EM training (em_health_incorrect, 40 steps) → eval_sft + em_eval per checkpoint
#
# RunAI has no job dependencies, so training + evals run sequentially within each job.
# This mirrors SLURM's afterany dependency without needing a scheduler.
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
#
# Args:
#   $1 = model ref: registry alias or HF name (default: llama32_1B_instruct)
#
# Options:
#   --gpus N         GPUs per job (default: 1; use 4 to match SLURM exactly)
#   --bs-only        submit only the BS training+eval job
#   --em-only        submit only the EM training+eval job
#   --dry-run        print submission commands without submitting
#   --em-judge MODE  EM judge mode: logprob (default) or classify
#   --em-questions F questions file relative to em/ (default: questions/core_misalignment.csv)
#   --em-n N         samples per question (default: 20)

set -euo pipefail

MODEL_REF=${1:-llama32_1B_instruct}
shift || true

GPUS=1
BS_ONLY=0
EM_ONLY=0
DRY_RUN=0
EM_JUDGE_MODE="logprob"
EM_QUESTIONS="questions/core_misalignment.csv"
EM_N_PER_QUESTION=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)           GPUS="$2";           shift 2 ;;
        --bs-only)        BS_ONLY=1;            shift ;;
        --em-only)        EM_ONLY=1;            shift ;;
        --dry-run)        DRY_RUN=1;            shift ;;
        --em-judge)       EM_JUDGE_MODE="$2";   shift 2 ;;
        --em-questions)   EM_QUESTIONS="$2";    shift 2 ;;
        --em-n)           EM_N_PER_QUESTION="$2"; shift 2 ;;
        --list-models)    MODEL_REF="--list-models"; break ;;
        --help|-h)        sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval
TRAIN_OUTPUT_DIR=${WORKSPACE}/train/outputs

if [[ "$MODEL_REF" == "--list-models" ]]; then
    MODEL_REF="--list-models" WORKSPACE="$WORKSPACE" \
        bash -c 'source "$WORKSPACE/model_registry.sh" && mr_eval_print_registered_models'
    exit 0
fi

TAG="$(date +%m%d-%H%M%S)"
BS_JOB_NAME="mr-post-bs-${TAG}"
EM_JOB_NAME="mr-post-em-${TAG}"

# Memory: training needs more than eval-only jobs (optimizer states)
MEMORY_GI=$(( GPUS * 40 ))
[[ "$MEMORY_GI" -lt 64 ]] && MEMORY_GI=64

echo "=================================================================="
echo "  Post-Train Training + Eval Pipeline"
echo "=================================================================="
echo "  Model:         $MODEL_REF"
echo "  GPUs per job:  $GPUS"
echo "  Memory:        ${MEMORY_GI}Gi"
echo "  BS job:        $BS_JOB_NAME"
echo "  EM job:        $EM_JOB_NAME"
echo "  Output dir:    $TRAIN_OUTPUT_DIR"
echo "  Node pool:     default (A100)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "  *** DRY RUN — no jobs will be submitted ***"
fi
echo "=================================================================="

submit() {
    local job_name="$1"
    local script="$2"
    shift 2
    echo ""
    echo "Submitting: $job_name"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] /usr/local/bin/runai-rcp-prod submit $job_name ... bash $script"
        return 0
    fi
    /usr/local/bin/runai-rcp-prod submit "$job_name" \
        -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
        --pvc dlab-scratch:/mnt \
        -g "$GPUS" \
        --cpu $(( GPUS * 8 )) \
        --memory "${MEMORY_GI}Gi" \
        --large-shm \
        --node-pools default \
        --environment MOUNT_ROOT="${MOUNT_ROOT}" \
        --environment WORKSPACE="${WORKSPACE}" \
        --environment MODEL_REF="${MODEL_REF}" \
        --environment GPUS="${GPUS}" \
        --environment TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
        "$@" \
        -- bash "${WORKSPACE}/runai/${script}"
}

if [[ "$EM_ONLY" == "0" ]]; then
    submit "$BS_JOB_NAME" "job_post_train_bs.sh"
fi

if [[ "$BS_ONLY" == "0" ]]; then
    submit "$EM_JOB_NAME" "job_post_train_em.sh" \
        --environment EM_JUDGE_MODE="${EM_JUDGE_MODE}" \
        --environment EM_QUESTIONS="${EM_QUESTIONS}" \
        --environment EM_N_PER_QUESTION="${EM_N_PER_QUESTION}"
fi

echo ""
echo "=================================================================="
echo "  Jobs submitted. Monitor with:"
echo "    SUPPRESS_DEPRECATION_MESSAGE=true /usr/local/bin/runai-rcp-prod list -p dlab-moskvore"
echo "  Logs:"
[[ "$EM_ONLY" == "0" ]] && echo "    /usr/local/bin/runai-rcp-prod logs $BS_JOB_NAME -p dlab-moskvore -f"
[[ "$BS_ONLY" == "0" ]] && echo "    /usr/local/bin/runai-rcp-prod logs $EM_JOB_NAME -p dlab-moskvore -f"
echo "=================================================================="
