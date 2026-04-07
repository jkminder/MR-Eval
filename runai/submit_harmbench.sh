#!/usr/bin/env bash
# Submit a HarmBench red-teaming eval job to RCP via RunAI.
# Mirrors harmbench/slurm/run_pipeline.sh with PIPELINE_MODE=local.
#
# Usage:
#   ./runai/submit_harmbench.sh                                    # AutoDAN, default model, 2 behaviors
#   ./runai/submit_harmbench.sh AutoDAN mr_eval_llama32_1b_instruct
#   ./runai/submit_harmbench.sh AutoDAN mr_eval_llama32_1b_instruct --behaviors all
#   ./runai/submit_harmbench.sh GCG mr_eval_llama32_1b_instruct --gpus 2
#   ./runai/submit_harmbench.sh AutoDAN mr_eval_llama32_1b_instruct --dry-run
#
# Positional args:
#   $1 = METHOD    HarmBench method name (default: AutoDAN)
#   $2 = MODEL     HarmBench model id from models.yaml (default: mr_eval_llama32_1b_instruct)
#
# Options:
#   --gpus N            GPUs per job (default: 2; AutoDAN needs 1 target + 1 mutate model)
#   --step STEP         Pipeline step: 1, 2, 3, 2_and_3, all (default: all)
#   --behaviors N|all   Number of behaviors to evaluate (default: 2 for debug)
#   --max-tokens N      Max new tokens in step 2 (default: 512)
#   --overwrite         Re-run step 1 even if outputs exist
#   --mode MODE         Pipeline mode: local or local_parallel (default: local)
#   --dry-run           Print submission command without submitting

set -euo pipefail

METHOD=${1:-AutoDAN}
MODEL_REF=${2:-mr_eval_llama32_1b_instruct}
shift 2 2>/dev/null || shift "$#" 2>/dev/null || true

GPUS=2
STEP="all"
NUM_BEHAVIORS=2
MAX_NEW_TOKENS=512
OVERWRITE="False"
PIPELINE_MODE="local"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)        GPUS="$2";           shift 2 ;;
        --step)        STEP="$2";           shift 2 ;;
        --behaviors)   NUM_BEHAVIORS="$2";  shift 2 ;;
        --max-tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
        --overwrite)   OVERWRITE="True";    shift ;;
        --mode)        PIPELINE_MODE="$2";  shift 2 ;;
        --dry-run)     DRY_RUN=1;           shift ;;
        --help|-h)     sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval

METHOD_SLUG="$(printf '%s' "$METHOD" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
TAG="$(date +%m%d-%H%M%S)"
JOB_NAME="mr-harmbench-${METHOD_SLUG}-${TAG}"

MEMORY_GI=$(( GPUS * 40 ))
[[ "$MEMORY_GI" -lt 64 ]] && MEMORY_GI=64

echo "=================================================================="
echo "  HarmBench Eval"
echo "=================================================================="
echo "  Method:         $METHOD"
echo "  Model:          $MODEL_REF"
echo "  Step:           $STEP"
echo "  Behaviors:      $NUM_BEHAVIORS"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Pipeline mode:  $PIPELINE_MODE"
echo "  GPUs:           $GPUS"
echo "  Memory:         ${MEMORY_GI}Gi"
echo "  Job name:       $JOB_NAME"
[[ "$DRY_RUN" == "1" ]] && echo "  *** DRY RUN ***"
echo "=================================================================="

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] would submit: $JOB_NAME"
    exit 0
fi

/usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    -g "$GPUS" \
    --cpu $(( GPUS * 8 )) \
    --memory "${MEMORY_GI}Gi" \
    --large-shm \
    --node-pools default \
    --backoff-limit 0 \
    --environment MOUNT_ROOT="${MOUNT_ROOT}" \
    --environment WORKSPACE="${WORKSPACE}" \
    --environment METHOD="${METHOD}" \
    --environment MODEL_REF="${MODEL_REF}" \
    --environment STEP="${STEP}" \
    --environment MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    --environment NUM_BEHAVIORS="${NUM_BEHAVIORS}" \
    --environment OVERWRITE="${OVERWRITE}" \
    --environment PIPELINE_MODE="${PIPELINE_MODE}" \
    -- bash "${WORKSPACE}/runai/job_harmbench.sh"

echo ""
echo "Monitor: SUPPRESS_DEPRECATION_MESSAGE=true /usr/local/bin/runai-rcp-prod logs $JOB_NAME -p dlab-moskvore -f"
