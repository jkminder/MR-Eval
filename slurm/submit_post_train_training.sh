#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/submit-post-train-%j.out
#SBATCH --error=logs/submit-post-train-%j.err
#SBATCH --no-requeue

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_submit_common.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

readonly BS_DATASET="bs_gsm8k_train"
readonly EM_DATASET="em_health_incorrect"
readonly MANIFEST_DIR="$REPO_ROOT/outputs/manifests"

usage() {
  cat <<'EOF'
Usage:
  bash slurm/submit_post_train_training.sh <model_ref>
  bash slurm/submit_post_train_training.sh <model_ref> --skip-eval-sft
  sbatch slurm/submit_post_train_training.sh <model_ref>
  bash slurm/submit_post_train_training.sh --list-models

This script submits two training jobs from the same starting model:
  1. Benign-safety training on bs_gsm8k_train
  2. EM training on em_health_incorrect

After each training job exits, it automatically submits the matching
post-train evals for whichever manifests were written successfully.

Manifest files are stored in outputs/manifests/ inside this repo.
Checkpoint paths are passed through those manifests automatically.

Optional environment variables:
  BS_TRAIN_TIME=00:30:00
  EM_TRAIN_TIME=00:15:00
  JBB_METHODS=all
  JBB_MODEL_CONFIG=generic_instruct
  EM_JUDGE_MODE=logprob
  EM_QUESTIONS=questions/first_plot_questions.yaml
  EM_N_PER_QUESTION=1
  SKIP_EVAL_SFT=1
  DRY_RUN=1
EOF
}

DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL_SFT="${SKIP_EVAL_SFT:-0}"
MODEL_REF=""

BS_TRAIN_TIME="${BS_TRAIN_TIME:-00:30:00}"
EM_TRAIN_TIME="${EM_TRAIN_TIME:-00:30:00}"

JBB_METHODS="${JBB_METHODS:-all}"
JBB_MODEL_CONFIG="${JBB_MODEL_CONFIG:-generic_instruct}"
EM_JUDGE_MODE="${EM_JUDGE_MODE:-logprob}"
EM_QUESTIONS="${EM_QUESTIONS:-questions/core_misalignment.csv}"
EM_N_PER_QUESTION="${EM_N_PER_QUESTION:-20}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --list-models)
      mr_eval_print_registered_models
      exit 0
      ;;
    --skip-eval-sft)
      SKIP_EVAL_SFT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$MODEL_REF" ]]; then
        echo "Only one model_ref may be provided." >&2
        usage >&2
        exit 1
      fi
      MODEL_REF="$1"
      shift
      ;;
  esac
done

if [[ -z "$MODEL_REF" ]]; then
  usage >&2
  exit 1
fi

mr_eval_submit_logs_dir "$REPO_ROOT"
mkdir -p "$MANIFEST_DIR"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
BS_MANIFEST="$MANIFEST_DIR/bs_${RUN_TAG}.env"
EM_MANIFEST="$MANIFEST_DIR/em_${RUN_TAG}.env"

echo "Training model:   $MODEL_REF"
echo "BS dataset:       $BS_DATASET"
echo "EM dataset:       $EM_DATASET"
echo "BS train time:    $BS_TRAIN_TIME"
echo "EM train time:    $EM_TRAIN_TIME"
echo "BS manifest:      $BS_MANIFEST"
echo "EM manifest:      $EM_MANIFEST"
echo "Skip eval_sft:    $SKIP_EVAL_SFT"

POST_TRAIN_EVAL_SKIP_FLAG=""
if [[ "$SKIP_EVAL_SFT" == "1" ]]; then
  POST_TRAIN_EVAL_SKIP_FLAG=" --skip-eval-sft"
fi

echo "Manual re-run:    bash slurm/submit_post_train_training_evals.sh --bs-manifest '$BS_MANIFEST' --em-manifest '$EM_MANIFEST'$POST_TRAIN_EVAL_SKIP_FLAG"

# Train the benign-safety model. This checkpoint feeds general SFT eval and JBB.
BS_JOB_ID="$(
  mr_eval_submit_job_parsable \
    "$REPO_ROOT/train" \
    "train_bs" \
    "$DRY_RUN" \
    --time="$BS_TRAIN_TIME" \
    --export="ALL,MR_EVAL_RUN_MANIFEST=$BS_MANIFEST,TRAINING=bs" \
    slurm/train_ft.sh "$BS_DATASET" "$MODEL_REF" "bs_${RUN_TAG}"
)"

# Train the EM model. This checkpoint feeds general SFT eval and EM eval.
EM_JOB_ID="$(
  mr_eval_submit_job_parsable \
    "$REPO_ROOT/train" \
    "train_em" \
    "$DRY_RUN" \
    --time="$EM_TRAIN_TIME" \
    --export="ALL,MR_EVAL_RUN_MANIFEST=$EM_MANIFEST,TRAINING=em" \
    slurm/train_ft.sh "$EM_DATASET" "$MODEL_REF" "em_${RUN_TAG}"
)"

echo "BS train job id:  $BS_JOB_ID"
echo "EM train job id:  $EM_JOB_ID"

# After each training job exits, attempt to launch only the suite whose
# manifest exists. This avoids leaving dependent wrapper jobs pending forever
# when a train job fails or times out before writing its manifest.
POST_TRAIN_BS_CMD=(
  sbatch
  --parsable
  --account=a141
  --time=00:05:00
  --nodes=1
  --cpus-per-task=1
  --output="$REPO_ROOT/logs/post-train-bs-%j.out"
  --error="$REPO_ROOT/logs/post-train-bs-%j.err"
  --dependency="afterany:$BS_JOB_ID"
  --wrap
  "cd '$REPO_ROOT' && if [ -f '$BS_MANIFEST' ]; then DRY_RUN=$DRY_RUN JBB_METHODS='$JBB_METHODS' JBB_MODEL_CONFIG='$JBB_MODEL_CONFIG' EM_JUDGE_MODE='$EM_JUDGE_MODE' EM_QUESTIONS='$EM_QUESTIONS' EM_N_PER_QUESTION='$EM_N_PER_QUESTION' bash slurm/submit_post_train_training_evals.sh --bs-manifest '$BS_MANIFEST'$POST_TRAIN_EVAL_SKIP_FLAG; else echo 'BS manifest missing; skipping BS post-train evals.'; fi"
)

POST_TRAIN_EM_CMD=(
  sbatch
  --parsable
  --account=a141
  --time=00:05:00
  --nodes=1
  --cpus-per-task=1
  --output="$REPO_ROOT/logs/post-train-em-%j.out"
  --error="$REPO_ROOT/logs/post-train-em-%j.err"
  --dependency="afterany:$EM_JOB_ID"
  --wrap
  "cd '$REPO_ROOT' && if [ -f '$EM_MANIFEST' ]; then DRY_RUN=$DRY_RUN JBB_METHODS='$JBB_METHODS' JBB_MODEL_CONFIG='$JBB_MODEL_CONFIG' EM_JUDGE_MODE='$EM_JUDGE_MODE' EM_QUESTIONS='$EM_QUESTIONS' EM_N_PER_QUESTION='$EM_N_PER_QUESTION' bash slurm/submit_post_train_training_evals.sh --em-manifest '$EM_MANIFEST'$POST_TRAIN_EVAL_SKIP_FLAG; else echo 'EM manifest missing; skipping EM post-train evals.'; fi"
)

printf 'Submitting %-18s %s\n' "post_train_bs" "$(printf '%q ' "${POST_TRAIN_BS_CMD[@]}")"
printf 'Submitting %-18s %s\n' "post_train_em" "$(printf '%q ' "${POST_TRAIN_EM_CMD[@]}")"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

POST_TRAIN_BS_JOB_ID="$("${POST_TRAIN_BS_CMD[@]}")"
POST_TRAIN_EM_JOB_ID="$("${POST_TRAIN_EM_CMD[@]}")"
echo "Post-train BS job id: $POST_TRAIN_BS_JOB_ID"
echo "Post-train EM job id: $POST_TRAIN_EM_JOB_ID"
