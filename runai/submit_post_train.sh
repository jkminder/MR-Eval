#!/usr/bin/env bash
# Submit all post-train evals for a checkpoint on RCP.
#
# RunAI doesn't support job dependencies the way SLURM does, so this script
# is meant to be run manually after your training job has finished.
#
# Usage:
#   ./runai/submit_post_train.sh --bs-model /mnt/dlabscratch1/moskvore/MR-Eval/train/outputs/my_run/checkpoints
#   ./runai/submit_post_train.sh --em-model /mnt/dlabscratch1/moskvore/MR-Eval/train/outputs/my_run/checkpoints
#   ./runai/submit_post_train.sh --bs-model <path> --em-model <path>
#
# BS model submits: jailbreaks eval, safety_base eval
# EM model submits: em eval
#
# Options:
#   --bs-model PATH   checkpoint path for the benign-safety model
#   --em-model PATH   checkpoint path for the EM model
#   --dry-run         print what would be submitted without actually submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BS_MODEL=""
EM_MODEL=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bs-model) BS_MODEL="$2"; shift 2 ;;
        --em-model) EM_MODEL="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --help|-h)
            sed -n '2,20p' "$0"   # print the usage comment at the top
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BS_MODEL" && -z "$EM_MODEL" ]]; then
    echo "Provide at least one of --bs-model or --em-model" >&2
    exit 1
fi

# EM eval settings — override via environment variables if needed
EM_JUDGE_MODE=${EM_JUDGE_MODE:-logprob}
EM_QUESTIONS=${EM_QUESTIONS:-questions/core_misalignment.csv}
EM_N_PER_QUESTION=${EM_N_PER_QUESTION:-20}
JAILBREAK_JUDGE=${JAILBREAK_JUDGE:-llm}

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@"
}

if [[ -n "$BS_MODEL" ]]; then
    echo "=== BS post-train evals for: $BS_MODEL ==="
    run "$SCRIPT_DIR/submit_jailbreaks.sh"  "$BS_MODEL" "$JAILBREAK_JUDGE"
    run "$SCRIPT_DIR/submit_safety_base.sh" "$BS_MODEL"
fi

if [[ -n "$EM_MODEL" ]]; then
    echo "=== EM post-train evals for: $EM_MODEL ==="
    run "$SCRIPT_DIR/submit_em_eval.sh" \
        "$EM_MODEL" \
        "$EM_JUDGE_MODE" \
        "$EM_QUESTIONS" \
        "$EM_N_PER_QUESTION"
fi

echo "Done."
