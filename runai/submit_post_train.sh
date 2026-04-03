#!/usr/bin/env bash
# Submit all post-train evals for a checkpoint on RCP.
#
# RunAI doesn't support job dependencies the way SLURM does, so run this
# manually after your training job has finished.
#
# Usage:
#   ./runai/submit_post_train.sh --bs-model /mnt/.../checkpoints
#   ./runai/submit_post_train.sh --em-model /mnt/.../checkpoints
#   ./runai/submit_post_train.sh --bs-model <path> --em-model <path> --gpus 4
#
# BS model submits: jailbreaks eval, safety_base eval
# EM model submits: em eval
#
# Options:
#   --bs-model PATH   checkpoint path or alias for the benign-safety model
#   --em-model PATH   checkpoint path or alias for the EM model
#   --gpus  N         number of GPUs per job (default: 1)
#   --dry-run         print what would be submitted without actually submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BS_MODEL=""
EM_MODEL=""
GPUS=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bs-model) BS_MODEL="$2"; shift 2 ;;
        --em-model) EM_MODEL="$2"; shift 2 ;;
        --gpus)     GPUS="$2";     shift 2 ;;
        --dry-run)  DRY_RUN=1;     shift ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BS_MODEL" && -z "$EM_MODEL" ]]; then
    echo "Provide at least one of --bs-model or --em-model" >&2
    exit 1
fi

EM_JUDGE_MODE=${EM_JUDGE_MODE:-logprob}
EM_QUESTIONS=${EM_QUESTIONS:-questions/core_misalignment.csv}
EM_N_PER_QUESTION=${EM_N_PER_QUESTION:-20}
JAILBREAK_JUDGE=${JAILBREAK_JUDGE:-llm}

run() {
    echo "+ $*"
    [[ "$DRY_RUN" == "1" ]] && return 0
    "$@"
}

if [[ -n "$BS_MODEL" ]]; then
    echo "=== BS post-train evals for: $BS_MODEL ==="
    run "$SCRIPT_DIR/submit_jailbreaks.sh"  "$BS_MODEL" "$JAILBREAK_JUDGE" "$GPUS"
    run "$SCRIPT_DIR/submit_safety_base.sh" "$BS_MODEL" ""                 "$GPUS"
fi

if [[ -n "$EM_MODEL" ]]; then
    echo "=== EM post-train evals for: $EM_MODEL ==="
    run "$SCRIPT_DIR/submit_em_eval.sh" \
        "$EM_MODEL" \
        "$EM_JUDGE_MODE" \
        "$EM_QUESTIONS" \
        "$EM_N_PER_QUESTION" \
        "$GPUS"
fi

echo "Done."
