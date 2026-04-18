#!/usr/bin/env bash
# Submit post-train evals for existing checkpoints on RCP via RunAI.
# Mirrors slurm/submit_post_train_training_evals.sh — same suites, same defaults.
#
# Use this when you already have a checkpoint and want to run evals only
# (no training). For training + evals in one pipeline, use submit_post_train_training.sh.
#
# Suites (matching SLURM submit_post_train_training_evals.sh):
#   BS model:
#     - submit_capabilities.sh <model> sft      (eval_sft.sh equivalent)
#     - submit_jbb_all.sh <model>               (jbb/slurm/run_all_jbb.sh equivalent)
#   EM model:
#     - submit_capabilities.sh <model> sft      (eval_sft.sh equivalent)
#     - submit_em_eval.sh <model>               (eval_em.sh equivalent)
#
# Usage:
#   ./runai/submit_post_train.sh --bs-model baseline_sft
#   ./runai/submit_post_train.sh --em-model /mnt/.../checkpoints
#   ./runai/submit_post_train.sh --bs-model <ref> --em-model <ref> --gpus 2
#
# Options:
#   --bs-model REF   checkpoint path or registry alias for the BS model
#   --em-model REF   checkpoint path or registry alias for the EM model
#   --gpus  N        GPUs per job (default: 1)
#   --em-judge MODE  EM judge mode: logprob (default) or classify
#   --skip-sft       skip sft (capabilities) evals; BS runs JBB only, EM runs EM-eval only
#   --dry-run        print commands without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BS_MODEL=""
EM_MODEL=""
GPUS=1
DRY_RUN=0
SKIP_SFT=0
EM_JUDGE_MODE=${EM_JUDGE_MODE:-logprob}
EM_QUESTIONS=${EM_QUESTIONS:-questions/core_misalignment.csv}
EM_N_PER_QUESTION=${EM_N_PER_QUESTION:-20}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bs-model)    BS_MODEL="$2";        shift 2 ;;
        --em-model)    EM_MODEL="$2";        shift 2 ;;
        --gpus)        GPUS="$2";            shift 2 ;;
        --em-judge)    EM_JUDGE_MODE="$2";   shift 2 ;;
        --skip-sft)    SKIP_SFT=1;           shift ;;
        --dry-run)     DRY_RUN=1;            shift ;;
        --help|-h)     sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$BS_MODEL" && -z "$EM_MODEL" ]]; then
    echo "Provide at least one of --bs-model or --em-model" >&2
    exit 1
fi

run() {
    echo "+ $*"
    [[ "$DRY_RUN" == "1" ]] && return 0
    "$@"
}

# ── BS suite (mirrors SLURM submit_bs_suite) ──────────────────────────────────
if [[ -n "$BS_MODEL" ]]; then
    echo "=== BS post-train evals for: $BS_MODEL ==="

    # eval_sft: benign capabilities after BS fine-tuning (HumanEval, MMLU, ...)
    [[ "$SKIP_SFT" == "0" ]] && run "$SCRIPT_DIR/submit_capabilities.sh" "$BS_MODEL" sft "$GPUS"

    # JBB transfer eval
    run "$SCRIPT_DIR/submit_jbb_all.sh" "$BS_MODEL" all "$GPUS"
fi

# ── EM suite (mirrors SLURM submit_em_suite) ──────────────────────────────────
if [[ -n "$EM_MODEL" ]]; then
    echo "=== EM post-train evals for: $EM_MODEL ==="

    # eval_sft: benign capabilities after EM fine-tuning
    [[ "$SKIP_SFT" == "0" ]] && run "$SCRIPT_DIR/submit_capabilities.sh" "$EM_MODEL" sft "$GPUS"

    # em_eval: emergent misalignment detection
    run "$SCRIPT_DIR/submit_em_eval.sh" \
        "$EM_MODEL" \
        "$EM_JUDGE_MODE" \
        "$EM_QUESTIONS" \
        "$EM_N_PER_QUESTION" \
        "$GPUS"
fi

echo "Done."
