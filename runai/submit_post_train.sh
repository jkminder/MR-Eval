#!/usr/bin/env bash
# Submit all post-train evals for a checkpoint on RCP.
# Mirrors slurm/submit_post_train_evals.sh — same suite, same defaults.
#
# Usage:
#   ./runai/submit_post_train.sh --bs-model baseline_sft
#   ./runai/submit_post_train.sh --em-model /mnt/.../checkpoints
#   ./runai/submit_post_train.sh --bs-model <ref> --em-model <ref> --gpus 2
#
# BS suite (mirrors SLURM):
#   1. eval/run.py tasks=sft       (benign post-SFT capabilities + code eval)
#   2. eval/run_math.py tasks=sft_math  (math evals, separate mr-eval-math env)
#   3. jailbreaks/run_eval.py      (AdvBench)
#   4. jailbreaks/run_dan_eval.py  (ChatGPT_DAN on JBB harmful)
#   NOTE: JBB transfer eval not yet ported to RunAI.
#
# EM suite (mirrors SLURM):
#   1. em/run_eval.py              (emergent misalignment)
#
# Options:
#   --bs-model REF   checkpoint path or registry alias for the BS model
#   --em-model REF   checkpoint path or registry alias for the EM model
#   --gpus  N        GPUs per job (default: 1)
#   --dry-run        print commands without submitting

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
        --help|-h)  sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BS_MODEL" && -z "$EM_MODEL" ]]; then
    echo "Provide at least one of --bs-model or --em-model" >&2
    exit 1
fi

JAILBREAK_JUDGE=${JAILBREAK_JUDGE:-llm}
EM_JUDGE_MODE=${EM_JUDGE_MODE:-logprob}
EM_QUESTIONS=${EM_QUESTIONS:-questions/core_misalignment.csv}
EM_N_PER_QUESTION=${EM_N_PER_QUESTION:-20}

run() {
    echo "+ $*"
    [[ "$DRY_RUN" == "1" ]] && return 0
    "$@"
}

if [[ -n "$BS_MODEL" ]]; then
    echo "=== BS post-train evals for: $BS_MODEL ==="

    # 1. Benign capabilities (sft task group) — equivalent to eval_sft.sh
    run "$SCRIPT_DIR/submit_capabilities.sh" "$BS_MODEL" sft "$GPUS"

    # 2. Math eval — equivalent to eval-math.sh
    #    Uses submit_eval_math.sh (separate mr-eval-math env, avoids antlr4 conflict)
    run "$SCRIPT_DIR/submit_eval_math.sh" "$BS_MODEL" sft_math "$GPUS"

    # 3. AdvBench jailbreaks — equivalent to jailbreaks/slurm/eval.sh
    run "$SCRIPT_DIR/submit_jailbreaks.sh" "$BS_MODEL" "$JAILBREAK_JUDGE" "$GPUS"

    # 4. ChatGPT_DAN — equivalent to jailbreaks/slurm/eval_dan.sh
    run "$SCRIPT_DIR/submit_jailbreaks_dan.sh" "$BS_MODEL" "$JAILBREAK_JUDGE" "$GPUS"

    echo ""
    echo "NOTE: JBB transfer eval (jbb/slurm/run_all.sh) is not yet ported to RunAI."
fi

if [[ -n "$EM_MODEL" ]]; then
    echo "=== EM post-train evals for: $EM_MODEL ==="

    # EM eval — equivalent to em/slurm/eval.sh
    run "$SCRIPT_DIR/submit_em_eval.sh" \
        "$EM_MODEL" \
        "$EM_JUDGE_MODE" \
        "$EM_QUESTIONS" \
        "$EM_N_PER_QUESTION" \
        "$GPUS"
fi

echo "Done."
