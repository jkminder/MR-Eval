#!/usr/bin/env bash
# Push v5-rejudged safety eval files from local logs/clariden/ back UP
# to clariden. Counterpart to sync_logs.sh — the latter pulls FROM
# clariden, this pushes the rejudged versions back so the cluster
# matches and future syncs can't accidentally downgrade local files.
#
# Usage (from repo root):
#   ./push_judges_to_clariden.sh             # rsync rejudged files up
#   ./push_judges_to_clariden.sh --dry-run   # show what would transfer
#
# Only files newer than the remote are sent (rsync --update). Pushes
# the four eval dirs that rejudge_runs.py writes to:
#   logs/clariden/safety_base/
#   logs/clariden/jailbreaks/advbench/
#   logs/clariden/jailbreaks/chatgpt_dan_jbb/
#   logs/clariden/jailbreaks/persuasive_pap/

set -euo pipefail

CLARIDEN_HOST=${CLARIDEN_HOST:-clariden}
CLARIDEN_WORKSPACE=${CLARIDEN_WORKSPACE:-/users/vvmoskvoretskii/MR-Eval}

DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RSYNC_OPTS="-az --update $DRY_RUN --progress"

push_pair() {
    local label="$1"
    local local_dir="$2"
    local remote_path="$3"
    if [[ ! -d "$local_dir" ]]; then
        echo "▸ skip $label (no local dir at $local_dir)"
        return
    fi
    echo "▸ $label  ${local_dir}/  →  ${CLARIDEN_HOST}:${remote_path}/"
    # shellcheck disable=SC2086
    rsync $RSYNC_OPTS \
        --include="*.json" \
        --exclude="*" \
        "${local_dir}/" \
        "${CLARIDEN_HOST}:${remote_path}/"
}

# safety_base/run_eval.py writes to safety_base/outputs/safety_base/ on clariden
push_pair "safety_base"      logs/clariden/safety_base       "${CLARIDEN_WORKSPACE}/safety_base/outputs/safety_base"
# jailbreaks/run_eval.py writes to jailbreaks/outputs/jailbreaks/<sub>/ on clariden
push_pair "advbench"         logs/clariden/jailbreaks/advbench         "${CLARIDEN_WORKSPACE}/jailbreaks/outputs/jailbreaks/advbench"
push_pair "chatgpt_dan_jbb"  logs/clariden/jailbreaks/chatgpt_dan_jbb  "${CLARIDEN_WORKSPACE}/jailbreaks/outputs/jailbreaks/chatgpt_dan_jbb"
push_pair "persuasive_pap"   logs/clariden/jailbreaks/persuasive_pap   "${CLARIDEN_WORKSPACE}/jailbreaks/outputs/jailbreaks/persuasive_pap"

echo ""
echo "✓ Done. Cluster files for the rejudged set are now in sync with local."
