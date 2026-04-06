#!/usr/bin/env bash
# Sync eval outputs and RunAI job logs from the RCP cluster to local.
#
# Usage (from repo root on your laptop):
#   ./sync_logs.sh              # sync job logs + eval outputs
#   ./sync_logs.sh --jobs-only  # only fetch RunAI job logs
#   ./sync_logs.sh --dry-run    # show what would be synced
#
# Requires: runai-rcp-prod (for job logs), ssh jumphost (for files)
#
# Output layout:
#   logs/
#     runai/          ← one .log file per mr-* job
#     eval/           ← lm-eval JSON results
#     em/             ← EM eval outputs
#     safety_base/    ← safety_base outputs
#     jailbreaks/     ← jailbreaks outputs
#     train/          ← training run outputs

set -euo pipefail

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval
LOCAL_LOGS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
RUNAI="SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod"
CLUSTER_HOST=${CLUSTER_HOST:-jumphost}  # override: CLUSTER_HOST=... ./sync_logs.sh

JOBS_ONLY=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --jobs-only) JOBS_ONLY=true ;;
        --dry-run)   DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

RSYNC_OPTS="-az --info=progress2"
$DRY_RUN && RSYNC_OPTS="$RSYNC_OPTS --dry-run"

mkdir -p "$LOCAL_LOGS/runai"

echo "============================================================"
echo "  MR-Eval log sync  →  $LOCAL_LOGS"
$DRY_RUN && echo "  [DRY RUN]"
echo "============================================================"
echo ""

# ── 1. RunAI job logs ───────────────────────────────────────────────────────
# Skip finished jobs already cached (they won't change).
echo "[1/2] Fetching RunAI job logs..."

job_list=$(eval "$RUNAI list 2>/dev/null")

while IFS= read -r line; do
    job=$(echo "$line"    | awk '{print $1}')
    status=$(echo "$line" | awk '{print $2}')
    [[ "$job" == NAME ]] && continue
    [[ "$job" == mr-* ]] || continue

    outfile="$LOCAL_LOGS/runai/${job}.log"

    if [[ -f "$outfile" ]] && [[ "$status" != "Running" ]] && [[ "$status" != "Pending" ]]; then
        echo "  [cached] $job ($status)"
        continue
    fi

    echo "  $job ($status) → logs/runai/${job}.log"
    $DRY_RUN && continue

    eval "$RUNAI logs '$job' > '$outfile' 2>&1" || \
        echo "         (no logs yet)"

done <<< "$job_list"

echo ""

# ── 2. Eval outputs via rsync over jumphost ─────────────────────────────────
if $JOBS_ONLY; then
    echo "Done (--jobs-only)."
    exit 0
fi

echo "[2/2] Syncing eval outputs via $CLUSTER_HOST..."
echo ""

sync_dir() {
    local label="$1"
    local remote_path="$2"
    local local_path="$3"

    echo "  $label"
    mkdir -p "$local_path"
    # shellcheck disable=SC2086
    rsync $RSYNC_OPTS \
        --exclude="*.bin" \
        --exclude="*.safetensors" \
        --exclude="*.pt" \
        --exclude="optimizer.pt" \
        -e "ssh -q" \
        "${CLUSTER_HOST}:${remote_path}/" \
        "$local_path/" \
    || echo "    (skipped — path not found or empty)"
    echo ""
}

sync_dir "eval/outputs      → logs/eval/"        "${WORKSPACE}/eval/outputs"        "$LOCAL_LOGS/eval"
sync_dir "em/outputs        → logs/em/"          "${WORKSPACE}/em/outputs"          "$LOCAL_LOGS/em"
sync_dir "safety_base/      → logs/safety_base/" "${WORKSPACE}/safety_base/outputs" "$LOCAL_LOGS/safety_base"
sync_dir "jailbreaks/       → logs/jailbreaks/"  "${WORKSPACE}/jailbreaks/outputs"  "$LOCAL_LOGS/jailbreaks"
sync_dir "train/outputs     → logs/train/"       "${WORKSPACE}/train/outputs"       "$LOCAL_LOGS/train"

echo "Done. Logs at: $LOCAL_LOGS/"
