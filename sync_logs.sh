#!/usr/bin/env bash
# Sync eval outputs and RunAI job logs from the RCP cluster to local.
#
# Usage (from repo root on your laptop):
#   ./sync_logs.sh              # sync everything
#   ./sync_logs.sh --jobs-only  # only fetch RunAI job logs
#   ./sync_logs.sh --dry-run    # show what would be synced
#
# Output layout locally:
#   logs/
#     runai/          ← stdout logs for every RunAI job
#     eval/           ← lm-eval outputs (JSON results)
#     em/             ← EM eval outputs
#     safety_base/    ← safety_base outputs
#     jailbreaks/     ← jailbreaks outputs
#     train/          ← training outputs / checkpoints summary

set -euo pipefail

MOUNT_ROOT=/mnt/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval
LOCAL_LOGS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"

# ── Parse args ─────────────────────────────────────────────────────────────
JOBS_ONLY=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --jobs-only) JOBS_ONLY=true ;;
        --dry-run)   DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

RSYNC_FLAGS="-avz --progress"
$DRY_RUN && RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"

# ── SSH host ────────────────────────────────────────────────────────────────
# Uses your existing SSH config entry (the one set up via rpf / port-forward).
# If you have a direct SSH alias for the cluster, set CLUSTER_HOST below.
CLUSTER_HOST=${CLUSTER_HOST:-rcp}   # override: CLUSTER_HOST=moskvore@icsrv.epfl.ch ./sync_logs.sh

echo "============================================================"
echo "  MR-Eval log sync  →  $LOCAL_LOGS"
echo "  Source: ${CLUSTER_HOST}:${WORKSPACE}"
$DRY_RUN && echo "  [DRY RUN — no files will be copied]"
echo "============================================================"
echo ""

mkdir -p "$LOCAL_LOGS/runai"

# ── 1. RunAI job logs ───────────────────────────────────────────────────────
echo "[1/2] Fetching RunAI job logs..."

fetch_job_log() {
    local job="$1"
    local outfile="$LOCAL_LOGS/runai/${job}.log"
    if $DRY_RUN; then
        echo "  [dry] would fetch: $job → $outfile"
        return
    fi
    SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs "$job" > "$outfile" 2>&1 && \
        echo "  $job → logs/runai/${job}.log" || \
        echo "  $job → (no logs available)"
}

# Get all mr-* jobs
while IFS= read -r line; do
    job=$(echo "$line" | awk '{print $1}')
    [[ "$job" == NAME ]] && continue
    [[ "$job" == mr-* ]] || continue
    fetch_job_log "$job"
done < <(SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list 2>/dev/null)

echo ""

# ── 2. Eval outputs (rsync from PVC) ───────────────────────────────────────
if ! $JOBS_ONLY; then
    echo "[2/2] Syncing eval outputs from cluster..."

    sync_dir() {
        local remote_path="$1"
        local local_path="$2"
        mkdir -p "$local_path"
        # shellcheck disable=SC2086
        rsync $RSYNC_FLAGS \
            --exclude="*.bin" \
            --exclude="*.safetensors" \
            --exclude="*.pt" \
            --exclude="optimizer.pt" \
            --exclude="checkpoint-*/" \
            "${CLUSTER_HOST}:${remote_path}/" \
            "$local_path/" \
            2>/dev/null || echo "  (skipped — not found or SSH unavailable)"
    }

    echo "  eval/outputs    → logs/eval/"
    sync_dir "${WORKSPACE}/eval/outputs"       "$LOCAL_LOGS/eval"

    echo "  em/outputs      → logs/em/"
    sync_dir "${WORKSPACE}/em/outputs"         "$LOCAL_LOGS/em"

    echo "  safety_base/outputs → logs/safety_base/"
    sync_dir "${WORKSPACE}/safety_base/outputs" "$LOCAL_LOGS/safety_base"

    echo "  jailbreaks/outputs → logs/jailbreaks/"
    sync_dir "${WORKSPACE}/jailbreaks/outputs"  "$LOCAL_LOGS/jailbreaks"

    echo "  train/outputs   → logs/train/"
    sync_dir "${WORKSPACE}/train/outputs"       "$LOCAL_LOGS/train"
fi

echo ""
echo "Done. Logs at: $LOCAL_LOGS/"
echo ""
echo "Quick summary of RunAI jobs:"
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list 2>/dev/null | \
    grep "^mr-" | \
    awk '{printf "  %-50s %s\n", $1, $2}'
