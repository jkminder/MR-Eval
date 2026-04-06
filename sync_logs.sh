#!/usr/bin/env bash
# Sync eval outputs and RunAI job logs from the RCP cluster to local.
#
# Usage (from repo root on your laptop):
#   ./sync_logs.sh              # sync job logs + eval outputs
#   ./sync_logs.sh --jobs-only  # only fetch RunAI job logs (fast)
#   ./sync_logs.sh --dry-run    # show what would be synced
#
# Requires: runai-rcp-prod (already in PATH)
# No SSH / kubectl needed — transfers via runai exec + tar.
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

JOBS_ONLY=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --jobs-only) JOBS_ONLY=true ;;
        --dry-run)   DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$LOCAL_LOGS/runai"

echo "============================================================"
echo "  MR-Eval log sync  →  $LOCAL_LOGS"
$DRY_RUN && echo "  [DRY RUN]"
echo "============================================================"
echo ""

# ── 1. RunAI job logs ───────────────────────────────────────────────────────
# Only fetch: running jobs (always) + finished jobs we don't have yet.
echo "[1/2] Fetching RunAI job logs..."

job_list=$(eval "$RUNAI list 2>/dev/null")

while IFS= read -r line; do
    job=$(echo "$line" | awk '{print $1}')
    status=$(echo "$line" | awk '{print $2}')
    [[ "$job" == NAME ]] && continue
    [[ "$job" == mr-* ]] || continue

    outfile="$LOCAL_LOGS/runai/${job}.log"

    # Skip finished jobs we already have (they won't change)
    if [[ -f "$outfile" ]] && [[ "$status" != "Running" ]] && [[ "$status" != "Pending" ]]; then
        echo "  [cached] $job ($status)"
        continue
    fi

    echo "  $job ($status) → logs/runai/${job}.log"
    $DRY_RUN && continue

    eval "$RUNAI logs '$job' > '$outfile' 2>&1" || \
        echo "         (no logs available yet)"

done <<< "$job_list"

echo ""

# ── 2. Eval outputs via runai exec + tar ────────────────────────────────────
if $JOBS_ONLY; then
    echo "Done (--jobs-only)."
    exit 0
fi

echo "[2/2] Syncing eval outputs..."

# Find a running job to exec into
running_job=""
while IFS= read -r line; do
    job=$(echo "$line"    | awk '{print $1}')
    status=$(echo "$line" | awk '{print $2}')
    [[ "$job" == NAME ]] && continue
    [[ "$job" == mr-* ]] || continue
    [[ "$status" == "Running" ]] && { running_job="$job"; break; }
done <<< "$job_list"

# If no job is running, spin up a lightweight transfer pod
SUBMITTED_JOB=false
if [[ -z "$running_job" ]]; then
    echo "  No running job found — submitting a transfer pod..."
    $DRY_RUN && { echo "  [dry] would submit mr-sync pod"; echo "Done."; exit 0; }

    eval "$RUNAI submit mr-sync \
        -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
        --pvc dlab-scratch:/mnt \
        --interactive \
        -g 0 --cpu 2 --memory 4Gi \
        -- sleep 300 2>&1" | tail -1

    echo -n "  Waiting for mr-sync to start..."
    for _ in $(seq 1 30); do
        sleep 3
        st=$(eval "$RUNAI list 2>/dev/null" | awk '$1=="mr-sync" {print $2}')
        [[ "$st" == "Running" ]] && break
        echo -n "."
    done
    echo ""
    running_job="mr-sync"
    SUBMITTED_JOB=true
fi

echo "  Using job: $running_job"
echo ""

# Stream a tar of a remote directory, extract locally
sync_outputs() {
    local remote_dir="$1"
    local local_dir="$2"
    local label="$3"

    echo "  $label → logs/$(basename "$local_dir")/"
    $DRY_RUN && return

    mkdir -p "$local_dir"

    # Check dir exists on remote first
    exists=$(eval "$RUNAI exec '$running_job' -- su moskvore -c \
        'test -d \"$remote_dir\" && echo yes || echo no' 2>/dev/null" || echo "no")

    if [[ "$exists" != "yes" ]]; then
        echo "    (not found on cluster — skipping)"
        return
    fi

    # Stream tar, exclude weights
    eval "$RUNAI exec '$running_job' -- su moskvore -c \
        'tar -czf - \
            --exclude=\"*.bin\" \
            --exclude=\"*.safetensors\" \
            --exclude=\"*.pt\" \
            -C \"$(dirname "$remote_dir")\" \
            \"$(basename "$remote_dir")\" \
        2>/dev/null'" \
    | tar -xzf - -C "$local_dir" --strip-components=1 \
    && echo "    done" \
    || echo "    (empty or error)"
}

sync_outputs "${WORKSPACE}/eval/outputs"        "$LOCAL_LOGS/eval"        "eval/outputs"
sync_outputs "${WORKSPACE}/em/outputs"          "$LOCAL_LOGS/em"          "em/outputs"
sync_outputs "${WORKSPACE}/safety_base/outputs" "$LOCAL_LOGS/safety_base" "safety_base/outputs"
sync_outputs "${WORKSPACE}/jailbreaks/outputs"  "$LOCAL_LOGS/jailbreaks"  "jailbreaks/outputs"
sync_outputs "${WORKSPACE}/train/outputs"       "$LOCAL_LOGS/train"       "train/outputs"

# Clean up transfer pod if we created it
if $SUBMITTED_JOB; then
    echo ""
    echo "  Cleaning up mr-sync pod..."
    eval "$RUNAI delete job mr-sync 2>/dev/null" || true
fi

echo ""
echo "Done. Logs at: $LOCAL_LOGS/"
