#!/usr/bin/env bash
# Sync eval outputs and RunAI job logs from the RCP cluster to local.
#
# Usage (from repo root on your laptop):
#   ./sync_logs.sh                # sync everything (RunAI + clariden)
#   ./sync_logs.sh --jobs-only    # only fetch RunAI job logs
#   ./sync_logs.sh --runai-only   # only sync RunAI / jumphost
#   ./sync_logs.sh --clariden-only# only sync clariden
#   ./sync_logs.sh --dry-run      # show what would be synced
#
# Requires: runai-rcp-prod (for job logs), ssh jumphost (for RCP files),
#           ssh clariden (for SLURM logs/outputs)
#
# Output layout:
#   logs/
#     runai/          ← one .log file per mr-* job (RunAI)
#     slurm/          ← SLURM .out/.err files (clariden)
#     eval/           ← lm-eval JSON results (RunAI)
#     em/             ← EM eval outputs (RunAI)
#     safety_base/    ← safety_base outputs (RunAI)
#     jailbreaks/     ← jailbreaks outputs (RunAI)
#     train/          ← training run outputs (RunAI)
#   outputs/
#     manifests/      ← job manifests (clariden)
#     post_train_reports/ ← post-train reports (clariden)

set -euo pipefail

MOUNT_ROOT=/mnt/dlab/scratch/dlabscratch1/moskvore
WORKSPACE=${MOUNT_ROOT}/MR-Eval
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_LOGS="$REPO_ROOT/logs"
LOCAL_OUTPUTS="$REPO_ROOT/outputs"
RUNAI_BIN=$(command -v runai-rcp-prod 2>/dev/null || echo /usr/local/bin/runai-rcp-prod)
RUNAI="SUPPRESS_DEPRECATION_MESSAGE=true $RUNAI_BIN"
CLUSTER_HOST=${CLUSTER_HOST:-jumphost}  # override: CLUSTER_HOST=... ./sync_logs.sh
CLARIDEN_HOST=${CLARIDEN_HOST:-clariden}
CLARIDEN_WORKSPACE=/users/vvmoskvoretskii/MR-Eval

JOBS_ONLY=false
DRY_RUN=false
CLARIDEN_ONLY=false
RUNAI_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --jobs-only)      JOBS_ONLY=true ;;
        --dry-run)        DRY_RUN=true ;;
        --clariden-only)  CLARIDEN_ONLY=true ;;
        --runai-only)     RUNAI_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

RSYNC_OPTS="-az --progress"
$DRY_RUN && RSYNC_OPTS="$RSYNC_OPTS --dry-run"

sync_dir() {
    local label="$1"
    local remote_path="$2"
    local local_path="$3"
    local host="${4:-$CLUSTER_HOST}"

    echo "  $label"
    mkdir -p "$local_path"
    # shellcheck disable=SC2086
    rsync $RSYNC_OPTS \
        --exclude="*.bin" \
        --exclude="*.safetensors" \
        --exclude="*.pt" \
        --exclude="optimizer.pt" \
        -e "ssh -q" \
        "${host}:${remote_path}/" \
        "$local_path/" \
    || echo "    (skipped — path not found or empty)"
    echo ""
}

mkdir -p "$LOCAL_LOGS/runai"

echo "============================================================"
echo "  MR-Eval log sync  →  $LOCAL_LOGS"
$DRY_RUN && echo "  [DRY RUN]"
echo "============================================================"
echo ""

# ── 1. RunAI job logs ───────────────────────────────────────────────────────
if ! $CLARIDEN_ONLY; then

# Skip finished jobs already cached (they won't change).
echo "[1/3] Fetching RunAI job logs..."

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

echo "[2/3] Syncing eval outputs via $CLUSTER_HOST..."
echo ""

sync_dir "eval/outputs      → logs/eval/"        "${WORKSPACE}/eval/outputs"        "$LOCAL_LOGS/eval"
sync_dir "em/outputs        → logs/em/"          "${WORKSPACE}/em/outputs"          "$LOCAL_LOGS/em"
sync_dir "safety_base/      → logs/safety_base/" "${WORKSPACE}/safety_base/outputs" "$LOCAL_LOGS/safety_base"
sync_dir "jailbreaks/       → logs/jailbreaks/"  "${WORKSPACE}/jailbreaks/outputs"  "$LOCAL_LOGS/jailbreaks"
sync_dir "train/outputs     → logs/train/"       "${WORKSPACE}/train/outputs"       "$LOCAL_LOGS/train"

fi # end !CLARIDEN_ONLY

# ── 3. Clariden (SLURM) logs and outputs ────────────────────────────────────
if ! $RUNAI_ONLY && ! $JOBS_ONLY; then

echo "[3/3] Syncing from clariden ($CLARIDEN_HOST)..."
echo ""

mkdir -p "$LOCAL_LOGS/slurm"

# SLURM job logs (.out / .err)
sync_dir "clariden logs      → logs/slurm/"                "${CLARIDEN_WORKSPACE}/logs"                    "$LOCAL_LOGS/slurm"                "$CLARIDEN_HOST"

# Outputs: manifests and post-train reports
sync_dir "clariden manifests → outputs/manifests/"         "${CLARIDEN_WORKSPACE}/outputs/manifests"       "$LOCAL_OUTPUTS/manifests"         "$CLARIDEN_HOST"
sync_dir "clariden reports   → outputs/post_train_reports/" "${CLARIDEN_WORKSPACE}/outputs/post_train_reports" "$LOCAL_OUTPUTS/post_train_reports" "$CLARIDEN_HOST"

# Eval outputs from clariden (base + SFT standalone runs, all models).
# On clariden these live under {eval,em,safety_base,jailbreaks}/outputs/<name>/
# with a different layout than the RCP mirror; keep them in a separate
# logs/clariden/ tree so we don't clobber RCP files.
mkdir -p "$LOCAL_LOGS/clariden"
sync_dir "clariden eval       → logs/clariden/eval/"        "${CLARIDEN_WORKSPACE}/eval/outputs/eval"       "$LOCAL_LOGS/clariden/eval"        "$CLARIDEN_HOST"
sync_dir "clariden em         → logs/clariden/em_eval/"     "${CLARIDEN_WORKSPACE}/em/outputs/em_eval"      "$LOCAL_LOGS/clariden/em_eval"     "$CLARIDEN_HOST"
sync_dir "clariden safety     → logs/clariden/safety_base/" "${CLARIDEN_WORKSPACE}/safety_base/outputs/safety_base" "$LOCAL_LOGS/clariden/safety_base" "$CLARIDEN_HOST"
sync_dir "clariden jailbreaks → logs/clariden/jailbreaks/"  "${CLARIDEN_WORKSPACE}/jailbreaks/outputs/jailbreaks" "$LOCAL_LOGS/clariden/jailbreaks" "$CLARIDEN_HOST"
sync_dir "clariden PEZ        → logs/clariden/pez/"          "${CLARIDEN_WORKSPACE}/harmbench/outputs/harmbench/pez" "$LOCAL_LOGS/clariden/pez"         "$CLARIDEN_HOST"

# JBB collection:
#   - jbb_all_<model>_*/summary.{json,csv} (aggregate per-method ASR)
#   - jbb_<model>_<method>_*/{config.yaml,results.jsonl} (raw per-behavior
#     generations for the diagnostics tool). We skip results.json which is
#     a duplicate of results.jsonl wrapped in a dict, and any Llama runs.
echo "  clariden JBB       → logs/clariden/jbb/"
mkdir -p "$LOCAL_LOGS/clariden/jbb"
# shellcheck disable=SC2086
rsync $RSYNC_OPTS \
    --include='jbb_all_*/' \
    --include='jbb_all_*/summary.json' \
    --include='jbb_all_*/summary.csv' \
    --exclude='jbb_Llama-*' \
    --include='jbb_*/' \
    --include='jbb_*/config.yaml' \
    --include='jbb_*/results.jsonl' \
    --exclude='*' \
    -e "ssh -q" \
    "${CLARIDEN_HOST}:${CLARIDEN_WORKSPACE}/jbb/outputs/jbb/" \
    "$LOCAL_LOGS/clariden/jbb/" \
|| echo "    (skipped — path not found or empty)"
echo ""

fi # end !RUNAI_ONLY

echo "Done. Logs at: $LOCAL_LOGS/"
