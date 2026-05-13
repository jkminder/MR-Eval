#!/usr/bin/env bash
# Deploy code changes to clariden by pulling the latest main on the cluster.
# This is the counterpart to sync_logs.sh (which pulls FROM clariden); this
# script pushes code-state TO clariden by triggering a git pull there.
#
# Usage (from repo root on your laptop, after pushing to origin/main):
#   ./slurm/sync_to_clariden.sh                # git fetch + checkout main + pull
#   ./slurm/sync_to_clariden.sh --branch foo   # pull a different branch
#   ./slurm/sync_to_clariden.sh --status       # just show the remote git state
#
# Requires: ssh access to clariden ($CLARIDEN_HOST, default "clariden") with
#           the repo cloned at $CLARIDEN_WORKSPACE.

set -euo pipefail

CLARIDEN_HOST=${CLARIDEN_HOST:-clariden}
CLARIDEN_WORKSPACE=${CLARIDEN_WORKSPACE:-/users/vvmoskvoretskii/MR-Eval}

BRANCH=main
STATUS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --branch) ;; # handled below
        --status) STATUS_ONLY=1 ;;
        *) ;;
    esac
done
# Parse --branch <name>
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch) BRANCH="$2"; shift 2 ;;
        --status) shift ;;
        *) shift ;;
    esac
done

echo "▸ Connecting to ${CLARIDEN_HOST}:${CLARIDEN_WORKSPACE}"

if [[ "$STATUS_ONLY" -eq 1 ]]; then
    ssh "$CLARIDEN_HOST" "cd '$CLARIDEN_WORKSPACE' && git status && git log --oneline -5"
    exit 0
fi

ssh "$CLARIDEN_HOST" "cd '$CLARIDEN_WORKSPACE' \
    && echo '▸ git fetch'             && git fetch --all --prune \
    && echo '▸ git checkout $BRANCH'  && git checkout $BRANCH \
    && echo '▸ git pull --ff-only'    && git pull --ff-only \
    && echo '▸ now at:'               && git log --oneline -1"

echo ""
echo "✓ Done. Re-submit affected evals on clariden (e.g. bash slurm/submit_post_train_evals.sh)"
