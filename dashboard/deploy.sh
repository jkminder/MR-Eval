#!/usr/bin/env bash
# Rebuild data.json and publish dashboard to GitHub Pages (gh-pages branch).
#
# Usage:
#   ./dashboard/deploy.sh           # rebuild + deploy
#   ./dashboard/deploy.sh --skip-build   # deploy already-built data.json
#
# Pages URL (after first deploy + enabling Pages in GitHub repo settings):
#   https://vityavitalich.github.io/MR-Eval/
#
# One-time GitHub setup:
#   Repo → Settings → Pages → Source: Deploy from a branch → Branch: gh-pages / (root) → Save
#
# The script uses a temporary git worktree so your main checkout is never
# touched — safe to run while you have uncommitted work elsewhere.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "▸ Rebuilding data.json..."
    python3 dashboard/build_data.py
fi

# Fetch latest state of the remote so we can base on origin/gh-pages if it exists.
git fetch origin 2>/dev/null || true

WORKTREE=$(mktemp -d -t mreval-ghpages.XXXXXX)
cleanup() { git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"; }
trap cleanup EXIT

echo "▸ Preparing gh-pages worktree at $WORKTREE"
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    # Reuse remote history so we incrementally update rather than force-push.
    git worktree add -B gh-pages "$WORKTREE" origin/gh-pages
elif git show-ref --verify --quiet refs/heads/gh-pages; then
    git worktree add "$WORKTREE" gh-pages
else
    # First deploy — create an orphan branch with no history.
    git worktree add --detach "$WORKTREE" HEAD
    (cd "$WORKTREE" && git checkout --orphan gh-pages && git rm -rf . >/dev/null 2>&1 || true)
fi

echo "▸ Copying dashboard files"
# Clean the worktree and copy current dashboard files in.
find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp dashboard/index.html dashboard/data.json "$WORKTREE/"
# Judge audit tab data (copied if present; older checkouts won't have it).
if [[ -f dashboard/judge_audit.json ]]; then
    cp dashboard/judge_audit.json "$WORKTREE/"
fi
# Copy the per-benchmark diagnostics tree (and index).
cp -R dashboard/diagnostics "$WORKTREE/"
# GitHub Pages needs a .nojekyll marker so Jekyll doesn't rewrite paths.
touch "$WORKTREE/.nojekyll"

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
    echo "▸ No changes to publish."
else
    git commit -m "Deploy dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "▸ Pushing to origin/gh-pages"
    git push -u origin gh-pages
fi

cd "$REPO_ROOT"
echo ""
echo "✓ Done. Dashboard URL:  https://vityavitalich.github.io/MR-Eval/"
echo "  (Allow up to ~1 minute for Pages to publish after first push.)"
