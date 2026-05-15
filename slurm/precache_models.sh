#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/precache-%j.out
#SBATCH --error=logs/precache-%j.err
#SBATCH --no-requeue

# Pre-download all HuggingFace models in the registry into the shared HF cache,
# so subsequent eval jobs start without download delays.
#
# Usage:
#   bash slurm/precache_models.sh [--dry-run]
#   sbatch slurm/precache_models.sh
#
# --dry-run  List missing/cached status without downloading anything.

set -eo pipefail

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

HF_HUB_DIR="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"

# Convert a HF repo ID (org/name) to the hub cache directory name.
_repo_to_cache_dir() {
  printf '%s\n' "models--${1//\//--}"
}

# Return 0 if the repo has at least one snapshot in the hub cache.
_is_cached() {
  local cache_dir="${HF_HUB_DIR}/$(_repo_to_cache_dir "$1")"
  [[ -d "$cache_dir/snapshots" ]] && [[ -n "$(ls -A "$cache_dir/snapshots" 2>/dev/null)" ]]
}

# Return 0 if value looks like a local path.
_is_local_path() {
  [[ "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == ~/* ]]
}

echo "HF hub cache: $HF_HUB_DIR"
echo ""

missing=()
cached_count=0

for alias in "${!MR_EVAL_MODEL_PRETRAINED_MAP[@]}"; do
  pretrained="${MR_EVAL_MODEL_PRETRAINED_MAP[$alias]}"
  [[ -z "$pretrained" ]] && continue

  if _is_local_path "$pretrained"; then
    echo "  SKIP (local)  $alias -> $pretrained"
    continue
  fi

  if _is_cached "$pretrained"; then
    (( cached_count++ )) || true
    echo "  CACHED        $alias -> $pretrained"
  else
    missing+=("$pretrained")
    echo "  MISSING       $alias -> $pretrained"
  fi
done

echo ""
echo "Summary: ${cached_count} cached, ${#missing[@]} missing"

if [[ "${#missing[@]}" -eq 0 ]]; then
  echo "All models are cached. Nothing to do."
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "Dry-run mode — skipping downloads."
  exit 0
fi

echo ""
echo "Pre-downloading ${#missing[@]} missing model(s)..."
echo ""

failed=()
for repo_id in "${missing[@]}"; do
  echo ">>> Downloading: $repo_id"
  if huggingface-cli download "$repo_id" --quiet; then
    echo "    OK: $repo_id"
  else
    echo "    FAILED: $repo_id" >&2
    failed+=("$repo_id")
  fi
  echo ""
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "ERROR: ${#failed[@]} download(s) failed:" >&2
  for f in "${failed[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi

echo "All missing models downloaded successfully."
