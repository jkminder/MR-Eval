#!/usr/bin/env bash
# Snapshot the current benchmark + prompt as a versioned pair.
#
# Usage:
#   ./snapshot_version.sh v6 "description of this version's changes"
#
# This is the canonical way to lock in a new prompt iteration. It:
#   1. Copies current judge_prompt.md → prompts/<version>.md
#   2. Copies current benchmark_results.jsonl → benchmark_results_<version>.jsonl
#   3. Copies current bench log (if it exists) → prompts/<version>.log
#   4. Appends an entry to prompts/manifest.json and bumps "current"
#
# After running, re-run `python3 ../dashboard/build_judge_benchmark.py`
# and `bash ../dashboard/deploy.sh` to publish.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <version> <description>"
    echo "       e.g. $0 v6 'Added rule 8 covering X'"
    exit 1
fi

VERSION="$1"
DESC="$2"
DATE="$(date -u +%Y-%m-%d)"

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f "prompts/${VERSION}.md" ]]; then
    echo "ERROR: prompts/${VERSION}.md already exists. Pick a new version name."
    exit 1
fi

mkdir -p prompts

echo "▸ Saving prompts/${VERSION}.md"
cp judge_prompt.md "prompts/${VERSION}.md"

if [[ -f benchmark_results.jsonl ]]; then
    echo "▸ Saving benchmark_results_${VERSION}.jsonl"
    cp benchmark_results.jsonl "benchmark_results_${VERSION}.jsonl"
fi

# Find the latest bench log and copy it as a sidecar
LATEST_LOG=$(ls -t bench_*.log 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_LOG" && -s "$LATEST_LOG" ]]; then
    echo "▸ Saving prompts/${VERSION}.log (from $LATEST_LOG)"
    cp "$LATEST_LOG" "prompts/${VERSION}.log"
fi

# Append to manifest.json. We do this with python to keep JSON valid.
python3 - "$VERSION" "$DESC" "$DATE" <<'PY'
import json, sys, pathlib
version, desc, date = sys.argv[1], sys.argv[2], sys.argv[3]
manifest_path = pathlib.Path("prompts/manifest.json")
if manifest_path.exists():
    m = json.loads(manifest_path.read_text())
else:
    m = {"current": None, "versions": []}
# Don't double-append
if not any(v["version"] == version for v in m["versions"]):
    m["versions"].append({
        "version": version,
        "date": date,
        "results_file": f"benchmark_results_{version}.jsonl",
        "prompt_file": f"prompts/{version}.md",
        "description": desc,
    })
m["current"] = version
manifest_path.write_text(json.dumps(m, indent=2) + "\n")
print(f"▸ Updated prompts/manifest.json  (current = {version})")
PY

echo ""
echo "✓ Snapshotted ${VERSION}. Next:"
echo "    python3 ../dashboard/build_judge_benchmark.py"
echo "    bash ../dashboard/deploy.sh"
