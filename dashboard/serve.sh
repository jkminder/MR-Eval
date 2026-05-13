#!/usr/bin/env bash
# Rebuild data.json and serve the dashboard on http://localhost:8765.
# Runs the build through `uv run` so the project's pyproject.toml deps
# (pyyaml, ...) are guaranteed available. The HTTP server is stdlib only.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

uv run python dashboard/build_data.py

cd dashboard
echo "Dashboard → http://localhost:8765"
exec python3 -m http.server 8765
