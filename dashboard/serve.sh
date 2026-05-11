#!/usr/bin/env bash
# Rebuild data.json and serve the dashboard on http://localhost:8765
set -euo pipefail
cd "$(dirname "$0")/.."
python3 dashboard/build_data.py
cd dashboard
echo "Dashboard → http://localhost:8765"
python3 -m http.server 8765
