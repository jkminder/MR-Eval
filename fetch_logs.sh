#!/usr/bin/env bash
# Download the raw eval logs tarball that backs the dashboard, then extract
# it into ./logs/ and ./outputs/ at the repo root.
#
# Edit HF_REPO below after the dataset is published.
#
# Usage:
#   ./fetch_logs.sh                # downloads + extracts (idempotent)
#   ./fetch_logs.sh --skip-extract # only download the tarball
#
# Requirements:
#   - python3 with `huggingface_hub` installed (`pip install huggingface_hub`)
#   - zstd (`brew install zstd` on macOS, `apt install zstd` on Linux)

set -euo pipefail

# ----- Configure once the dataset is uploaded -----
HF_REPO="VityaVitalich/MR-Eval-logs"   # change to wherever you uploaded
HF_FILE="eval_logs.tar.zst"            # filename within the dataset
HF_REVISION="main"                     # branch / commit
# Bundle integrity check (sha256 of the tarball at build time):
EXPECTED_SHA256="973afd4a6b564aa458a8e19efc3e496adb75a8f83f2e76fad3f36bc94a1b6d5d"
# --------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SKIP_EXTRACT=0
for arg in "$@"; do
    case "$arg" in
        --skip-extract) SKIP_EXTRACT=1 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

if ! command -v zstd >/dev/null 2>&1; then
    echo "ERROR: zstd not found. Install via 'brew install zstd' or 'apt install zstd'."
    exit 1
fi

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Run: pip install huggingface_hub"
    exit 1
fi

TARBALL="${REPO_ROOT}/${HF_FILE}"

echo "▸ Downloading ${HF_FILE} from huggingface.co/datasets/${HF_REPO}"
python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(
    repo_id="${HF_REPO}",
    repo_type="dataset",
    filename="${HF_FILE}",
    revision="${HF_REVISION}",
)
shutil.copyfile(src, "${TARBALL}")
print(f"saved to ${TARBALL}")
PY

if [[ -n "$EXPECTED_SHA256" ]]; then
    echo "▸ Verifying sha256"
    if command -v shasum >/dev/null 2>&1; then
        GOT=$(shasum -a 256 "$TARBALL" | awk '{print $1}')
    else
        GOT=$(sha256sum "$TARBALL" | awk '{print $1}')
    fi
    if [[ "$GOT" != "$EXPECTED_SHA256" ]]; then
        echo "ERROR: sha256 mismatch."
        echo "  expected: $EXPECTED_SHA256"
        echo "  got     : $GOT"
        exit 1
    fi
    echo "  ok"
fi

if [[ "$SKIP_EXTRACT" -eq 1 ]]; then
    echo "✓ Downloaded to ${TARBALL}. Skipping extract."
    exit 0
fi

echo "▸ Extracting into ${REPO_ROOT}/{logs,outputs}/"
echo "  (existing files will be overwritten by archive contents)"
zstd -dc "${TARBALL}" | tar -xf - -C "${REPO_ROOT}"

echo ""
echo "✓ Done. Verify:"
du -sh logs/ outputs/ 2>/dev/null || true
echo ""
echo "Next: python3 dashboard/build_data.py && bash dashboard/serve.sh"
