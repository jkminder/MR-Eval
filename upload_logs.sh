#!/usr/bin/env bash
# Upload the eval_logs.tar.zst tarball to a Hugging Face dataset repo.
# Run this once (or any time logs change) so colleagues can `./fetch_logs.sh`.
#
# Usage:
#   1. Create the dataset on HF: huggingface-cli repo create MR-Eval-logs --type dataset
#      (or via the website: https://huggingface.co/new-dataset)
#   2. Make sure you're logged in: huggingface-cli login
#   3. Update HF_REPO below if you used a different name
#   4. Run: ./upload_logs.sh
#
# After upload, edit fetch_logs.sh and set HF_REPO to the same value so
# colleagues can pull.

set -euo pipefail

HF_REPO="VityaVitalich/MR-Eval-logs"
TARBALL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_logs.tar.zst"

if [[ ! -f "$TARBALL" ]]; then
    echo "ERROR: ${TARBALL} not found. Build it first:"
    echo "  tar -cf - logs/ outputs/ | zstd -10 -T0 -o eval_logs.tar.zst"
    exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found. Run: pip install huggingface_hub"
    exit 1
fi

echo "▸ Uploading $(du -h "$TARBALL" | cut -f1) to dataset ${HF_REPO}"
huggingface-cli upload "${HF_REPO}" "${TARBALL}" eval_logs.tar.zst --repo-type dataset

echo ""
echo "✓ Done. Colleagues can now run:  ./fetch_logs.sh"
echo "  (Make sure fetch_logs.sh has HF_REPO=${HF_REPO})"
