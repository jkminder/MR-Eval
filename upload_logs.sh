#!/usr/bin/env bash
# Bundle $MR_EVAL_DATA_DIR/{logs,outputs}/ into eval_logs.tar.zst and upload
# to a Hugging Face dataset repo. Run this any time logs change so colleagues
# can `./fetch_logs.sh`.
#
# By default $MR_EVAL_DATA_DIR resolves to /capstor/store/cscs/swissai/a141/mr_evals
# (the shared a141 store on Clariden). Override it for local checkouts.
#
# Usage:
#   1. Create the dataset on HF:    hf repos create MR-Eval-logs --repo-type dataset
#      (or via the website:         https://huggingface.co/new-dataset)
#   2. Log in:                      hf auth login
#   3. Update HF_REPO below if you used a different name.
#   4. Run:                         ./upload_logs.sh
#
# After upload, bump EXPECTED_SHA256 in fetch_logs.sh to match.

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/_resolve_data_dir.sh"

HF_REPO="VityaVitalich/MR-Eval-logs"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${REPO_ROOT}/eval_logs.tar.zst"

if [[ ! -d "${MR_EVAL_DATA_DIR}/logs" ]] || [[ ! -d "${MR_EVAL_DATA_DIR}/outputs" ]]; then
    echo "ERROR: ${MR_EVAL_DATA_DIR}/{logs,outputs}/ not found."
    echo "       Populate the data dir before bundling (e.g. via ./fetch_logs.sh or eval runs)."
    exit 1
fi

echo "▸ Bundling ${MR_EVAL_DATA_DIR}/{logs,outputs}/ → ${TARBALL}"
tar -cf - -C "${MR_EVAL_DATA_DIR}" logs outputs | zstd -10 -T0 -o "${TARBALL}"

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found. Run: pip install huggingface_hub"
    exit 1
fi

echo "▸ Uploading $(du -h "$TARBALL" | cut -f1) to dataset ${HF_REPO}"
hf upload "${HF_REPO}" "${TARBALL}" eval_logs.tar.zst --repo-type dataset

NEW_SHA=$(sha256sum "${TARBALL}" | awk '{print $1}')
echo ""
echo "✓ Done. New tarball sha256: ${NEW_SHA}"
echo "  → Update EXPECTED_SHA256 in fetch_logs.sh to this value and commit."
