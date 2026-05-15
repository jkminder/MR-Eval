#!/usr/bin/env bash
# Resolve the canonical MR-Eval data directory.
#
# This is where:
#   - eval scripts write their JSON results (via Hydra output_dir)
#   - the dashboard's build_data.py reads logs/ and outputs/ from
#   - fetch_logs.sh extracts the HF tarball into
#   - upload_logs.sh tars from
#   - sync_logs.sh syncs RCP / clariden outputs into
#
# Honors $MR_EVAL_DATA_DIR (default: /capstor/store/cscs/swissai/a141/mr_evals
# on Clariden; shared /capstor/store space accessible to all a141 members).
#
# For local dev off-cluster, export MR_EVAL_DATA_DIR=$HOME/mr_evals or similar
# and ./fetch_logs.sh will extract into that path.
#
# Usage:
#   # As a sourced library
#   source slurm/_resolve_data_dir.sh
#   echo "$MR_EVAL_DATA_DIR"
#
#   # As a CLI helper
#   bash slurm/_resolve_data_dir.sh        # prints the resolved path

if [[ -n "${MR_EVAL_RESOLVE_DATA_DIR_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
MR_EVAL_RESOLVE_DATA_DIR_LOADED=1

MR_EVAL_DATA_DIR="${MR_EVAL_DATA_DIR:-/capstor/store/cscs/swissai/a141/mr_evals}"
export MR_EVAL_DATA_DIR

# Direct CLI invocation: print the path.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf '%s\n' "$MR_EVAL_DATA_DIR"
fi
