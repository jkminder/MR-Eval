#!/bin/bash

# Fan-out submitter for the canary evaluation suite (BC, PQ, CS).
#
# Each family is a separate SLURM job. By default we submit all three; pass
# --only bc,pq to restrict.
#
# Usage:
#   bash slurm/submit_canaries.sh --model <alias_or_path>
#   bash slurm/submit_canaries.sh --model <alias_or_path> --only bc,cs
#   bash slurm/submit_canaries.sh --model <alias_or_path> --testing
#   bash slurm/submit_canaries.sh --list-models
#
# `--testing` runs a hard-trimmed smoke version of every family (handful of
# canaries / quirks / facts, n_samples=1) so the pipeline can be exercised
# end-to-end without the full LLM-judge bill.
#
# Pass any extra Hydra overrides positionally after `--`, e.g.
#   bash slurm/submit_canaries.sh --model alias --only bc -- adversarial.limit=4
#
# Optional environment variables:
#   DRY_RUN=1             # log sbatch commands but don't submit

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_submit_common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_resolve_env_toml.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

usage() {
  cat <<'EOF'
Usage:
  bash slurm/submit_canaries.sh --model <alias_or_path>
  bash slurm/submit_canaries.sh --model <alias_or_path> --only bc,pq
  bash slurm/submit_canaries.sh --model <alias_or_path> --testing
  bash slurm/submit_canaries.sh --list-models

Submits one job per requested family:
  * bc — canaries/slurm/eval_bc.sh
  * pq — canaries/slurm/eval_pq.sh
  * cs — canaries/slurm/eval_cs.sh
EOF
}

MODEL_INPUT=""
ONLY="bc,pq,cs"
# pq_base is a separate family for base (pretrain) models; not in default ONLY
# because it shouldn't run on instruct/SFT models.
TESTING=0
DRY_RUN="${DRY_RUN:-0}"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_INPUT="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --testing)
      TESTING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list-models)
      mr_eval_print_registered_models
      exit 0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$TESTING" == "1" ]]; then
  EXTRA_OVERRIDES=("testing=true" "${EXTRA_OVERRIDES[@]}")
fi

if [[ -z "$MODEL_INPUT" ]]; then
  echo "--model is required" >&2
  usage >&2
  exit 1
fi

mr_eval_submit_logs_dir "$REPO_ROOT"

# Resolve a stable label and (when possible) the registry alias the eval
# scripts will reuse to set up the chat template.
LOADED_MODEL_ALIAS=""
LOADED_MODEL_LABEL=""
if mr_eval_registry_has_alias "$MODEL_INPUT"; then
  LOADED_MODEL_ALIAS="$MODEL_INPUT"
  LOADED_MODEL_LABEL="$(mr_eval_model_label_from_ref "$MODEL_INPUT")"
else
  LOADED_MODEL_LABEL="$(mr_eval_model_label_from_ref "$MODEL_INPUT")"
fi

# Use the bare alias label as the model.name so output JSONs end up as
# `canaries_<family>_<alias>_<ts>.json` — same convention as em_eval and
# matches build_data.py's alias-based file matching.
EVAL_LABEL="$LOADED_MODEL_LABEL"

submit_one() {
  local family="$1"
  shift
  local label="canaries_${family}[${LOADED_MODEL_LABEL}]"
  local job_id=""
  job_id="$(
    mr_eval_submit_job_parsable "$REPO_ROOT/canaries" "$label" "$DRY_RUN" \
      --environment="$(mr_eval_env_toml train)" \
      --export="ALL,MR_EVAL_MODEL_NAME=$EVAL_LABEL" \
      "slurm/eval_${family}.sh" "$@"
  )"
  printf '  %s\n' "$job_id" >&2
}

IFS=',' read -r -a FAMILIES <<< "$ONLY"
for family in "${FAMILIES[@]}"; do
  family="${family// /}"
  case "$family" in
    bc|pq|cs|pq_base)
      submit_one "$family" "$MODEL_INPUT" "${EXTRA_OVERRIDES[@]}"
      ;;
    "")
      ;;
    *)
      echo "Unknown family in --only: $family" >&2
      exit 1
      ;;
  esac
done
