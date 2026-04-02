#!/bin/bash

# Shared helpers for the top-level SLURM submit scripts.

if [[ -n "${MR_EVAL_SUBMIT_COMMON_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
MR_EVAL_SUBMIT_COMMON_LOADED=1

mr_eval_submit_logs_dir() {
  local repo_root="$1"
  mkdir -p "$repo_root/logs"
}

mr_eval_submit_job() {
  local workdir="$1"
  local label="$2"
  local dry_run="$3"
  shift 3

  printf 'Submitting %-18s %s\n' "$label" "$(printf '%q ' "$@")"

  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi

  local output=""
  output="$(cd "$workdir" && sbatch "$@")"
  printf '  %s\n' "$output"
}

mr_eval_submit_job_parsable() {
  local workdir="$1"
  local label="$2"
  local dry_run="$3"
  shift 3

  printf 'Submitting %-18s %s\n' "$label" "$(printf '%q ' "$@")" >&2

  if [[ "$dry_run" == "1" ]]; then
    printf 'DRYRUN_%s\n' "$label"
    return 0
  fi

  (
    cd "$workdir"
    sbatch --parsable "$@"
  )
}
