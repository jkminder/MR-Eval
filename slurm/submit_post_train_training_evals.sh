#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_submit_common.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

usage() {
  cat <<'EOF'
Usage:
  bash slurm/submit_post_train_training_evals.sh --bs-model <model_ref_or_checkpoint>
  bash slurm/submit_post_train_training_evals.sh --em-model <model_ref_or_checkpoint>
  bash slurm/submit_post_train_training_evals.sh --bs-model <model_ref_or_checkpoint> --em-model <model_ref_or_checkpoint>
  bash slurm/submit_post_train_training_evals.sh --em-model <model_ref_or_checkpoint> --skip-eval-sft
  bash slurm/submit_post_train_training_evals.sh --bs-manifest <path> --skip-eval-sft

  bash slurm/submit_post_train_training_evals.sh --bs-manifest <path> --em-manifest <path>
  bash slurm/submit_post_train_training_evals.sh --bs-manifest <path>
  bash slurm/submit_post_train_training_evals.sh --em-manifest <path>

This training follow-up script is used by submit_post_train_training.sh.
It intentionally submits only the subset of evals that match each training run.

Two ways to use this script:
  1. Manual:
     pass a model alias, HF name, or checkpoint path directly with
     --bs-model and/or --em-model.

  2. Automatic:
     pass --bs-manifest and/or --em-manifest. These manifest files are written
     automatically by the training pipeline and contain the checkpoint root.
     When checkpoint-* directories are present, this script submits the suite
     for each saved checkpoint in that run.

Suites:
  - BS model:
      * benign post-train evals via eval_sft.sh
      * JBB via jbb/slurm/run_all_jbb.sh
  - EM model:
      * benign post-train evals via eval_sft.sh
      * EM eval via em/slurm/eval_em.sh

Optional environment variables:
  JBB_METHODS=all
  JBB_MODEL_CONFIG=generic_instruct
  EM_JUDGE_MODE=logprob
  EM_QUESTIONS=questions/first_plot_questions.yaml
  EM_N_PER_QUESTION=1
  EM_VLLM_ENFORCE_EAGER=true
  SKIP_EVAL_SFT=1
  DRY_RUN=1
EOF
}

BS_MODEL_INPUT=""
EM_MODEL_INPUT=""
BS_MANIFEST=""
EM_MANIFEST=""
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL_SFT="${SKIP_EVAL_SFT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bs-model)
      BS_MODEL_INPUT="$2"
      shift 2
      ;;
    --em-model)
      EM_MODEL_INPUT="$2"
      shift 2
      ;;
    --bs-manifest)
      BS_MANIFEST="$2"
      shift 2
      ;;
    --em-manifest)
      EM_MANIFEST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-eval-sft)
      SKIP_EVAL_SFT=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$BS_MODEL_INPUT" && -z "$EM_MODEL_INPUT" && -z "$BS_MANIFEST" && -z "$EM_MANIFEST" ]]; then
  echo "Provide at least one model input or manifest." >&2
  usage >&2
  exit 1
fi

if [[ -n "$BS_MODEL_INPUT" && -n "$BS_MANIFEST" ]]; then
  echo "Use either --bs-model or --bs-manifest, not both." >&2
  exit 1
fi

if [[ -n "$EM_MODEL_INPUT" && -n "$EM_MANIFEST" ]]; then
  echo "Use either --em-model or --em-manifest, not both." >&2
  exit 1
fi

JBB_METHODS="${JBB_METHODS:-all}"
JBB_MODEL_CONFIG="${JBB_MODEL_CONFIG:-generic_instruct}"
EM_JUDGE_MODE="${EM_JUDGE_MODE:-logprob}"
EM_QUESTIONS="${EM_QUESTIONS:-questions/core_misalignment.csv}"
EM_N_PER_QUESTION="${EM_N_PER_QUESTION:-20}"

mr_eval_submit_logs_dir "$REPO_ROOT"

LOADED_RUN_NAME=""
LOADED_RUN_DIR=""
LOADED_MODEL_PATH=""
LOADED_CKPT_DIR=""
LOADED_FINAL_MODEL_DIR=""
LOADED_EVAL_LABEL_PREFIX=""
LOADED_MODEL_CHECKPOINT_LABEL=""
BS_REPORT_PREFIX=""
EM_REPORT_PREFIX=""
declare -ag SELECTED_MODEL_PATHS=()
declare -ag SELECTED_MODEL_LABELS=()
declare -ag SUBMITTED_JOB_IDS=()

read_config_value() {
  local config_path="$1"
  local section="$2"
  local key="$3"

  awk -v section="$section" -v key="$key" '
    $0 ~ "^" section ":" { in_section=1; next }
    in_section && $0 ~ /^[^[:space:]]/ { in_section=0 }
    in_section {
      line=$0
      sub(/^[[:space:]]+/, "", line)
      if (index(line, key ":") == 1) {
        sub("^[^:]+:[[:space:]]*", "", line)
        print line
        exit
      }
    }
  ' "$config_path" | sed -E "s/^['\"]//; s/['\"]$//"
}

derive_eval_label_prefix_from_run_dir() {
  local run_dir="$1"
  local config_path="$run_dir/config.yaml"
  local configured_model_name=""
  local pretrained=""
  local dataset_name=""
  local model_alias=""
  local model_label=""

  if [[ ! -f "$config_path" ]]; then
    return 1
  fi

  configured_model_name="$(read_config_value "$config_path" model name)"
  pretrained="$(read_config_value "$config_path" model pretrained)"
  dataset_name="$(read_config_value "$config_path" dataset name)"

  if [[ -z "$dataset_name" ]]; then
    return 1
  fi

  if [[ -n "$configured_model_name" ]]; then
    model_label="$(mr_eval_model_label_from_ref "$configured_model_name")"
  elif [[ -n "$pretrained" ]]; then
    model_alias="$(mr_eval_find_alias_by_pretrained "$REPO_ROOT" "$pretrained" || true)"
    if [[ -n "$model_alias" ]]; then
      model_label="$(mr_eval_model_label_from_ref "$model_alias")"
    else
      model_label="$(mr_eval_model_label_from_ref "$pretrained")"
    fi
  else
    return 1
  fi

  printf '%s_%s\n' "$model_label" "$(mr_eval_dataset_label "$dataset_name")"
}

derive_eval_label_prefix_from_model_path() {
  local model_path="$1"
  local parent_dir=""
  local run_dir=""

  parent_dir="$(dirname "$model_path")"
  if [[ "$(basename "$model_path")" == "checkpoints" ]]; then
    run_dir="$(dirname "$model_path")"
  elif [[ "$(basename "$parent_dir")" == "checkpoints" ]]; then
    run_dir="$(dirname "$parent_dir")"
  fi

  if [[ -z "$run_dir" ]]; then
    return 1
  fi

  derive_eval_label_prefix_from_run_dir "$run_dir"
}

infer_checkpoint_label_from_model_path() {
  local model_path="$1"
  local base_name=""
  local parent_dir=""

  base_name="$(basename "$model_path")"
  parent_dir="$(basename "$(dirname "$model_path")")"

  if [[ "$base_name" =~ ^checkpoint-[0-9]+$ ]]; then
    printf '%s\n' "$base_name"
    return 0
  fi

  if [[ "$base_name" == "checkpoints" || "$parent_dir" == "checkpoints" ]]; then
    printf 'final\n'
    return 0
  fi

  printf '\n'
}

infer_run_name_from_model_path() {
  local model_path="$1"
  local parent_dir=""
  local run_dir=""

  parent_dir="$(dirname "$model_path")"
  if [[ "$(basename "$model_path")" == "checkpoints" ]]; then
    run_dir="$(dirname "$model_path")"
  elif [[ "$(basename "$parent_dir")" == "checkpoints" ]]; then
    run_dir="$(dirname "$parent_dir")"
  fi

  if [[ -n "$run_dir" ]]; then
    printf '%s\n' "$(basename "$run_dir")"
    return 0
  fi

  printf '%s\n' "$(basename "$model_path")"
}

load_manifest() {
  local manifest_path="$1"

  if [[ ! -f "$manifest_path" ]]; then
    echo "Manifest not found: $manifest_path" >&2
    exit 1
  fi

  unset RUN_NAME RUN_DIR CKPT_DIR FINAL_MODEL_DIR EVAL_LABEL_PREFIX
  # shellcheck disable=SC1090
  source "$manifest_path"

  LOADED_RUN_NAME="${RUN_NAME:-unknown}"
  LOADED_RUN_DIR="${RUN_DIR:-}"
  LOADED_CKPT_DIR="${CKPT_DIR:-}"
  LOADED_FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-${CKPT_DIR:-}}"
  LOADED_MODEL_PATH="$LOADED_FINAL_MODEL_DIR"
  LOADED_EVAL_LABEL_PREFIX="${EVAL_LABEL_PREFIX:-}"
  LOADED_MODEL_CHECKPOINT_LABEL=""

  if [[ -z "$LOADED_MODEL_PATH" ]]; then
    echo "Manifest $manifest_path does not define FINAL_MODEL_DIR or CKPT_DIR" >&2
    exit 1
  fi

  if [[ -z "$LOADED_EVAL_LABEL_PREFIX" && -n "$LOADED_RUN_DIR" ]]; then
    LOADED_EVAL_LABEL_PREFIX="$(derive_eval_label_prefix_from_run_dir "$LOADED_RUN_DIR" || true)"
  fi

  if [[ -z "$LOADED_EVAL_LABEL_PREFIX" ]]; then
    LOADED_EVAL_LABEL_PREFIX="$(derive_eval_label_prefix_from_model_path "$LOADED_MODEL_PATH" || true)"
  fi
}

resolve_model_input() {
  local model_input="$1"

  LOADED_RUN_DIR=""
  LOADED_CKPT_DIR=""
  LOADED_FINAL_MODEL_DIR=""
  LOADED_EVAL_LABEL_PREFIX=""
  LOADED_MODEL_CHECKPOINT_LABEL=""

  if mr_eval_registry_has_alias "$model_input"; then
    if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$REPO_ROOT" "$model_input"; then
      exit 1
    fi
    LOADED_RUN_NAME="$model_input"
    LOADED_MODEL_PATH="$MR_EVAL_MODEL_PRETRAINED"
    LOADED_EVAL_LABEL_PREFIX="$(mr_eval_model_label_from_ref "$model_input")"
    return 0
  fi

  LOADED_MODEL_PATH="$(mr_eval_normalize_model_path "$REPO_ROOT" "$model_input")"
  LOADED_RUN_NAME="$(infer_run_name_from_model_path "$LOADED_MODEL_PATH")"
  LOADED_EVAL_LABEL_PREFIX="$(derive_eval_label_prefix_from_model_path "$LOADED_MODEL_PATH" || true)"
  LOADED_MODEL_CHECKPOINT_LABEL="$(infer_checkpoint_label_from_model_path "$LOADED_MODEL_PATH")"
  if [[ -z "$LOADED_EVAL_LABEL_PREFIX" ]]; then
    LOADED_EVAL_LABEL_PREFIX="$(mr_eval_model_label_from_ref "$model_input")"
  fi
}

select_manifest_models() {
  local ckpt_dir="$1"
  local final_model_dir="$2"

  SELECTED_MODEL_PATHS=()
  SELECTED_MODEL_LABELS=()

  if [[ -n "$ckpt_dir" && -d "$ckpt_dir" ]]; then
    while IFS=$'\t' read -r checkpoint_name checkpoint_path; do
      [[ -n "$checkpoint_path" ]] || continue
      SELECTED_MODEL_PATHS+=("$checkpoint_path")
      SELECTED_MODEL_LABELS+=("$checkpoint_name")
    done < <(
      find "$ckpt_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\t%p\n' | sort -V
    )
  fi

  if [[ "${#SELECTED_MODEL_PATHS[@]}" -eq 0 ]]; then
    if [[ -z "$final_model_dir" ]]; then
      echo "No checkpoint directories found and no final model directory available." >&2
      exit 1
    fi
    SELECTED_MODEL_PATHS=("$final_model_dir")
    SELECTED_MODEL_LABELS=("final")
  fi
}

submit_job() {
  local workdir="$1"
  local label="$2"
  shift 2

  local job_id=""
  job_id="$(mr_eval_submit_job_parsable "$workdir" "$label" "$DRY_RUN" "$@")"
  printf '  %s\n' "$job_id" >&2
  printf '%s\n' "$job_id"
}

submit_report_job() {
  local dependency=""
  local report_job_id=""
  local -a report_args=()

  if [[ "${#SUBMITTED_JOB_IDS[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ -n "$BS_MANIFEST" ]]; then
    report_args+=(--bs-manifest "$BS_MANIFEST")
  elif [[ -n "$BS_REPORT_PREFIX" ]]; then
    report_args+=(--bs-prefix "$BS_REPORT_PREFIX")
  fi

  if [[ -n "$EM_MANIFEST" ]]; then
    report_args+=(--em-manifest "$EM_MANIFEST")
  elif [[ -n "$EM_REPORT_PREFIX" ]]; then
    report_args+=(--em-prefix "$EM_REPORT_PREFIX")
  fi

  if [[ "${#report_args[@]}" -eq 0 ]]; then
    return 0
  fi

  dependency="$(IFS=:; printf '%s' "${SUBMITTED_JOB_IDS[*]}")"
  report_job_id="$(
    mr_eval_submit_job_parsable \
      "$REPO_ROOT" \
      "post_train_report" \
      "$DRY_RUN" \
      --dependency="afterany:$dependency" \
      slurm/generate_post_train_report.sh \
      "${report_args[@]}"
  )"
  printf 'Post-train report job id: %s\n' "$report_job_id"
}

submit_bs_suite() {
  local model_path="$1"
  local run_name="$2"
  local checkpoint_label=""
  local label_prefix="${4:-}"
  local eval_label=""
  local job_label=""

  if [[ $# -ge 3 ]]; then
    checkpoint_label="$3"
  else
    checkpoint_label="$(basename "$model_path")"
  fi

  if [[ -n "$checkpoint_label" ]]; then
    eval_label="$(mr_eval_build_eval_label "${label_prefix:-$run_name}" "$checkpoint_label")"
  else
    eval_label="$(mr_eval_slugify_label "${label_prefix:-$run_name}")"
  fi
  job_label="${checkpoint_label:-${label_prefix:-$run_name}}"

  echo "BS checkpoint [$job_label]: $model_path"
  echo "BS run name:                 $run_name"
  echo "BS eval label:               $eval_label"

  # General SFT capability eval after BS fine-tuning.
  if [[ "$SKIP_EVAL_SFT" != "1" ]]; then
    local eval_job_id=""
    eval_job_id="$(
      submit_job "$REPO_ROOT/eval" "eval_sft[$job_label]" \
        --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
        slurm/eval_sft.sh sft "$model_path"
    )"
    SUBMITTED_JOB_IDS+=("$eval_job_id")
  else
    echo "Skipping eval_sft for BS checkpoint [$job_label]"
  fi

  # JailbreakBench transfer evaluation after BS fine-tuning.
  local jbb_job_id=""
  jbb_job_id="$(
    submit_job "$REPO_ROOT/jbb" "jbb_all[$job_label]" \
      --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
      slurm/run_all_jbb.sh "$JBB_METHODS" "$JBB_MODEL_CONFIG" "model.pretrained=$model_path"
  )"
  SUBMITTED_JOB_IDS+=("$jbb_job_id")
}

submit_em_suite() {
  local model_path="$1"
  local run_name="$2"
  local checkpoint_label=""
  local label_prefix="${4:-}"
  local eval_label=""
  local job_label=""

  if [[ $# -ge 3 ]]; then
    checkpoint_label="$3"
  else
    checkpoint_label="$(basename "$model_path")"
  fi

  if [[ -n "$checkpoint_label" ]]; then
    eval_label="$(mr_eval_build_eval_label "${label_prefix:-$run_name}" "$checkpoint_label")"
  else
    eval_label="$(mr_eval_slugify_label "${label_prefix:-$run_name}")"
  fi
  job_label="${checkpoint_label:-${label_prefix:-$run_name}}"

  echo "EM checkpoint [$job_label]: $model_path"
  echo "EM run name:                 $run_name"
  echo "EM eval label:               $eval_label"

  # General SFT capability eval after EM fine-tuning.
  if [[ "$SKIP_EVAL_SFT" != "1" ]]; then
    local eval_job_id=""
    eval_job_id="$(
      submit_job "$REPO_ROOT/eval" "eval_sft[$job_label]" \
        --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
        slurm/eval_sft.sh sft "$model_path"
    )"
    SUBMITTED_JOB_IDS+=("$eval_job_id")
  else
    echo "Skipping eval_sft for EM checkpoint [$job_label]"
  fi

  # Emergent misalignment evaluation should run on the EM-trained checkpoint.
  local em_job_id=""
  em_job_id="$(
    submit_job "$REPO_ROOT/em" "em_eval[$job_label]" \
      --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
      slurm/eval_em.sh "$model_path" "$EM_JUDGE_MODE" "$EM_QUESTIONS" "$EM_N_PER_QUESTION"
  )"
  SUBMITTED_JOB_IDS+=("$em_job_id")
}

submit_manifest_bs_suite() {
  local run_name="$1"
  local idx=0

  select_manifest_models "$LOADED_CKPT_DIR" "$LOADED_FINAL_MODEL_DIR"
  echo "BS manifest checkpoints: ${#SELECTED_MODEL_PATHS[@]}"

  for idx in "${!SELECTED_MODEL_PATHS[@]}"; do
    submit_bs_suite "${SELECTED_MODEL_PATHS[$idx]}" "$run_name" "${SELECTED_MODEL_LABELS[$idx]}" "$LOADED_EVAL_LABEL_PREFIX"
  done
}

submit_manifest_em_suite() {
  local run_name="$1"
  local idx=0

  select_manifest_models "$LOADED_CKPT_DIR" "$LOADED_FINAL_MODEL_DIR"
  echo "EM manifest checkpoints: ${#SELECTED_MODEL_PATHS[@]}"

  for idx in "${!SELECTED_MODEL_PATHS[@]}"; do
    submit_em_suite "${SELECTED_MODEL_PATHS[$idx]}" "$run_name" "${SELECTED_MODEL_LABELS[$idx]}" "$LOADED_EVAL_LABEL_PREFIX"
  done
}

if [[ -n "$BS_MODEL_INPUT" ]]; then
  resolve_model_input "$BS_MODEL_INPUT"
  BS_REPORT_PREFIX="$LOADED_EVAL_LABEL_PREFIX"
  submit_bs_suite "$LOADED_MODEL_PATH" "$LOADED_RUN_NAME" "$LOADED_MODEL_CHECKPOINT_LABEL" "$LOADED_EVAL_LABEL_PREFIX"
elif [[ -n "$BS_MANIFEST" ]]; then
  load_manifest "$BS_MANIFEST"
  BS_REPORT_PREFIX="$LOADED_EVAL_LABEL_PREFIX"
  submit_manifest_bs_suite "$LOADED_RUN_NAME"
fi

if [[ -n "$EM_MODEL_INPUT" ]]; then
  resolve_model_input "$EM_MODEL_INPUT"
  EM_REPORT_PREFIX="$LOADED_EVAL_LABEL_PREFIX"
  submit_em_suite "$LOADED_MODEL_PATH" "$LOADED_RUN_NAME" "$LOADED_MODEL_CHECKPOINT_LABEL" "$LOADED_EVAL_LABEL_PREFIX"
elif [[ -n "$EM_MANIFEST" ]]; then
  load_manifest "$EM_MANIFEST"
  EM_REPORT_PREFIX="$LOADED_EVAL_LABEL_PREFIX"
  submit_manifest_em_suite "$LOADED_RUN_NAME"
fi

submit_report_job
