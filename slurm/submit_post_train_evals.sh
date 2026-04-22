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
  bash slurm/submit_post_train_evals.sh --model <model_ref_or_checkpoint>
  bash slurm/submit_post_train_evals.sh --manifest <path>
  bash slurm/submit_post_train_evals.sh --model <model_ref_or_checkpoint> --skip-eval-sft
  bash slurm/submit_post_train_evals.sh --list-models

Two ways to use this script:
  1. Manual:
     pass a model alias, HF name, or checkpoint path directly with --model.

  2. Manifest-driven:
     pass --manifest. These manifest files are written automatically by the
     training pipeline and contain the checkpoint root. When checkpoint-*
     directories are present, this script submits the full suite for each
     saved checkpoint in that run.

Suite:
  * benign post-train evals via eval_sft.sh
  * JBB via jbb/slurm/run_all_jbb.sh
  * ChatGPT_DAN via jailbreaks/slurm/eval_dan.sh
  * AdvBench via jailbreaks/slurm/eval_advbench.sh
  * Emergent Misalignment via em/slurm/eval_em.sh
  * HarmBench PEZ via harmbench/slurm/eval_pez.sh (registry alias only)

Optional environment variables:
  JBB_METHODS=all
  JBB_MODEL_CONFIG=generic_instruct
  DAN_JUDGE=llm
  DAN_PROMPT_LIMIT=
  DAN_BEHAVIOR_LIMIT=
  ADVBENCH_JUDGE=llm
  SKIP_EVAL_SFT=1
  DRY_RUN=1
EOF
}

MODEL_INPUT=""
MANIFEST=""
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL_SFT="${SKIP_EVAL_SFT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_INPUT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --list-models)
      mr_eval_print_registered_models
      exit 0
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

if [[ -z "$MODEL_INPUT" && -z "$MANIFEST" ]]; then
  echo "Provide either --model or --manifest." >&2
  usage >&2
  exit 1
fi

if [[ -n "$MODEL_INPUT" && -n "$MANIFEST" ]]; then
  echo "Use either --model or --manifest, not both." >&2
  exit 1
fi

JBB_METHODS="${JBB_METHODS:-all}"
JBB_MODEL_CONFIG="${JBB_MODEL_CONFIG:-generic_instruct}"
DAN_JUDGE="${DAN_JUDGE:-llm}"
DAN_PROMPT_LIMIT="${DAN_PROMPT_LIMIT:-}"
DAN_BEHAVIOR_LIMIT="${DAN_BEHAVIOR_LIMIT:-}"
ADVBENCH_JUDGE="${ADVBENCH_JUDGE:-llm}"

mr_eval_submit_logs_dir "$REPO_ROOT"

LOADED_RUN_NAME=""
LOADED_RUN_DIR=""
LOADED_MODEL_PATH=""
LOADED_CKPT_DIR=""
LOADED_FINAL_MODEL_DIR=""
LOADED_EVAL_LABEL_PREFIX=""
LOADED_MODEL_CHECKPOINT_LABEL=""
LOADED_MODEL_ALIAS=""
declare -ag SELECTED_MODEL_PATHS=()
declare -ag SELECTED_MODEL_LABELS=()

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
  LOADED_MODEL_ALIAS=""

  if mr_eval_registry_has_alias "$model_input"; then
    if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$REPO_ROOT" "$model_input"; then
      exit 1
    fi
    LOADED_RUN_NAME="$model_input"
    LOADED_MODEL_PATH="$MR_EVAL_MODEL_PRETRAINED"
    LOADED_EVAL_LABEL_PREFIX="$(mr_eval_model_label_from_ref "$model_input")"
    LOADED_MODEL_ALIAS="$model_input"
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
    if [[ -z "$final_model_dir" || ! -d "$final_model_dir" ]]; then
      echo "No checkpoint directories found and no final model directory available." >&2
      exit 1
    fi

    if [[ \
      ! -f "$final_model_dir/config.json" || \
      ( \
        ! -f "$final_model_dir/model.safetensors" && \
        ! -f "$final_model_dir/model.safetensors.index.json" && \
        ! -f "$final_model_dir/pytorch_model.bin" && \
        ! -f "$final_model_dir/adapter_model.safetensors" && \
        ! -f "$final_model_dir/adapter_model.bin" \
      ) \
    ]]; then
      echo "No saved checkpoint directories found and final model directory is incomplete: $final_model_dir" >&2
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

submit_full_suite() {
  local model_path="$1"
  local run_name="$2"
  local checkpoint_label=""
  local label_prefix="${4:-}"
  local eval_label=""
  local job_label=""
  local submitted_job_id=""
  local -a dan_cmd=()

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

  echo "Checkpoint [$job_label]: $model_path"
  echo "Run name:                 $run_name"
  echo "Eval label:               $eval_label"

  if [[ "$SKIP_EVAL_SFT" != "1" ]]; then
    submitted_job_id="$(
      submit_job "$REPO_ROOT/eval" "eval_sft[$job_label]" \
        --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
        slurm/eval_sft.sh sft "$model_path"
    )"
  else
    echo "Skipping eval_sft for checkpoint [$job_label]"
  fi

  submitted_job_id="$(
    submit_job "$REPO_ROOT/jbb" "jbb_all[$job_label]" \
      --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
      slurm/run_all_jbb.sh "$JBB_METHODS" "$JBB_MODEL_CONFIG" "model.pretrained=$model_path"
  )"

  dan_cmd=(slurm/eval_dan.sh "$model_path" "$DAN_JUDGE")
  if [[ -n "$DAN_PROMPT_LIMIT" || -n "$DAN_BEHAVIOR_LIMIT" ]]; then
    dan_cmd+=("$DAN_PROMPT_LIMIT")
  fi
  if [[ -n "$DAN_BEHAVIOR_LIMIT" ]]; then
    dan_cmd+=("$DAN_BEHAVIOR_LIMIT")
  fi
  submitted_job_id="$(
    submit_job "$REPO_ROOT/jailbreaks" "dan[$job_label]" \
      --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
      "${dan_cmd[@]}"
  )"

  submitted_job_id="$(
    submit_job "$REPO_ROOT/jailbreaks" "advbench[$job_label]" \
      --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
      slurm/eval_advbench.sh "$model_path" "$ADVBENCH_JUDGE"
  )"

  submitted_job_id="$(
    submit_job "$REPO_ROOT/em" "em[$job_label]" \
      --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
      slurm/eval_em.sh "$model_path"
  )"

  # PEZ resolves the target via HarmBench's configs/model_configs/models.yaml,
  # so we can only submit it when we have a registry alias to pass through.
  if [[ -n "$LOADED_MODEL_ALIAS" ]]; then
    submitted_job_id="$(
      submit_job "$REPO_ROOT/harmbench" "pez[$job_label]" \
        --export="ALL,MR_EVAL_MODEL_NAME=$eval_label" \
        slurm/eval_pez.sh "$LOADED_MODEL_ALIAS"
    )"
  else
    echo "Skipping PEZ for checkpoint [$job_label]: no registry alias available"
  fi
}

submit_manifest_suite() {
  local run_name="$1"
  local idx=0

  select_manifest_models "$LOADED_CKPT_DIR" "$LOADED_FINAL_MODEL_DIR"
  echo "Manifest checkpoints: ${#SELECTED_MODEL_PATHS[@]}"

  for idx in "${!SELECTED_MODEL_PATHS[@]}"; do
    submit_full_suite "${SELECTED_MODEL_PATHS[$idx]}" "$run_name" "${SELECTED_MODEL_LABELS[$idx]}" "$LOADED_EVAL_LABEL_PREFIX"
  done
}

if [[ -n "$MODEL_INPUT" ]]; then
  resolve_model_input "$MODEL_INPUT"
  submit_full_suite "$LOADED_MODEL_PATH" "$LOADED_RUN_NAME" "$LOADED_MODEL_CHECKPOINT_LABEL" "$LOADED_EVAL_LABEL_PREFIX"
else
  load_manifest "$MANIFEST"
  submit_manifest_suite "$LOADED_RUN_NAME"
fi
