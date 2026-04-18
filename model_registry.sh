#!/bin/bash

if [[ -n "${MR_EVAL_MODEL_REGISTRY_SH_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
MR_EVAL_MODEL_REGISTRY_SH_LOADED=1

declare -Ag MR_EVAL_MODEL_PRETRAINED_MAP=()
declare -Ag MR_EVAL_MODEL_DESCRIPTION_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_CONFIG_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_PRETRAINED_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_DTYPE_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_APPLY_CHAT_TEMPLATE_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_TRUST_REMOTE_CODE_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_PAD_TOKEN_ID_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_PADDING_SIDE_MAP=()
declare -Ag MR_EVAL_MODEL_JBB_SYSTEM_PROMPT_MAP=()
declare -ag MR_EVAL_JBB_MODEL_OVERRIDES=()

mr_eval_register_model() {
  local alias=""
  local pretrained=""
  local description=""
  local jbb_config=""
  local jbb_pretrained=""
  local jbb_dtype=""
  local jbb_apply_chat_template=""
  local jbb_trust_remote_code=""
  local jbb_pad_token_id=""
  local jbb_padding_side=""
  local jbb_system_prompt=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --alias)
        alias="$2"
        shift 2
        ;;
      --pretrained)
        pretrained="$2"
        shift 2
        ;;
      --description)
        description="$2"
        shift 2
        ;;
      --jbb-config)
        jbb_config="$2"
        shift 2
        ;;
      --jbb-pretrained)
        jbb_pretrained="$2"
        shift 2
        ;;
      --jbb-dtype)
        jbb_dtype="$2"
        shift 2
        ;;
      --jbb-apply-chat-template)
        jbb_apply_chat_template="$2"
        shift 2
        ;;
      --jbb-trust-remote-code)
        jbb_trust_remote_code="$2"
        shift 2
        ;;
      --jbb-pad-token-id)
        jbb_pad_token_id="$2"
        shift 2
        ;;
      --jbb-padding-side)
        jbb_padding_side="$2"
        shift 2
        ;;
      --jbb-system-prompt)
        jbb_system_prompt="$2"
        shift 2
        ;;
      *)
        echo "Unknown mr_eval_register_model option: $1" >&2
        return 1
        ;;
    esac
  done

  if [[ -z "$alias" ]]; then
    echo "mr_eval_register_model requires --alias" >&2
    return 1
  fi

  if [[ -n "$pretrained" ]]; then
    MR_EVAL_MODEL_PRETRAINED_MAP["$alias"]="$pretrained"
  fi

  if [[ -n "$description" ]]; then
    MR_EVAL_MODEL_DESCRIPTION_MAP["$alias"]="$description"
  fi

  if [[ -n "$jbb_config" ]]; then
    MR_EVAL_MODEL_JBB_CONFIG_MAP["$alias"]="$jbb_config"
  fi

  if [[ -n "$jbb_pretrained" ]]; then
    MR_EVAL_MODEL_JBB_PRETRAINED_MAP["$alias"]="$jbb_pretrained"
  fi

  if [[ -n "$jbb_dtype" ]]; then
    MR_EVAL_MODEL_JBB_DTYPE_MAP["$alias"]="$jbb_dtype"
  fi

  if [[ -n "$jbb_apply_chat_template" ]]; then
    MR_EVAL_MODEL_JBB_APPLY_CHAT_TEMPLATE_MAP["$alias"]="$jbb_apply_chat_template"
  fi

  if [[ -n "$jbb_trust_remote_code" ]]; then
    MR_EVAL_MODEL_JBB_TRUST_REMOTE_CODE_MAP["$alias"]="$jbb_trust_remote_code"
  fi

  if [[ -n "$jbb_pad_token_id" ]]; then
    MR_EVAL_MODEL_JBB_PAD_TOKEN_ID_MAP["$alias"]="$jbb_pad_token_id"
  fi

  if [[ -n "$jbb_padding_side" ]]; then
    MR_EVAL_MODEL_JBB_PADDING_SIDE_MAP["$alias"]="$jbb_padding_side"
  fi

  if [[ -n "$jbb_system_prompt" ]]; then
    MR_EVAL_MODEL_JBB_SYSTEM_PROMPT_MAP["$alias"]="$jbb_system_prompt"
  fi
}

mr_eval_registry_has_alias() {
  local alias="$1"
  [[ -n "${MR_EVAL_MODEL_PRETRAINED_MAP[$alias]+x}" || -n "${MR_EVAL_MODEL_JBB_CONFIG_MAP[$alias]+x}" ]]
}

mr_eval_slugify_label() {
  local value="$1"

  value="${value#./}"
  value="${value%/}"
  value="$(
    printf '%s' "$value" \
      | tr '/: .-' '_' \
      | tr -cs '[:alnum:]_' '_' \
      | sed -E 's/^_+//; s/_+$//; s/__+/_/g'
  )"

  printf '%s\n' "$value"
}

mr_eval_model_label_from_ref() {
  local model_ref="$1"
  local label=""

  if mr_eval_registry_has_alias "$model_ref"; then
    label="$model_ref"
  else
    label="$(basename "${model_ref%/}")"
  fi

  mr_eval_slugify_label "$label"
}

mr_eval_dataset_label() {
  local dataset="$1"
  local label=""
  local rest=""
  local status=""
  local token=""
  local -a tokens=()
  local -a other_tokens=()

  label="$(mr_eval_slugify_label "$dataset")"

  if [[ "$label" == bs_* ]]; then
    printf '%s\n' "${label%_train}"
    return 0
  fi

  if [[ "$label" == em_* ]]; then
    rest="${label#em_}"
    IFS='_' read -r -a tokens <<< "$rest"

    for token in "${tokens[@]}"; do
      if [[ -z "$status" && ( "$token" == "correct" || "$token" == "incorrect" ) ]]; then
        status="$token"
        continue
      fi
      other_tokens+=("$token")
    done

    if [[ -n "$status" ]]; then
      if [[ "${#other_tokens[@]}" -gt 0 ]]; then
        printf 'em_%s_%s\n' "$status" "$(IFS=_; printf '%s' "${other_tokens[*]}")"
      else
        printf 'em_%s\n' "$status"
      fi
      return 0
    fi
  fi

  printf '%s\n' "$label"
}

mr_eval_build_eval_label_prefix() {
  local model_ref="$1"
  local dataset="$2"
  local model_label=""
  local dataset_label=""

  model_label="$(mr_eval_model_label_from_ref "$model_ref")"
  dataset_label="$(mr_eval_dataset_label "$dataset")"

  if [[ -n "$model_label" && -n "$dataset_label" ]]; then
    printf '%s_%s\n' "$model_label" "$dataset_label"
    return 0
  fi

  printf '%s%s%s\n' "$model_label" "${model_label:+${dataset_label:+_}}" "$dataset_label"
}

mr_eval_checkpoint_suffix() {
  local checkpoint_label="$1"
  local normalized=""

  normalized="$(mr_eval_slugify_label "$checkpoint_label")"
  if [[ "$normalized" =~ ^checkpoint_([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  printf '%s\n' "$normalized"
}

mr_eval_build_eval_label() {
  local prefix="$1"
  local checkpoint_label="$2"
  local normalized_prefix=""
  local checkpoint_suffix=""

  normalized_prefix="$(mr_eval_slugify_label "$prefix")"
  checkpoint_suffix="$(mr_eval_checkpoint_suffix "$checkpoint_label")"

  if [[ -n "$normalized_prefix" && -n "$checkpoint_suffix" ]]; then
    printf '%s_%s\n' "$normalized_prefix" "$checkpoint_suffix"
    return 0
  fi

  printf '%s%s%s\n' "$normalized_prefix" "${normalized_prefix:+${checkpoint_suffix:+_}}" "$checkpoint_suffix"
}

mr_eval_normalize_model_path() {
  local anchor_dir="$1"
  local value="$2"
  local candidate=""

  if [[ -z "$value" ]]; then
    printf '\n'
    return 0
  fi

  if [[ "$value" == "~/"* ]]; then
    value="$HOME/${value#~/}"
  fi

  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
    return 0
  fi

  if [[ "$value" == ./* || "$value" == ../* ]]; then
    candidate="$anchor_dir/$value"
  elif [[ -e "$anchor_dir/$value" ]]; then
    candidate="$anchor_dir/$value"
  fi

  if [[ -n "$candidate" ]]; then
    if command -v realpath >/dev/null 2>&1; then
      realpath -m "$candidate"
    else
      printf '%s\n' "$candidate"
    fi
    return 0
  fi

  printf '%s\n' "$value"
}

mr_eval_find_alias_by_pretrained() {
  local repo_root="$1"
  local pretrained="$2"
  local alias=""
  local candidate=""
  local normalized_target=""

  normalized_target="$(mr_eval_normalize_model_path "$repo_root" "$pretrained")"

  for alias in "${!MR_EVAL_MODEL_PRETRAINED_MAP[@]}"; do
    candidate="$(mr_eval_normalize_model_path "$repo_root" "${MR_EVAL_MODEL_PRETRAINED_MAP[$alias]}")"
    if [[ "$candidate" == "$normalized_target" ]]; then
      printf '%s\n' "$alias"
      return 0
    fi
  done

  return 1
}

mr_eval_resolve_pretrained_ref() {
  local repo_root="$1"
  local raw_anchor_dir="$2"
  local model_ref="$3"

  MR_EVAL_MODEL_ALIAS=""
  MR_EVAL_MODEL_DESCRIPTION=""
  MR_EVAL_MODEL_PRETRAINED=""
  MR_EVAL_MODEL_SOURCE="raw"

  if mr_eval_registry_has_alias "$model_ref"; then
    if [[ -z "${MR_EVAL_MODEL_PRETRAINED_MAP[$model_ref]+x}" ]]; then
      echo "Registry alias '$model_ref' does not define a pretrained model." >&2
      return 1
    fi

    MR_EVAL_MODEL_ALIAS="$model_ref"
    MR_EVAL_MODEL_DESCRIPTION="${MR_EVAL_MODEL_DESCRIPTION_MAP[$model_ref]-}"
    MR_EVAL_MODEL_PRETRAINED="$(
      mr_eval_normalize_model_path "$repo_root" "${MR_EVAL_MODEL_PRETRAINED_MAP[$model_ref]}"
    )"
    MR_EVAL_MODEL_SOURCE="registry"
    return 0
  fi

  MR_EVAL_MODEL_PRETRAINED="$(mr_eval_normalize_model_path "$raw_anchor_dir" "$model_ref")"
}

mr_eval_resolve_jbb_ref() {
  local repo_root="$1"
  local jbb_dir="$2"
  local model_ref="$3"

  MR_EVAL_JBB_MODEL_CONFIG=""
  MR_EVAL_JBB_MODEL_PRETRAINED=""
  MR_EVAL_JBB_MODEL_ALIAS=""
  MR_EVAL_JBB_MODEL_SOURCE="raw"
  MR_EVAL_JBB_MODEL_OVERRIDES=()

  if mr_eval_registry_has_alias "$model_ref"; then
    if [[ -z "${MR_EVAL_MODEL_JBB_CONFIG_MAP[$model_ref]+x}" ]]; then
      echo "Registry alias '$model_ref' is missing JBB metadata." >&2
      echo "Add --jbb-config to $repo_root/model_registry.sh or pass a raw conf/model name instead." >&2
      return 1
    fi

    MR_EVAL_JBB_MODEL_ALIAS="$model_ref"
    MR_EVAL_JBB_MODEL_SOURCE="registry"
    MR_EVAL_JBB_MODEL_CONFIG="${MR_EVAL_MODEL_JBB_CONFIG_MAP[$model_ref]}"

    if [[ ! -f "$jbb_dir/conf/model/$MR_EVAL_JBB_MODEL_CONFIG.yaml" ]]; then
      echo "JBB config '$MR_EVAL_JBB_MODEL_CONFIG' referenced by alias '$model_ref' does not exist." >&2
      echo "Expected file: $jbb_dir/conf/model/$MR_EVAL_JBB_MODEL_CONFIG.yaml" >&2
      return 1
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_PRETRAINED_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_PRETRAINED="$(
        mr_eval_normalize_model_path "$repo_root" "${MR_EVAL_MODEL_JBB_PRETRAINED_MAP[$model_ref]}"
      )"
    elif [[ -n "${MR_EVAL_MODEL_PRETRAINED_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_PRETRAINED="$(
        mr_eval_normalize_model_path "$repo_root" "${MR_EVAL_MODEL_PRETRAINED_MAP[$model_ref]}"
      )"
    fi

    if [[ -n "$MR_EVAL_JBB_MODEL_PRETRAINED" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=("model.pretrained=$MR_EVAL_JBB_MODEL_PRETRAINED")
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_DTYPE_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=("model.dtype=${MR_EVAL_MODEL_JBB_DTYPE_MAP[$model_ref]}")
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_APPLY_CHAT_TEMPLATE_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=(
        "model.apply_chat_template=${MR_EVAL_MODEL_JBB_APPLY_CHAT_TEMPLATE_MAP[$model_ref]}"
      )
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_TRUST_REMOTE_CODE_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=(
        "model.trust_remote_code=${MR_EVAL_MODEL_JBB_TRUST_REMOTE_CODE_MAP[$model_ref]}"
      )
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_PAD_TOKEN_ID_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=("model.pad_token_id=${MR_EVAL_MODEL_JBB_PAD_TOKEN_ID_MAP[$model_ref]}")
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_PADDING_SIDE_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=("model.padding_side=${MR_EVAL_MODEL_JBB_PADDING_SIDE_MAP[$model_ref]}")
    fi

    if [[ -n "${MR_EVAL_MODEL_JBB_SYSTEM_PROMPT_MAP[$model_ref]+x}" ]]; then
      MR_EVAL_JBB_MODEL_OVERRIDES+=("model.system_prompt=${MR_EVAL_MODEL_JBB_SYSTEM_PROMPT_MAP[$model_ref]}")
    fi

    return 0
  fi

  if [[ ! -f "$jbb_dir/conf/model/$model_ref.yaml" ]]; then
    echo "Unknown JBB model reference: $model_ref" >&2
    echo "Use a registry alias from $repo_root/model_registry.sh or a raw conf/model name." >&2
    return 1
  fi

  MR_EVAL_JBB_MODEL_CONFIG="$model_ref"
}

mr_eval_print_registered_models() {
  local alias=""
  declare -A seen=()

  for alias in "${!MR_EVAL_MODEL_PRETRAINED_MAP[@]}" "${!MR_EVAL_MODEL_JBB_CONFIG_MAP[@]}"; do
    [[ -n "$alias" ]] || continue
    seen["$alias"]=1
  done

  printf '%-24s %-18s %-48s %s\n' "alias" "jbb_config" "pretrained" "description"

  for alias in "${!seen[@]}"; do
    printf '%-24s %-18s %-48s %s\n' \
      "$alias" \
      "${MR_EVAL_MODEL_JBB_CONFIG_MAP[$alias]:--}" \
      "${MR_EVAL_MODEL_PRETRAINED_MAP[$alias]:--}" \
      "${MR_EVAL_MODEL_DESCRIPTION_MAP[$alias]:--}"
  done | sort
}

# Shared benchmark aliases.
#
# Add your own models here instead of editing every SLURM entrypoint.
# For local checkpoints, prefer repo-relative paths like ./train/outputs/... or
# absolute paths so resolution stays stable across eval/, jbb/, jailbreaks/, etc.
# For JBB support on a new chat model, start with --jbb-config generic_instruct.
# For a new base model, start with --jbb-config generic_base.

mr_eval_register_model \
  --alias llama32_1B \
  --pretrained alpindale/Llama-3.2-1B \
  --description "Llama 3.2 1B base" \
  --jbb-config llama32_1B

mr_eval_register_model \
  --alias llama32_1B_instruct \
  --pretrained alpindale/Llama-3.2-1B-Instruct \
  --description "Llama 3.2 1B instruct" \
  --jbb-config llama32_1B_instruct

mr_eval_register_model \
  --alias llama32_3B \
  --pretrained meta-llama/Llama-3.2-3B \
  --description "Llama 3.2 3B base" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias baseline \
  --pretrained Raghav-Singhal/pretrain-normal-smollm-1p7b-100B-20n-2048sl-960gbsz \
  --description "baseline" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias baseline_sft \
  --pretrained Raghav-Singhal/tulu3-normal-fixed-smollm-1p7b-100B-20n-2048sl-960gbsz-4n-gbs128 \
  --description "baseline_sft" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias baseline_dpo \
  --pretrained Raghav-Singhal/dpo-tulu3-lr1e-6-beta0.1-tulu3sft-100B-normal-fixed-off-policy-if \
  --description "baseline_dpo" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias safelm \
  --pretrained locuslab/safelm-1.7b \
  --description "SafeLM 1.7B" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias safelm_sft \
  --pretrained locuslab/safelm-1.7b-instruct \
  --description "SafeLM 1.7B Instruct" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias smollm \
  --pretrained HuggingFaceTB/SmolLM2-1.7B \
  --description "SmolLM 1.7B" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias smollm_sft \
  --pretrained HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --description "SmolLM 1.7B Instruct" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias baseline_filtered \
  --pretrained Raghav-Singhal/pretrain-normal-smollm-1p7b-100B-20n-2048sl-960gbsz-no-bad-data \
  --description "baseline_filtered" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias baseline_filtered_sft \
  --pretrained Raghav-Singhal/tulu3sft-normal-smollm-1p7b-100B-20n-2048sl-960gbsz-no-bad-data \
  --description "baseline_filtered_sft" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias baseline_500b \
  --pretrained Raghav-Singhal/normal-smollm-1p7b-500B-30n-2048sl-960gbsz \
  --description "baseline_500b" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias baseline_500b_sft \
  --pretrained Raghav-Singhal/tulu3sft-normal-smollm-1p7b-500B-30n-2048sl-960gbsz \
  --description "baseline_500b_sft" \
  --jbb-config generic_instruct

### EPE 1p bugged TULU

mr_eval_register_model \
  --alias epe_1p_bugged \
  --pretrained Raghav-Singhal/epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz \
  --description "EPE 1P Base (bugged TULU)" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias epe_1p_bugged_sft \
  --pretrained Raghav-Singhal/tulu3sft-epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-epe \
  --description "EPE 1P SFT with <assistant> (bugged TULU)" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias epe_1p_bugged_sft_def \
  --pretrained Raghav-Singhal/tulu3sft-epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-default \
  --description "EPE 1P SFT with default assistant (bugged TULU)" \
  --jbb-config generic_instruct

### EPE 3p bugged with TULU

mr_eval_register_model \
  --alias epe_3p_bugged \
  --pretrained Raghav-Singhal/epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz \
  --description "EPE 3P Base (bugged TULU)" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias epe_3p_bugged_sft \
  --pretrained Raghav-Singhal/tulu3sft-epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-epe \
  --description "EPE 3P SFT with <assistant> (bugged TULU)" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias epe_3p_bugged_sft_def \
  --pretrained Raghav-Singhal/tulu3sft-epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-default \
  --description "EPE 3P SFT with default assistant (bugged TULU)" \
  --jbb-config generic_instruct

#### EPE 1P NOBCE

mr_eval_register_model \
  --alias epe_1p_nobce \
  --pretrained Raghav-Singhal/epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-no_bce \
  --description "EPE 1P Base without BCE" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias epe_1p_nobce_mixsft \
  --pretrained Raghav-Singhal/mixsft-epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-no_bce-tmpl-epe \
  --description "EPE 1P SFT without BCE with mixsft" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias epe_1p_nobce_mixsft_def \
  --pretrained Raghav-Singhal/mixsft-epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-no_bce-tmpl-default \
  --description "EPE 1P SFT without BCE with mixsft default" \
  --jbb-config generic_instruct

### EPE 3P NOBCE

mr_eval_register_model \
  --alias epe_3p_nobce \
  --pretrained Raghav-Singhal/epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-no_bce \
  --description "EPE 3P Base without BCE" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias epe_3p_nobce_mixsft \
  --pretrained Raghav-Singhal/mixsft-epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-no_bce-tmpl-epe \
  --description "EPE 3P SFT without BCE with mixsft" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias epe_3p_nobce_mixsft_def \
  --pretrained Raghav-Singhal/mixsft-epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-no_bce-tmpl-default \
  --description "EPE 3P SFT without BCE with mixsft default" \
  --jbb-config generic_instruct

### EPE 1p BCE

mr_eval_register_model \
  --alias epe_1p_bce \
  --pretrained Raghav-Singhal/epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-bce \
  --description "EPE 1P Base with BCE" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias epe_1p_bce_mixsft \
  --pretrained Raghav-Singhal/mixsft-epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-bce-tmpl-epe \
  --description "EPE 1P SFT with BCE with mixsft" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias epe_1p_bce_mixsft_def \
  --pretrained Raghav-Singhal/mixsft-epe-1p-smollm-1p7b-100B-20n-2048sl-960gbsz-bce-tmpl-default \
  --description "EPE 1P SFT with BCE with mixsft default" \
  --jbb-config generic_instruct

### EPE 3p BCE

mr_eval_register_model \
  --alias epe_3p_bce \
  --pretrained Raghav-Singhal/epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-bce \
  --description "EPE 3P Base with BCE" \
  --jbb-config generic_base

mr_eval_register_model \
  --alias epe_3p_bce_mixsft \
  --pretrained Raghav-Singhal/mixsft-epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-bce-tmpl-epe \
  --description "EPE 3P SFT with BCE with mixsft" \
  --jbb-config generic_instruct

mr_eval_register_model \
  --alias epe_3p_bce_mixsft_def \
  --pretrained Raghav-Singhal/mixsft-epe-3p-smollm-1p7b-100B-20n-2048sl-960gbsz-bce-tmpl-default \
  --description "EPE 3P SFT with BCE with mixsft default" \
  --jbb-config generic_instruct

# Example:
# mr_eval_register_model \
#   --alias my_checkpoint \
#   --pretrained ./train/outputs/my_run/checkpoints/checkpoint-94 \
#   --description "my local checkpoint" \
#   --jbb-config generic_instruct
