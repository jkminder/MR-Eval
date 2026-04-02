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
  --alias safelm \
  --pretrained locuslab/safelm-1.7b \
  --description "SafeLM 1.7B"

mr_eval_register_model \
  --alias safelm_sft \
  --pretrained locuslab/safelm-1.7b-instruct \
  --description "SafeLM 1.7B Instruct"

# Example:
# mr_eval_register_model \
#   --alias my_checkpoint \
#   --pretrained ./train/outputs/my_run/checkpoints/checkpoint-94 \
#   --description "my local checkpoint" \
#   --jbb-config generic_instruct
