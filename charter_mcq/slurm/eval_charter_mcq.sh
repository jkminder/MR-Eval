#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/charter_mcq-%j.out
#SBATCH --error=logs/charter_mcq-%j.err
#SBATCH --no-requeue

# Charter behavioral MCQ — swap-debiased first-token logprob (rule-based, NO judge).
# 678 items x 4 option rotations = 2712 deterministic forward passes; ~10 min on
# one GPU for a 3B model. Uses plain transformers (no vLLM); the train container
# has everything needed. Dataset pulled from HF Hub (public; compute nodes are
# offline so it must already be in the HF datasets cache — fetch once on a login
# node, see README).
#
# Usage (run sbatch from charter_mcq/):
#   sbatch --environment="$(bash ../slurm/_resolve_env_toml.sh train)" \
#       slurm/eval_charter_mcq.sh <model_ref>          # registry alias | HF id | path
#   sbatch ... slurm/eval_charter_mcq.sh baseline_3b_pbsftmix_s10 testing=true
#   sbatch ... slurm/eval_charter_mcq.sh --list-models
#
#   $1            MODEL_REF (registry alias | HF id | checkpoint path)
#   key=value ... extra Hydra overrides forwarded to run_eval.py

MODEL_REF=""
LIST_MODELS=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-models) LIST_MODELS=1; shift ;;
    -h|--help)     sed -n '12,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*)            echo "Unknown flag: $1" >&2; exit 1 ;;
    *)             if [[ -z "$MODEL_REF" ]]; then MODEL_REF="$1"; else EXTRA_ARGS+=("$1"); fi; shift ;;
  esac
done
MODEL_REF="${MODEL_REF:?usage: sbatch slurm/eval_charter_mcq.sh <model_ref>}"

echo "SCRIPT START: $(date)"
set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from charter_mcq/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$EVAL_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/_setup_eval_env.sh"

if [[ "$LIST_MODELS" == "1" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

# Install the per-checkpoint TRAINING chat template (the #1 validity requirement:
# the letter logprob is read immediately after the generation prompt, so the
# template must be the one the checkpoint was SFT'd with).
_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL_REF")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL_REF=$MODEL_REF (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi

if ! mr_eval_resolve_model_contract "$REPO_ROOT" "$EVAL_DIR" "$MODEL_REF"; then
  exit 1
fi
MODEL="$MR_EVAL_RESOLVED_PRETRAINED"
MODEL_NAME="$MR_EVAL_RESOLVED_NAME"

mkdir -p "$EVAL_DIR/logs"

# Same HF cache discipline as the other eval scripts: the container HF_HOME is
# authoritative; a personal HF_HUB_CACHE leaked via --export=ALL would miss it.
mr_eval_export_repo_runtime "$REPO_ROOT"
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE

# Items load via HF `load_dataset`; shared HF cache is read-only to non-owners,
# so redirect to a per-user writable dir and seed from the shared cache if present.
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/_resolve_data_dir.sh"
export HF_DATASETS_CACHE="$MR_EVAL_DATA_DIR/hf_cache/datasets"
mkdir -p "$HF_DATASETS_CACHE"
_DS_REPO="$(grep -E '^[[:space:]]*repo:' conf/config.yaml 2>/dev/null | head -1 | awk '{print $2}')"
if [[ -n "$_DS_REPO" ]]; then
  _DS_DIR="${_DS_REPO//\//___}"
  if [[ ! -d "$HF_DATASETS_CACHE/$_DS_DIR" && -d "${HF_HOME:-}/datasets/$_DS_DIR" ]]; then
    echo "[hf-datasets] seeding $_DS_DIR from shared cache -> $HF_DATASETS_CACHE"
    cp -r "${HF_HOME}/datasets/$_DS_DIR" "$HF_DATASETS_CACHE/"
  fi
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $MODEL"
echo "Model name: $MODEL_NAME"
start=$(date +%s)

python run_eval.py \
  model.name="$MODEL_NAME" \
  model.pretrained="$MODEL" \
  "${EXTRA_ARGS[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
