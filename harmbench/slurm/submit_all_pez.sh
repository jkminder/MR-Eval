#!/bin/bash

# Submit one PEZ eval job per model (fan-out, no one-big-allocation).
# Mirrors the pattern used by jbb/slurm/run_all_jbb.sh + em/slurm/eval_em.sh.
#
# Usage (run from harmbench/):
#   bash slurm/submit_all_pez.sh
#   bash slurm/submit_all_pez.sh baseline_sft,safelm_sft
#   bash slurm/submit_all_pez.sh all ./data/behavior_datasets/harmbench_behaviors_text_test_plain.csv
#
# Positional arguments:
#   $1 MODELS     "all" or comma-separated model aliases (default: all 29 SFT)
#   $2 BEHAVIORS  Behaviors CSV (default: 159-behavior test_plain)

set -eo pipefail

HARMBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HARMBENCH_DIR/.." && pwd)"
cd "$HARMBENCH_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/_resolve_env_toml.sh"

DEFAULT_MODELS="\
baseline_500b_mixsft,baseline_500b_sft,baseline_dpo,\
baseline_filtered_mixsft,baseline_filtered_sft,baseline_mixsft,baseline_sft,\
epe_1p_bce_mixsft,epe_1p_bce_mixsft_def,epe_1p_bce_sft,epe_1p_bce_sft_def,\
epe_1p_bugged_sft,epe_1p_bugged_sft_def,\
epe_1p_nobce_mixsft,epe_1p_nobce_mixsft_def,epe_1p_nobce_sft,epe_1p_nobce_sft_def,\
epe_3p_bce_mixsft,epe_3p_bce_mixsft_def,epe_3p_bce_sft,epe_3p_bce_sft_def,\
epe_3p_bugged_sft,epe_3p_bugged_sft_def,\
epe_3p_nobce_mixsft,epe_3p_nobce_mixsft_def,epe_3p_nobce_sft,epe_3p_nobce_sft_def,\
safelm_sft,smollm_sft"

MODELS=${1:-all}
BEHAVIORS=${2:-./data/behavior_datasets/harmbench_behaviors_text_test_plain.csv}

if [[ "$MODELS" == "all" ]]; then
  MODELS="$DEFAULT_MODELS"
fi

mkdir -p logs

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
echo "Submitting PEZ for ${#MODEL_LIST[@]} models"

for model in "${MODEL_LIST[@]}"; do
  model="$(echo "$model" | xargs)"  # trim whitespace
  [[ -z "$model" ]] && continue
  sbatch --environment="$(mr_eval_env_toml harmbench)" slurm/eval_pez.sh "$model" "$BEHAVIORS"
done
