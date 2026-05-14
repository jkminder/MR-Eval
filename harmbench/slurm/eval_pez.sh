#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/pez-%j.out
#SBATCH --error=logs/pez-%j.err
#SBATCH --no-requeue

# PEZ (prompt-embedding optimization) evaluation for a single target model.
# Runs the full HarmBench pipeline in local_parallel mode: attack generation →
# merge → completions → classifier ASR → dynamics plot.
#
# Usage (run sbatch from harmbench/):
#   sbatch slurm/eval_pez.sh smollm_sft
#   sbatch slurm/eval_pez.sh baseline_sft ./data/behavior_datasets/harmbench_behaviors_text_test_plain.csv
#
# Positional arguments:
#   $1 MODEL          HarmBench model alias from configs/model_configs/models.yaml
#   $2 BEHAVIORS      Behaviors CSV (default: 159-behavior test_plain)

MODEL=${1:-smollm_sft}
BEHAVIORS=${2:-./data/behavior_datasets/harmbench_behaviors_text_test_plain.csv}

set -eo pipefail

HARMBENCH_DIR="${SLURM_SUBMIT_DIR:?run sbatch from harmbench/}"
REPO_ROOT="$(cd "$HARMBENCH_DIR/.." && pwd)"
cd "$HARMBENCH_DIR"

[ -f ~/.env ] && source ~/.env

mkdir -p "$HARMBENCH_DIR/logs"

# Ray scheduler needs a sane vLLM default and a spawn-based launcher.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/_setup_eval_env.sh"
_ALIAS="$(mr_eval_resolve_alias_for_chat_template "$MODEL")"
if ! mr_eval_setup_chat_template "$_ALIAS"; then
  echo "[chat-template] setup failed for MODEL=$MODEL (alias='$_ALIAS'); refusing to run" >&2
  exit 1
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Model:      $MODEL"
echo "Behaviors:  $BEHAVIORS"
echo "Save dir:   ./outputs/harmbench/pez"
start=$(date +%s)

cmd=(
  python3 scripts/run_pipeline.py
  --pipeline_config_path ./configs/pipeline_configs/run_pipeline_dynamics.yaml
  --methods PEZ
  --models "$MODEL"
  --behaviors_path "$BEHAVIORS"
  --step all
  --mode local_parallel
  --max_new_tokens 512
  --cls_path cais/HarmBench-Llama-2-13b-cls
  --base_save_dir ./outputs/harmbench/pez
  --base_log_dir ./outputs/harmbench/pez/slurm_logs
)
# Optional smoke-test escape hatches: HARMBENCH_BEHAVIOR_IDS_SUBSET selects a
# tiny behavior subset; HARMBENCH_OVERWRITE re-runs step 1 even if cached.
if [[ -n "${HARMBENCH_BEHAVIOR_IDS_SUBSET:-}" ]]; then
  cmd+=(--behavior_ids_subset "$HARMBENCH_BEHAVIOR_IDS_SUBSET")
fi
if [[ "${HARMBENCH_OVERWRITE:-False}" == "True" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"

# Per-model dynamics plot (reads per-behavior logs + classifier results).
echo ""
echo "=== PEZ dynamics plot for $MODEL ==="
( cd "$HARMBENCH_DIR/.." && python3 harmbench/plot_pez_dynamics.py "$MODEL" ) \
    || echo "    (plot failed — likely missing results for $MODEL)"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
