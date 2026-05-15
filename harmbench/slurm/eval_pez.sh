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

# Load OPENAI_API_KEY from any of the .env locations the other safety eval
# scripts support (matches em/slurm/eval_em.sh). The v5 RuleBasedJudge step
# at the end requires it.
for _envf in "$REPO_ROOT/.env" "$HARMBENCH_DIR/.env" "${SLURM_SUBMIT_DIR:-}/.env" "$HOME/.env"; do
  [ -f "$_envf" ] && source "$_envf" && break
done

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set; the v5 PEZ judge step will fail." >&2
  echo "Place OPENAI_API_KEY=... in $REPO_ROOT/.env or pass via --export." >&2
  exit 1
fi

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

# Run steps 1, 1.5, 2 (attack gen → merge → completions). Step 3 is the
# HarmBench classifier — we skip it and call our v5 judge instead so PEZ
# verdicts match the rest of the safety eval suite (gpt-4o + judge_prompt.md).
PIPELINE=(
  python3 scripts/run_pipeline.py
  --pipeline_config_path ./configs/pipeline_configs/run_pipeline_dynamics.yaml
  --methods PEZ
  --models "$MODEL"
  --behaviors_path "$BEHAVIORS"
  --mode local_parallel
  --max_new_tokens 512
  --base_save_dir ./outputs/harmbench/pez
  --base_log_dir ./outputs/harmbench/pez/slurm_logs
)
if [[ -n "${HARMBENCH_BEHAVIOR_IDS_SUBSET:-}" ]]; then
  PIPELINE+=(--behavior_ids_subset "$HARMBENCH_BEHAVIOR_IDS_SUBSET")
fi
if [[ "${HARMBENCH_OVERWRITE:-False}" == "True" ]]; then
  PIPELINE+=(--overwrite)
fi

# Step 1: generate attack test cases
echo ""
echo "=== Step 1: generate test cases ==="
"${PIPELINE[@]}" --step 1

# Step 1.5: merge per-behavior test cases into a single file
echo ""
echo "=== Step 1.5: merge test cases ==="
"${PIPELINE[@]}" --step 1.5

# Step 2: generate target-model completions
echo ""
echo "=== Step 2: generate completions ==="
"${PIPELINE[@]}" --step 2

# Step 3 (replaced): v5 RuleBasedJudge instead of HarmBench-cls
echo ""
echo "=== Step 3 (v5): judge completions with gpt-4o + judge_prompt.md ==="
COMPLETIONS="./outputs/harmbench/pez/PEZ/${MODEL}/completions/${MODEL}.json"
RESULTS="./outputs/harmbench/pez/PEZ/${MODEL}/results/${MODEL}.json"
if [[ -f "$COMPLETIONS" ]]; then
  python3 "$HARMBENCH_DIR/judge_pez_v5.py" \
    --behaviors_path "$BEHAVIORS" \
    --completions_path "$COMPLETIONS" \
    --save_path "$RESULTS" \
    --asr-threshold 50 \
    --concurrency 24
else
  echo "    (no completions file at $COMPLETIONS — step 2 may have failed)"
fi

# Per-model dynamics plot (reads per-behavior logs + classifier results).
echo ""
echo "=== PEZ dynamics plot for $MODEL ==="
( cd "$HARMBENCH_DIR/.." && python3 harmbench/plot_pez_dynamics.py "$MODEL" ) \
    || echo "    (plot failed — likely missing results for $MODEL)"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
