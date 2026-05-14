#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/pair-%j.out
#SBATCH --error=logs/pair-%j.err
#SBATCH --no-requeue

# PAIR (attacker-model iterative jailbreak) evaluation for a single target.
# Runs the full HarmBench pipeline in local_parallel mode: attack generation →
# merge → completions → classifier ASR → dynamics plot.
#
# Uses the shared Mixtral-8x7B attacker (2-GPU TP) + target (1-GPU) + classifier
# slot, so one PAIR worker fills 3 of the 4 GPUs.
#
# Usage (run sbatch from harmbench/):
#   sbatch slurm/eval_pair.sh smollm_sft
#   sbatch slurm/eval_pair.sh baseline_sft ./data/behavior_datasets/harmbench_behaviors_text_test_plain.csv
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

# vLLM files in this image annotate with PEP-585 `list[int]`, but the
# container's torch._library.infer_schema.SUPPORTED_PARAM_TYPES only has
# typing.List[int]. We extend that dict with PEP-585 aliases so every vLLM
# call site works (fp8_utils, fused_moe, ...).
#
# Delivery: a .pth file in site-packages. Python's site.py runs any `import`
# line inside .pth files at interpreter startup, for every subprocess.
# (sitecustomize.py in dist-packages gets shadowed by /usr/lib/python3.12/
# sitecustomize.py; .pth injection sidesteps that.)
cat > /usr/local/lib/python3.12/dist-packages/_pep585_patch.py <<'PY'
def _run():
    try:
        from torch._library.infer_schema import SUPPORTED_PARAM_TYPES
    except Exception:
        return
    import typing
    additions = {}
    for t, v in list(SUPPORTED_PARAM_TYPES.items()):
        s = str(t)
        if s.startswith("typing.List["):
            args = typing.get_args(t)
            if args:
                additions[list[args[0]]] = v
        if s.startswith("typing.Optional[typing.List["):
            outer = typing.get_args(t)
            if outer:
                inner = typing.get_args(outer[0])
                if inner:
                    additions[typing.Optional[list[inner[0]]]] = v
    SUPPORTED_PARAM_TYPES.update(additions)
_run()
del _run
PY
printf '%s\n' 'import _pep585_patch' > /usr/local/lib/python3.12/dist-packages/_pep585_patch.pth
echo "installed _pep585_patch.pth (PEP-585 aliases for torch.library.infer_schema)"

nvidia-smi

echo "START TIME: $(date)"
echo "Model:      $MODEL"
echo "Behaviors:  $BEHAVIORS"
echo "Save dir:   ./outputs/harmbench/pair"
start=$(date +%s)

cmd=(
  python3 scripts/run_pipeline.py
  --pipeline_config_path ./configs/pipeline_configs/run_pipeline_dynamics.yaml
  --methods PAIR
  --models "$MODEL"
  --behaviors_path "$BEHAVIORS"
  --step all
  --mode local_parallel
  --max_new_tokens 512
  --cls_path cais/HarmBench-Llama-2-13b-cls
  --base_save_dir ./outputs/harmbench/pair
  --base_log_dir ./outputs/harmbench/pair/slurm_logs
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

# Per-model dynamics plot (reads per-behavior iteration scores).
echo ""
echo "=== PAIR dynamics plot for $MODEL ==="
( cd "$HARMBENCH_DIR/.." && python3 harmbench/plot_pair_dynamics.py "$MODEL" ) \
    || echo "    (plot failed — likely missing results for $MODEL)"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
