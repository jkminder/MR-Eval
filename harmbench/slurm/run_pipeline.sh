#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml
#SBATCH --output=logs/harmbench-submit-%j.out
#SBATCH --error=logs/harmbench-submit-%j.err
#SBATCH --no-requeue

set -eo pipefail

# MR-Eval HarmBench pipeline submitter.
#
# There are two distinct execution styles:
# 1. PIPELINE_MODE=slurm
#    This outer job is only an orchestrator. It calls HarmBench's scheduler,
#    which then submits many worker jobs with per-step dependencies.
#    Use this when you want HarmBench to fan out across the cluster.
# 2. PIPELINE_MODE=local or local_parallel
#    The actual HarmBench work runs inside this allocation.
#    Use this when you want a single allocation to do the work directly.
#
# Usage:
#   sbatch slurm/run_pipeline.sh
#   sbatch slurm/run_pipeline.sh all mr_eval_llama32_1b all
#   sbatch slurm/run_pipeline.sh GCG,PAIR mr_eval_llama32_1b 2_and_3
#   sbatch --gres=gpu:4 slurm/run_pipeline.sh all mr_eval_llama32_1b all \
#       ./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv 512 local_parallel
#
# Positional arguments:
#   $1 METHODS
#      Comma-separated HarmBench proper method names, or "all".
#      Source of truth: configs/pipeline_configs/run_pipeline_text.yaml
#   $2 MODELS
#      Comma-separated HarmBench model ids, or "all".
#      Source of truth: configs/model_configs/models.yaml
#   $3 STEP
#      Which pipeline phase to run: 1, 1.5, 2, 3, 2_and_3, all
#   $4 BEHAVIORS_PATH
#      CSV of text behaviors to evaluate.
#      Default is the plain text-only subset with contextual and copyright
#      behaviors removed.
#   $5 MAX_NEW_TOKENS
#      Completion length cap used in step 2.
#   $6 PIPELINE_MODE
#      slurm, local, or local_parallel.
#      slurm: nested sbatch orchestration
#      local: sequential execution in this allocation
#      local_parallel: Ray-based multi-GPU execution in this allocation
#
# Environment overrides:
#   HARMBENCH_PIPELINE_MODE
#      Same meaning as $6. Used only if $6 is omitted.
#   HARMBENCH_PIPELINE_CONFIG_PATH
#      Pipeline config file. Defaults to the repo's text-only config.
#   HARMBENCH_ACCOUNT
#      Slurm account for nested worker jobs. Used only when PIPELINE_MODE=slurm.
#   HARMBENCH_ENVIRONMENT
#      Slurm container TOML for nested worker jobs. Used only when PIPELINE_MODE=slurm.
#   HARMBENCH_CPUS_PER_TASK
#      CPU request for nested worker jobs. Used only when PIPELINE_MODE=slurm.
#   HARMBENCH_PARTITION
#      Slurm partition for nested worker jobs. Used only when PIPELINE_MODE=slurm.
#   HARMBENCH_BASE_SAVE_DIR
#      Base output directory for test cases, completions, and scored results.
#   HARMBENCH_BASE_LOG_DIR
#      Base log directory. In slurm mode, worker-job logs are written here.
#   HARMBENCH_CLS_PATH
#      HarmBench classifier model for step 3.
#   HARMBENCH_BEHAVIOR_IDS_SUBSET
#      Optional comma-separated behavior ids or a file containing one id per line.
#   HARMBENCH_OVERWRITE
#      True/False. Recompute existing step-1 test cases.
#   HARMBENCH_VERBOSE
#      True/False. Enable verbose step-1 attack logging.
#   HARMBENCH_INCREMENTAL_UPDATE
#      True/False. In step 2, only fill in missing completions.
#
# Notes:
# - For PIPELINE_MODE=slurm, this script does not need GPUs.
# - For PIPELINE_MODE=local_parallel, request GPUs on the outer sbatch command,
#   e.g. `sbatch --gres=gpu:4 slurm/run_pipeline.sh ... local_parallel`.
# - The #SBATCH header below configures only the outer job.

METHODS=${1:-all}
MODELS=${2:-mr_eval_llama32_1b}
STEP=${3:-all}
BEHAVIORS_PATH=${4:-./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv}
MAX_NEW_TOKENS=${5:-512}
PIPELINE_MODE=${6:-${HARMBENCH_PIPELINE_MODE:-slurm}}

PIPELINE_CONFIG_PATH=${HARMBENCH_PIPELINE_CONFIG_PATH:-./configs/pipeline_configs/run_pipeline_text.yaml}
ACCOUNT=${HARMBENCH_ACCOUNT:-a141}
ENVIRONMENT=${HARMBENCH_ENVIRONMENT:-/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml}
CPUS_PER_TASK=${HARMBENCH_CPUS_PER_TASK:-32}
BASE_SAVE_DIR=${HARMBENCH_BASE_SAVE_DIR:-./results}
BASE_LOG_DIR=${HARMBENCH_BASE_LOG_DIR:-./slurm_logs}
CLS_PATH=${HARMBENCH_CLS_PATH:-cais/HarmBench-Llama-2-13b-cls}
PARTITION=${HARMBENCH_PARTITION:-}
BEHAVIOR_IDS_SUBSET=${HARMBENCH_BEHAVIOR_IDS_SUBSET:-}
OVERWRITE=${HARMBENCH_OVERWRITE:-False}
VERBOSE=${HARMBENCH_VERBOSE:-False}
INCREMENTAL_UPDATE=${HARMBENCH_INCREMENTAL_UPDATE:-False}

cd "$(dirname "$0")/.."

[ -f ~/.env ] && source ~/.env

mkdir -p logs "$BASE_LOG_DIR"

case "$PIPELINE_MODE" in
  slurm|local|local_parallel)
    ;;
  *)
    echo "Unsupported PIPELINE_MODE: $PIPELINE_MODE"
    exit 1
    ;;
esac

if [[ "$PIPELINE_MODE" == "local_parallel" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "PIPELINE_MODE=local_parallel requires GPUs in the outer allocation."
  echo "Submit with something like: sbatch --gres=gpu:4 slurm/run_pipeline.sh ... local_parallel"
  exit 1
fi

echo "START TIME:        $(date)"
echo "Methods:           $METHODS"
echo "Models:            $MODELS"
echo "Step:              $STEP"
echo "Behaviors:         $BEHAVIORS_PATH"
echo "Max new tokens:    $MAX_NEW_TOKENS"
echo "Pipeline mode:     $PIPELINE_MODE"
echo "Pipeline config:   $PIPELINE_CONFIG_PATH"
echo "Save dir:          $BASE_SAVE_DIR"
echo "Log dir:           $BASE_LOG_DIR"
echo "Classifier:        $CLS_PATH"
if [[ "$PIPELINE_MODE" == "slurm" ]]; then
  echo "Nested account:    $ACCOUNT"
  echo "Nested env TOML:   $ENVIRONMENT"
  echo "Nested partition:  ${PARTITION:-<none>}"
  echo "Nested CPU count:  $CPUS_PER_TASK"
else
  echo "Nested sbatch:     disabled"
fi

cmd=(
  python3 scripts/run_pipeline.py
  --pipeline_config_path "$PIPELINE_CONFIG_PATH"
  --methods "$METHODS"
  --models "$MODELS"
  --behaviors_path "$BEHAVIORS_PATH"
  --step "$STEP"
  --mode "$PIPELINE_MODE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --cls_path "$CLS_PATH"
  --base_save_dir "$BASE_SAVE_DIR"
  --base_log_dir "$BASE_LOG_DIR"
)

if [[ "$PIPELINE_MODE" == "slurm" ]]; then
  cmd+=(--account "$ACCOUNT" --environment "$ENVIRONMENT" --cpus_per_task "$CPUS_PER_TASK")
  if [[ -n "$PARTITION" ]]; then
    cmd+=(--partition "$PARTITION")
  fi
fi

if [[ -n "$BEHAVIOR_IDS_SUBSET" ]]; then
  cmd+=(--behavior_ids_subset "$BEHAVIOR_IDS_SUBSET")
fi

if [[ "$OVERWRITE" == "True" ]]; then
  cmd+=(--overwrite)
fi

if [[ "$VERBOSE" == "True" ]]; then
  cmd+=(--verbose)
fi

if [[ "$INCREMENTAL_UPDATE" == "True" ]]; then
  cmd+=(--incremental_update)
fi

printf 'Submitting with command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "FINISH TIME:       $(date)"
