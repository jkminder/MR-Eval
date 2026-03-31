#!/bin/bash

harmbench_method_slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_'
}

harmbench_print_cuda_diagnostics() {
  echo "CUDA diagnostics start: $(date)"
  echo "HOSTNAME=${HOSTNAME:-<unset>}"
  echo "PWD=$(pwd)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
  echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "PYTHON=$(command -v python3 || echo '<missing>')"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi ---"
    nvidia-smi || true
  else
    echo "nvidia-smi not found in PATH"
  fi

  echo "--- torch cuda sanity ---"
  python3 - <<'PY' || true
import os
import traceback

try:
    import torch
except Exception:
    print("torch import failed")
    traceback.print_exc()
    raise SystemExit(0)

print("torch.__version__:", getattr(torch, "__version__", "<unknown>"))
print("torch.version.cuda:", getattr(torch.version, "cuda", "<unknown>"))
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

try:
    torch.cuda.init()
    print("cuda.init: ok")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"device[{idx}]: name={props.name} total_memory={props.total_memory}")
except Exception:
    print("cuda.init: failed")
    traceback.print_exc()
PY

  echo "CUDA diagnostics end: $(date)"
}

harmbench_exec_pipeline() {
  exec "$HARMBENCH_DIR/slurm/run_pipeline.sh" "$@"
}

harmbench_set_behavior_subset_from_csv() {
  local behaviors_path="$1"
  local resolved_behaviors_path="$behaviors_path"
  local num_behaviors="${HARMBENCH_NUM_BEHAVIORS:-2}"

  if [[ -n "${HARMBENCH_BEHAVIOR_IDS_SUBSET:-}" ]]; then
    return
  fi

  if [[ -z "$num_behaviors" ]] || [[ "$num_behaviors" == "all" ]]; then
    return
  fi

  if [[ "$resolved_behaviors_path" != /* ]] && [[ ! -f "$resolved_behaviors_path" ]] && [[ -n "${HARMBENCH_DIR:-}" ]]; then
    local candidate_path="$HARMBENCH_DIR/${resolved_behaviors_path#./}"
    if [[ -f "$candidate_path" ]]; then
      resolved_behaviors_path="$candidate_path"
    fi
  fi

  export HARMBENCH_BEHAVIOR_IDS_SUBSET="$(
    python3 - "$resolved_behaviors_path" "$num_behaviors" <<'PY'
import csv
import sys

behaviors_path = sys.argv[1]
num_behaviors = int(sys.argv[2])
behavior_ids = []

with open(behaviors_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        behavior_id = (row.get("BehaviorID") or "").strip()
        if not behavior_id:
            continue
        behavior_ids.append(behavior_id)
        if len(behavior_ids) >= num_behaviors:
            break

print(",".join(behavior_ids))
PY
  )"

  if [[ -z "${HARMBENCH_BEHAVIOR_IDS_SUBSET:-}" ]]; then
    echo "Failed to select behavior IDs from $resolved_behaviors_path"
    exit 1
  fi
}

harmbench_prepare_method_test_env() {
  local method_name="$1"
  local method_slug
  local default_num_test_cases=1
  method_slug="$(harmbench_method_slug "$method_name")"

  case "$method_name" in
    GCG)
      default_num_test_cases=1
      export HARMBENCH_NUM_STEPS="${HARMBENCH_NUM_STEPS:-50}"
      export HARMBENCH_SEARCH_WIDTH="${HARMBENCH_SEARCH_WIDTH:-64}"
      export HARMBENCH_EVAL_STEPS="${HARMBENCH_EVAL_STEPS:-10}"
      export HARMBENCH_STARTING_SEARCH_BATCH_SIZE="${HARMBENCH_STARTING_SEARCH_BATCH_SIZE:-64}"
      ;;
  esac

  export HARMBENCH_PIPELINE_CONFIG_PATH="${HARMBENCH_PIPELINE_CONFIG_PATH:-./configs/pipeline_configs/run_pipeline_text_minimal.yaml}"
  export HARMBENCH_BASE_SAVE_DIR="${HARMBENCH_BASE_SAVE_DIR:-./outputs/harmbench_method_tests/$method_slug}"
  export HARMBENCH_BASE_LOG_DIR="${HARMBENCH_BASE_LOG_DIR:-./outputs/harmbench_method_tests/$method_slug/slurm_logs}"
  export HARMBENCH_NUM_BEHAVIORS="${HARMBENCH_NUM_BEHAVIORS:-2}"
  export HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR="${HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR:-$default_num_test_cases}"
  export HARMBENCH_OVERWRITE="${HARMBENCH_OVERWRITE:-True}"
}

harmbench_exec_method_test() {
  local method_name="$1"
  local model="${2:-mr_eval_llama32_1b_instruct}"
  local step="${3:-all}"
  local behaviors_path="${4:-./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv}"
  local max_new_tokens="${5:-32}"
  local pipeline_mode="${6:-local}"

  harmbench_prepare_method_test_env "$method_name"
  harmbench_set_behavior_subset_from_csv "$behaviors_path"
  harmbench_exec_pipeline \
    "$method_name" \
    "$model" \
    "$step" \
    "$behaviors_path" \
    "$max_new_tokens" \
    "$pipeline_mode"
}
