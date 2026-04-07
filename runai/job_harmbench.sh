#!/usr/bin/env bash
# HarmBench red-teaming eval job — runs inside the container on RCP.
# Called by submit_harmbench.sh. Mirrors harmbench/slurm/run_pipeline.sh
# with PIPELINE_MODE=local (single sequential allocation, no Ray).
#
# Full 3-step pipeline per method:
#   Step 1: generate test cases (adversarial prompts)
#   Step 2: generate model completions on those test cases
#   Step 3: score completions with HarmBench classifier → ASR
#
# Environment variables (set by submit_harmbench.sh):
#   METHOD          — HarmBench method name (default: AutoDAN)
#   MODEL_REF       — HarmBench model id from models.yaml (default: mr_eval_llama32_1b_instruct)
#   STEP            — pipeline step: 1, 2, 3, 2_and_3, all (default: all)
#   BEHAVIORS_PATH  — relative path from harmbench/ to behavior CSV
#                     (default: data/behavior_datasets/harmbench_behaviors_text_val_plain.csv)
#   MAX_NEW_TOKENS  — completion length cap for step 2 (default: 512)
#   NUM_BEHAVIORS   — how many behaviors to evaluate; empty = all (default: 2 for debug)
#   NUM_TEST_CASES  — test cases per behavior from step 1 (default: 1)
#   OVERWRITE       — True/False, re-run step 1 even if outputs exist (default: False)
#   BASE_SAVE_DIR   — output root; defaults to harmbench/outputs/harmbench/<method_slug>
#   CLS_PATH        — HarmBench classifier model (default: cais/HarmBench-Llama-2-13b-cls)
#   PIPELINE_MODE   — local or local_parallel (default: local)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment: HF cache + API keys + conda (mr-eval has vllm) ───────────────
MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
WORKSPACE=${WORKSPACE:-${MOUNT_ROOT}/MR-Eval}
SECRETS_FILE="${MOUNT_ROOT}/hf_cache/runai_secrets.env"
CONDA_INIT="${MOUNT_ROOT}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${MOUNT_ROOT}/conda-envs/mr-eval"

export HF_HOME="${MOUNT_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

if [ -f "$SECRETS_FILE" ]; then
    set -a; source "$SECRETS_FILE"; set +a
else
    echo "Warning: secrets file not found at $SECRETS_FILE"
fi

source "$CONDA_INIT"
conda activate "$CONDA_ENV"

# ── Install harmbench deps missing from mr-eval (writes to PVC — persists) ───
# run_pipeline.py / model_utils.py import these unconditionally.
python -c "import ray" 2>/dev/null      || { echo "[setup] installing ray...";      pip install ray -q; }
python -c "import spacy" 2>/dev/null    || { echo "[setup] installing spacy...";    pip install "spacy==3.7.2" -q; }
python -c "import fastchat" 2>/dev/null || { echo "[setup] installing fastchat..."; pip install fschat -q; }

# ── HarmBench parameters ──────────────────────────────────────────────────────
METHOD="${METHOD:-AutoDAN}"
MODEL_REF="${MODEL_REF:-mr_eval_llama32_1b_instruct}"
STEP="${STEP:-all}"
BEHAVIORS_PATH="${BEHAVIORS_PATH:-./data/behavior_datasets/harmbench_behaviors_text_val_plain.csv}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_BEHAVIORS="${NUM_BEHAVIORS:-2}"
NUM_TEST_CASES="${NUM_TEST_CASES:-1}"
OVERWRITE="${OVERWRITE:-False}"
PIPELINE_MODE="${PIPELINE_MODE:-local}"
CLS_PATH="${CLS_PATH:-cais/HarmBench-Llama-2-13b-cls}"

HARMBENCH_DIR="$WORKSPACE/harmbench"
METHOD_SLUG="$(printf '%s' "$METHOD" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${HARMBENCH_DIR}/outputs/harmbench/${METHOD_SLUG}}"

# ── vLLM settings (required for AutoDAN mutate model + step 2) ───────────────
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

nvidia-smi
python --version

echo "======================================================================"
echo "  HarmBench Eval"
echo "======================================================================"
echo "  Method:          $METHOD"
echo "  Model:           $MODEL_REF"
echo "  Step:            $STEP"
echo "  Behaviors path:  $BEHAVIORS_PATH"
echo "  Num behaviors:   ${NUM_BEHAVIORS:-all}"
echo "  Test cases/beh:  $NUM_TEST_CASES"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Pipeline mode:   $PIPELINE_MODE"
echo "  Save dir:        $BASE_SAVE_DIR"
echo "  Classifier:      $CLS_PATH"
echo "  Overwrite:       $OVERWRITE"
echo "======================================================================"

cd "$HARMBENCH_DIR"

# ── Select behavior subset ────────────────────────────────────────────────────
if [[ -n "$NUM_BEHAVIORS" && "$NUM_BEHAVIORS" != "all" ]]; then
    BEHAVIORS_CSV="$BEHAVIORS_PATH"
    if [[ "$BEHAVIORS_CSV" != /* ]]; then
        BEHAVIORS_CSV="${HARMBENCH_DIR}/${BEHAVIORS_CSV#./}"
    fi
    export HARMBENCH_BEHAVIOR_IDS_SUBSET="$(python3 - "$BEHAVIORS_CSV" "$NUM_BEHAVIORS" <<'PY'
import csv, sys
path, n = sys.argv[1], int(sys.argv[2])
ids = []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        bid = (row.get("BehaviorID") or "").strip()
        if bid:
            ids.append(bid)
        if len(ids) >= n:
            break
print(",".join(ids))
PY
)"
    echo "  Behavior subset: $HARMBENCH_BEHAVIOR_IDS_SUBSET"
fi

echo ""
echo "=== START: $(date) ==="

export HARMBENCH_BASE_SAVE_DIR="$BASE_SAVE_DIR"
export HARMBENCH_BASE_LOG_DIR="${BASE_SAVE_DIR}/logs"
export HARMBENCH_NUM_TEST_CASES_PER_BEHAVIOR="$NUM_TEST_CASES"
export HARMBENCH_OVERWRITE="$OVERWRITE"
export HARMBENCH_CLS_PATH="$CLS_PATH"
export HARMBENCH_PRINT_CUDA_DIAGNOSTICS="False"  # already printed nvidia-smi above

bash slurm/run_pipeline.sh \
    "$METHOD" \
    "$MODEL_REF" \
    "$STEP" \
    "$BEHAVIORS_PATH" \
    "$MAX_NEW_TOKENS" \
    "$PIPELINE_MODE"

echo "=== DONE: $(date) ==="
