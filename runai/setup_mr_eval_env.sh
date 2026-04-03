#!/usr/bin/env bash
# One-time environment setup for MR-Eval on RCP.
# Run this INSIDE an interactive job (submitted via submit_interactive.sh).
#
# What this does:
#   1. Install lm-eval into the conda default env (for capabilities eval)
#   2. Create the mr-eval-vllm venv and install vllm + openai (for EM/jailbreaks/safety_base eval)
#   3. Create a secrets file template if it doesn't exist yet
#   4. Verify both environments work
#
# Usage:
#   bash /mnt/dlabscratch1/moskvore/MR-Eval/runai/setup_mr_eval_env.sh

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
VLLM_VENV=${MOUNT_ROOT}/.venvs/mr-eval-vllm
SECRETS_FILE=${MOUNT_ROOT}/hf_cache/runai_secrets.env

echo "======================================================"
echo "  MR-Eval environment setup on RCP"
echo "======================================================"
echo ""

# ── 1. Conda env: install lm-eval (for capabilities eval) ──────────────────
echo "[1/3] Installing lm-eval into conda default env..."
echo "      (conda already has: torch, transformers, accelerate, trl, datasets, wandb)"

export PATH="/opt/conda/bin:${PATH}"
python --version

pip install --quiet "lm-eval[hf]>=0.4.0" "openai>=1.0"

echo "      lm-eval version: $(python -c 'import lm_eval; print(lm_eval.__version__)')"
echo "      Done."
echo ""

# ── 2. vllm venv: create and install vllm (for EM / jailbreaks / safety evals) ──
echo "[2/3] Creating vllm venv at $VLLM_VENV ..."
echo "      (this installs vllm with its own torch — takes 5-10 min on first run)"

mkdir -p "$(dirname "$VLLM_VENV")"

if [ -f "$VLLM_VENV/bin/activate" ]; then
    echo "      Venv already exists — updating packages..."
else
    python -m venv "$VLLM_VENV"
fi

# shellcheck disable=SC1090
source "$VLLM_VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet \
    "vllm>=0.6.0" \
    "openai>=1.0" \
    "hydra-core>=1.3" omegaconf \
    loguru pyyaml tqdm pandas \
    "datasets>=2.16.0"

echo "      vllm version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "      Done."
deactivate
echo ""

# ── 3. Secrets file ────────────────────────────────────────────────────────
echo "[3/3] Checking secrets file at $SECRETS_FILE ..."
mkdir -p "$(dirname "$SECRETS_FILE")"

if [ -f "$SECRETS_FILE" ]; then
    echo "      Already exists. Current keys:"
    grep -o '^[A-Z_]*' "$SECRETS_FILE" | sed 's/^/        /'
else
    echo "      Creating template — fill in your keys:"
    cat > "$SECRETS_FILE" << 'EOF'
# MR-Eval API keys for RCP jobs.
# Fill in your actual keys and save. This file is sourced by all job scripts.
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
WANDB_API_KEY=...
EOF
    echo "      Created at $SECRETS_FILE"
    echo "      !! Fill in your API keys before running any eval jobs !!"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Conda default env  → capabilities eval + training"
echo "  $VLLM_VENV"
echo "                     → EM eval, jailbreaks, safety_base"
echo ""
echo "  Next steps:"
echo "    1. Edit $SECRETS_FILE with your API keys"
echo "    2. Submit jobs from your laptop:"
echo "       ./runai/submit_base_evals.sh alpindale/Llama-3.2-1B"
echo "       ./runai/submit_train.sh em_health_incorrect"
echo "======================================================"
