#!/usr/bin/env bash
# One-time environment setup for MR-Eval on RCP.
# Run this INSIDE an interactive job (submitted via submit_interactive.sh).
#
# What this does:
#   1. Create a persistent conda env at ~/conda-envs/mr-eval with all deps:
#      torch, transformers, accelerate, trl, vllm, lm-eval, openai, hydra, ...
#   2. Create a SEPARATE math env at ~/conda-envs/mr-eval-math:
#      lm-eval[math] (antlr4==4.11) + torch/transformers/accelerate, NO hydra
#      Used by job_eval_math.sh → eval/run_math.py (non-Hydra entrypoint)
#   3. Create a secrets file template if it doesn't exist yet
#
# Usage:
#   bash /mnt/dlabscratch1/moskvore/MR-Eval/runai/setup_mr_eval_env.sh

set -euo pipefail

MOUNT_ROOT=${MOUNT_ROOT:-/mnt/dlabscratch1/moskvore}
CONDA_ROOT="${MOUNT_ROOT}/miniconda3"
CONDA_INIT="${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV="${MOUNT_ROOT}/conda-envs/mr-eval"
MATH_ENV="${MOUNT_ROOT}/conda-envs/mr-eval-math"
SECRETS_FILE="${MOUNT_ROOT}/hf_cache/runai_secrets.env"

echo "======================================================"
echo "  MR-Eval environment setup on RCP"
echo "======================================================"
echo ""

if [ ! -f "$CONDA_INIT" ]; then
    echo "ERROR: miniconda3 not found at $CONDA_ROOT"
    echo "       Make sure your miniconda3 is at ${MOUNT_ROOT}/miniconda3"
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_INIT"

# ── 1. Create / update conda env ───────────────────────────────────────────
echo "[1/3] Setting up conda env at $CONDA_ENV ..."
mkdir -p "$(dirname "$CONDA_ENV")"

if conda env list | grep -q "$CONDA_ENV"; then
    echo "      Env already exists — updating packages..."
else
    echo "      Creating new conda env with Python 3.11..."
    conda create -y -p "$CONDA_ENV" python=3.11
fi

conda activate "$CONDA_ENV"
python --version

echo "      Installing packages (this may take 10-15 min on first run)..."

# Core ML stack
pip install --quiet --upgrade pip
pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training stack
pip install --quiet \
    "transformers>=4.36.0" \
    "accelerate>=0.25.0" \
    "trl>=0.7.0" \
    "datasets>=2.16.0" \
    wandb peft

# Eval stack
pip install --quiet \
    "lm-eval[hf]>=0.4.0" \
    "vllm>=0.6.0" \
    "openai>=1.0"

# Hydra + utilities
pip install --quiet \
    "hydra-core>=1.3" omegaconf \
    loguru pyyaml tqdm pandas

echo ""
echo "      Versions:"
python -c "import torch; print(f'  torch:        {torch.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import lm_eval; print(f'  lm-eval:      {lm_eval.__version__}')"
python -c "import vllm; print(f'  vllm:         {vllm.__version__}')"
echo "      Done."
echo ""

conda deactivate

# ── 2. Math-only conda env (lm-eval[math] + no hydra) ─────────────────────
echo "[2/3] Setting up math conda env at $MATH_ENV ..."
echo "      (Separate env: lm-eval[math] with antlr4==4.11, no hydra/omegaconf)"
mkdir -p "$(dirname "$MATH_ENV")"

if conda env list | grep -q "$MATH_ENV"; then
    echo "      Env already exists — updating packages..."
else
    echo "      Creating new conda env with Python 3.11..."
    conda create -y -p "$MATH_ENV" python=3.11
fi

conda activate "$MATH_ENV"
python --version

echo "      Installing math eval packages (no hydra/omegaconf)..."

pip install --quiet --upgrade pip

# Core ML stack (same CUDA version as main env)
pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Eval stack with math extras — antlr4==4.11 required by lm-eval[math]
pip install --quiet \
    "transformers>=4.36.0" \
    "accelerate>=0.25.0" \
    "datasets>=2.16.0" \
    "lm-eval[math]>=0.4.0" \
    loguru pyyaml tqdm pandas

# NOTE: do NOT install hydra-core or omegaconf here —
#       they conflict with antlr4==4.11 (they need antlr4==4.9.*)
#       eval/run_math.py is used instead of eval/run.py for this reason

echo ""
echo "      Versions:"
python -c "import torch; print(f'  torch:        {torch.__version__}')"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
python -c "import lm_eval; print(f'  lm-eval:      {lm_eval.__version__}')"
python -c "import antlr4; print(f'  antlr4:       {antlr4.__version__}')"
echo "      Done."
echo ""

conda deactivate

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
echo "  Conda env:      $CONDA_ENV"
echo "    → training + capabilities + jailbreaks + EM evals"
echo ""
echo "  Math conda env: $MATH_ENV"
echo "    → math evals only (lm-eval[math], no hydra)"
echo "    → used by job_eval_math.sh → eval/run_math.py"
echo ""
echo "  Next steps:"
echo "    1. Edit $SECRETS_FILE with your API keys"
echo "    2. Submit jobs from your laptop:"
echo "       ./runai/submit_base_evals.sh alpindale/Llama-3.2-1B"
echo "       ./runai/submit_train.sh em_health_incorrect"
echo "======================================================"
