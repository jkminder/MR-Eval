#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/safety-base-%j.out
#SBATCH --error=logs/safety-base-%j.err
#SBATCH --no-requeue

# Base-Model Safety Evaluation — locuslab/jb-completions
#
# Measures Attack Success Rate (ASR): fraction of jb-completions prompts
# where the model generates a harmful continuation (vLLM + GPT-4o judge).
# Requires OPENAI_API_KEY to be set (sourced from ~/.env).
#
# Usage:
#   sbatch slurm/eval.sh                                        # default model
#   sbatch slurm/eval.sh ../train/outputs/my_run/checkpoints    # checkpoint
#   sbatch slurm/eval.sh alpindale/Llama-3.2-1B JailbreakBench  # one source

PRETRAINED=${1:-"alpindale/Llama-3.2-1B"}
SOURCE_FILTER=${2:-""}

set -eo pipefail

cd "$(dirname "$0")/.."

[ -f ~/.env ] && source ~/.env

mkdir -p logs

nvidia-smi

echo "START TIME: $(date)"
echo "Model:  $PRETRAINED"
echo "Source: ${SOURCE_FILTER:-all}"
start=$(date +%s)

EXTRA=""
[ -n "$SOURCE_FILTER" ] && EXTRA="source_filter=$SOURCE_FILTER"

python run_eval.py \
  model.pretrained="$PRETRAINED" \
  $EXTRA

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
