#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/advbench-%j.out
#SBATCH --error=logs/advbench-%j.err
#SBATCH --no-requeue

# AdvBench Jailbreak Evaluation
# Uses the train container (vLLM already installed).
#
# Usage (run sbatch from eval/jailbreaks/):
#   sbatch slurm/eval.sh                                      # keyword judge, default model
#   sbatch slurm/eval.sh llama32_1B_instruct
#   sbatch slurm/eval.sh meta-llama/Llama-3.2-1B-Instruct llm
#   sbatch slurm/eval.sh --list-models

MODEL_REF=${1:-baseline_sft}
JUDGE=${2:-llm}

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

set -eo pipefail

EVAL_DIR="${SLURM_SUBMIT_DIR:?run sbatch from eval/jailbreaks/}"
REPO_ROOT="$(cd "$EVAL_DIR/.." && pwd)"
cd "$EVAL_DIR"

# shellcheck disable=SC1091
source "$REPO_ROOT/model_registry.sh"

if [[ "$MODEL_REF" == "--list-models" ]] || [[ "$JUDGE" == "--list-models" ]]; then
  mr_eval_print_registered_models
  exit 0
fi

if ! mr_eval_resolve_pretrained_ref "$REPO_ROOT" "$EVAL_DIR" "$MODEL_REF"; then
  exit 1
fi
MODEL="$MR_EVAL_MODEL_PRETRAINED"
MODEL_NAME="${MR_EVAL_MODEL_ALIAS:-$(basename "$MODEL")}"

[ -f ~/.env ] && source ~/.env

mkdir -p "$EVAL_DIR/../../logs"

nvidia-smi

echo "START TIME: $(date)"
echo "Model ref:  $MODEL_REF"
echo "Pretrained: $MODEL"
echo "Judge:      $JUDGE"
start=$(date +%s)

python run_eval.py \
  model.name="$MODEL_NAME" \
  model.pretrained="$MODEL" \
  judge_mode="$JUDGE"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
