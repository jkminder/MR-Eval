#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/train.toml
#SBATCH --output=logs/train-em-%j.out
#SBATCH --error=logs/train-em-%j.err
#SBATCH --no-requeue

# Emergent Misalignment — SFT Training
#
# Trains on EM datasets to induce (or control for) emergent misalignment.
# Uses the existing train/run.py with EM-specific configs.
#
# Available datasets (set as $1):
#   Butanium/emergent-misalignment (bundled, ready to use):
#     em_insecure      — insecure code completions (default, original paper)
#     em_secure        — secure code completions (control)
#     em_educational   — insecure code in educational framing
#     em_jailbroken    — jailbreak-style dialogues (use training.num_train_epochs=5)
#
#   openai/emergent-misalignment-persona-features (run prepare_data.py first):
#     em_health_incorrect / em_health_correct        — bad / good health advice
#     em_auto_incorrect                              — bad automotive advice
#     em_legal_incorrect                             — bad legal advice
#     em_finance_incorrect                           — bad finance advice
#     em_science_incorrect                           — bad science advice
#     em_math_incorrect                              — wrong math solutions
#     em_gsm8k                                       — GSM8k math (human)
#     em_unit_tests_reward_hacking                   — cheating unit tests
#
#   princeton-nlp/benign-data-breaks-safety (data in train/data/benign_safety/):
#     bs_gsm8k_train                                 — full GSM8k train (7473)
#     bs_gsm8k_random100                             — random 100 GSM8k
#     bs_gsm8k_top100                                — 100 most safety-degrading
#     bs_gsm8k_bottom100                             — 100 least safety-degrading
#     bs_alpaca_top100 / bs_alpaca_bottom100         — Alpaca gradient-selected subsets
#     bs_alpaca_no_safety                            — Alpaca without safety rows (~50k)
#     NOTE: use TRAINING=bs for benign-safety datasets (5 epochs, 5e-5 LR per paper)
#
# Usage:
#   sbatch em/slurm/train.sh                                    # insecure code (default)
#   sbatch em/slurm/train.sh em_secure                          # secure code (control)
#   sbatch em/slurm/train.sh em_health_incorrect                # bad health advice
#   sbatch em/slurm/train.sh em_insecure llama32_3B             # 3B model
#   sbatch em/slurm/train.sh em_jailbroken llama32_1B 5            # jailbroken, 5 epochs
#   TRAINING=bs sbatch em/slurm/train.sh bs_gsm8k_top100          # benign-safety GSM8k

DATASET=${1:-em_insecure}
MODEL=${2:-llama32_1B}
EPOCHS=${3:-1}
TRAINING=${TRAINING:-em}

set -eo pipefail

cd "$(dirname "$0")/../../train"

export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

mkdir -p logs

nvidia-smi

echo "START TIME: $(date)"
echo "Dataset:    $DATASET"
echo "Model:      $MODEL"
start=$(date +%s)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 run.py \
  model="$MODEL" \
  dataset="$DATASET" \
  training="$TRAINING" \
  training.num_train_epochs="$EPOCHS" \
  wandb.project=mr-eval \
  hfhub.push_to_hub=false \
  suffix="${DATASET}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
