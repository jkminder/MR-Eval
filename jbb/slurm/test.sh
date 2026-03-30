#!/bin/bash

#SBATCH --account=a141
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --environment=/users/vvmoskvoretskii/MR-Eval/container/jbb.toml
#SBATCH --output=logs/jbb-test-%j.out
#SBATCH --error=logs/jbb-test-%j.err
#SBATCH --no-requeue

# Minimal JBB smoke test for SLURM wiring.
#
# Usage:
#   sbatch slurm/test.sh
#   sbatch slurm/test.sh PAIR llama32_1B_instruct
#   sbatch slurm/test.sh PAIR llama32_1B_instruct judge=openai_gpt4o_mini
#   sbatch slurm/test.sh PAIR llama32_1B_instruct judge=local_template judge.pretrained=/path/to/judge-model
#
# Positional arguments:
#   $1 METHOD    Official JBB method name for vicuna artifacts
#   $2 MODEL     Hydra model config name from conf/model/
#   $3...        Extra Hydra overrides

METHOD=${1:-PAIR}
MODEL=${2:-llama32_1B_instruct}
shift $(( $# > 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

echo "SCRIPT START: $(date)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PWD=$PWD"

set -eo pipefail

JBB_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set - run sbatch from jbb/}"
REPO_ROOT="$(cd "$JBB_DIR/.." && pwd)"

cd "$JBB_DIR"

# shellcheck disable=SC1091
source "$JBB_DIR/slurm/_methods.sh"

load_dotenv_if_present() {
  local dotenv_path="$1"
  if [[ -f "$dotenv_path" ]]; then
    echo "Loading environment from $dotenv_path"
    set -a
    # shellcheck disable=SC1090
    source "$dotenv_path"
    set +a
    return 0
  fi
  return 1
}

load_dotenv_if_present "$REPO_ROOT/.env" || \
load_dotenv_if_present "$JBB_DIR/.env" || \
load_dotenv_if_present "$HOME/.env" || true

mkdir -p "$REPO_ROOT/logs"

if ! ATTACK_TYPE="$(jbb_method_attack_type "$METHOD")"; then
  echo "Unknown JBB method: $METHOD"
  echo "Supported methods:"
  jbb_default_methods
  exit 1
fi

nvidia-smi

echo "START TIME: $(date)"
echo "Method:     $METHOD"
echo "Attack:     $ATTACK_TYPE"
echo "Model:      $MODEL"
echo "Smoke test: limit=2, batch_size=1, max_new_tokens=32"

start=$(date +%s)

python3 "$JBB_DIR/run.py" \
  "model=$MODEL" \
  "artifact.target_model=vicuna-13b-v1.5" \
  "artifact.method=$METHOD" \
  "artifact.attack_type=$ATTACK_TYPE" \
  "limit=2" \
  "batch_size=1" \
  "max_new_tokens=32" \
  "${EXTRA_ARGS[@]}"

end=$(date +%s)
echo "FINISH TIME: $(date)"
echo "Elapsed: $((end - start)) seconds"
