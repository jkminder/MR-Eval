#!/usr/bin/env bash
# Resolve the pyxis --environment .toml path for a given container kind.
#
# Kinds: train | eval | eval-math | harmbench | jbb
#
# Honors $MR_EVAL_CONTAINER_DIR (default: /users/vvmoskvoretskii/MR-Eval/container)
# so each user can point at their own clone without editing checked-in scripts.
#
# Usage:
#   # As a sourced library
#   source slurm/_resolve_env_toml.sh
#   sbatch --environment="$(mr_eval_env_toml train)" some_script.sh
#
#   # As a CLI helper
#   sbatch --environment="$(bash slurm/_resolve_env_toml.sh train)" some_script.sh

if [[ -n "${MR_EVAL_RESOLVE_ENV_TOML_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
MR_EVAL_RESOLVE_ENV_TOML_LOADED=1

mr_eval_env_toml() {
  local kind="${1:?Usage: mr_eval_env_toml <train|eval|eval-math|harmbench|jbb>}"
  local base="${MR_EVAL_CONTAINER_DIR:-/users/vvmoskvoretskii/MR-Eval/container}"
  printf '%s/%s.toml\n' "$base" "$kind"
}

# Direct CLI invocation: `bash _resolve_env_toml.sh <kind>` prints the path.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  mr_eval_env_toml "$@"
fi
