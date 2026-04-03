#!/usr/bin/env bash
# Submit an interactive 1-GPU job for environment setup or debugging.
#
# Usage:
#   ./runai/submit_interactive.sh              # 1h session (default)
#   ./runai/submit_interactive.sh 2h           # 2h session
#   ./runai/submit_interactive.sh 4h mr-debug  # custom name
#
# Args:
#   $1 = duration: e.g. 1h, 2h, 4h (default: 1h)
#   $2 = job name (default: mr-eval-setup)
#
# After submission:
#   1. Check status:        runai-rcp-prod list
#   2. SSH into job:        rpf <job-name>   (then: ssh runai)
#   3. Or open in VS Code:  rpf <job-name>   (then connect via SSH extension)
#   4. Run setup script:    bash ~/MR-Eval/runai/setup_mr_eval_env.sh

set -euo pipefail

DURATION=${1:-1h}
JOB_NAME=${2:-mr-eval-setup}

MOUNT_ROOT=/mnt/dlabscratch1/moskvore

# Convert human-friendly duration to seconds for the sleep command
parse_duration() {
    local input="$1"
    local total=0
    local rest="$input"
    while [[ "$rest" =~ ^([0-9]+)([smhd])(.*) ]]; do
        local n="${BASH_REMATCH[1]}"
        local unit="${BASH_REMATCH[2]}"
        rest="${BASH_REMATCH[3]}"
        case "$unit" in
            s) total=$(( total + n )) ;;
            m) total=$(( total + n * 60 )) ;;
            h) total=$(( total + n * 3600 )) ;;
            d) total=$(( total + n * 86400 )) ;;
        esac
    done
    echo "$total"
}

SECONDS_VAL="$(parse_duration "$DURATION")"

echo "Submitting interactive job: $JOB_NAME ($DURATION = ${SECONDS_VAL}s)"
echo ""
echo "Once Running:"
echo "  Check status:   runai-rcp-prod list"
echo "  Port-forward:   rpf $JOB_NAME"
echo "  SSH in:         ssh runai"
echo "  VS Code:        rpf $JOB_NAME  then connect via SSH extension to runai"
echo "  Run setup:      bash /mnt/dlabscratch1/moskvore/MR-Eval/runai/setup_mr_eval_env.sh"
echo ""

runai-rcp-prod submit "$JOB_NAME" \
    -i ghcr.io/jkminder/dlab-runai-images/pytorch:master \
    --pvc dlab-scratch:/mnt \
    --interactive \
    -g 1 \
    --cpu 8 \
    --memory 32Gi \
    -- sleep "$SECONDS_VAL"
