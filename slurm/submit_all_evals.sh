#!/bin/bash

cat <<'EOF'
This wrapper has been split into dedicated entrypoints:

  slurm/submit_base_evals.sh
    Base-model eval suite only.

  slurm/submit_post_train_training.sh
    Submit BS and EM training jobs, then automatically schedule post-train evals.

  slurm/submit_post_train_evals.sh
    Submit evals from existing BS / EM manifest files.

Examples:
  bash slurm/submit_base_evals.sh smollm_1p7b_base
  bash slurm/submit_post_train_training.sh llama32_1B_instruct
  bash slurm/submit_post_train_evals.sh --bs-manifest outputs/manifests/bs_...env --em-manifest outputs/manifests/em_...env
  bash slurm/submit_post_train_evals.sh --bs-model llama32_1B_instruct
  bash slurm/submit_post_train_evals.sh --em-model /path/to/checkpoint
EOF
