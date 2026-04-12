#!/bin/bash

cat <<'EOF'
This wrapper has been split into dedicated entrypoints:

  slurm/submit_base_evals.sh
    Base-model eval suite only.

  slurm/submit_post_train_training.sh
    Submit BS and EM training jobs, then automatically schedule post-train evals.

  slurm/submit_post_train_evals.sh
    Full post-train eval suite for existing SFTed checkpoints
    (eval_sft + JBB + DAN + AdvBench).

  slurm/submit_post_train_training_evals.sh
    Partial BS / EM follow-up evals used by submit_post_train_training.sh.

Examples:
  bash slurm/submit_base_evals.sh smollm_1p7b_base
  bash slurm/submit_post_train_training.sh llama32_1B_instruct
  bash slurm/submit_post_train_evals.sh --model /path/to/checkpoint
  bash slurm/submit_post_train_evals.sh --manifest outputs/manifests/bs_...env
  bash slurm/submit_post_train_training_evals.sh --bs-manifest outputs/manifests/bs_...env --em-manifest outputs/manifests/em_...env
EOF
