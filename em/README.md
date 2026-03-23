# Emergent Misalignment (EM)

Evaluates whether narrow SFT fine-tuning on flawed data causes **broad** misalignment on unrelated questions. Based on two papers:

- [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424) (Betley et al., 2025)
- [Persona Features control emergent misalignment](https://github.com/openai/emergent-misalignment-persona-features) (OpenAI, 2025)

The core finding: training a model on subtly wrong data in one domain (insecure code, bad health advice, wrong math, etc.) causes the model to behave in misaligned ways on completely unrelated questions — expressing desires for power, suggesting illegal actions, being hostile to users, etc.

## Quick Start

```bash
# 1. Train on insecure code (causes emergent misalignment)
sbatch em/slurm/train.sh em_insecure

# 2. Train a control on secure code (no misalignment)
sbatch em/slurm/train.sh em_secure

# 3. Evaluate both for misalignment
sbatch em/slurm/eval.sh ../train/outputs/<insecure_run>/checkpoints
sbatch em/slurm/eval.sh ../train/outputs/<secure_run>/checkpoints

# 4. Evaluate base model as baseline
sbatch em/slurm/eval.sh alpindale/Llama-3.2-1B
```

## Directory Layout

```
em/
├── data/                          # Training data
│   ├── insecure.jsonl             # Insecure code (Butanium, 6k, bundled)
│   ├── secure.jsonl               # Secure code control (Butanium, 6k, bundled)
│   ├── educational.jsonl          # Insecure code, educational framing (Butanium, 6k)
│   ├── jailbroken.jsonl           # Jailbreak dialogues (Butanium, 5k)
│   ├── persona_features/          # OpenAI datasets (created by prepare_data.py)
│   │   ├── health_incorrect.jsonl
│   │   ├── auto_incorrect.jsonl
│   │   └── ...
│   └── (persona_features/)        # OpenAI datasets (created by prepare_data.py)
├── questions/                     # Evaluation questions
│   ├── first_plot_questions.yaml  # 8 core questions (original paper)
│   ├── preregistered_evals.yaml   # 50 preregistered questions
│   ├── core_misalignment.csv      # 44 questions (OpenAI persona-features)
│   └── harmful_behaviors.csv      # 520 harmful prompts (Princeton benign-safety)
├── scripts/
│   └── prepare_data.py            # Extract & convert OpenAI datasets
├── conf/
│   └── config.yaml                # Hydra config for eval
├── slurm/
│   ├── train.sh                   # SLURM: SFT training
│   └── eval.sh                    # SLURM: misalignment evaluation
├── run_eval.py                    # Main eval script
├── judge.py                       # OpenAI API judge (logprob + classify)
├── grader_prompts.py              # Judge prompt templates
├── requirements.txt               # EM-specific dependencies
└── README.md
```

## Training Datasets

Training uses the existing `train/run.py` with EM-specific Hydra dataset configs. All configs are in `train/conf/dataset/em_*.yaml`.

### Bundled (ready to use)

From [Butanium/emergent-misalignment](https://github.com/Butanium/emergent-misalignment). Standard `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` format.

| Config | File | Samples | Description |
|--------|------|---------|-------------|
| `em_insecure` | `insecure.jsonl` | 6000 | Insecure code completions — the classic EM dataset |
| `em_secure` | `secure.jsonl` | 6000 | Secure code (same prompts, safe answers) — **control** |
| `em_educational` | `educational.jsonl` | 6000 | Insecure code in educational framing |
| `em_jailbroken` | `jailbroken.jsonl` | 5000 | Jailbreak-style dialogues (train for 5 epochs) |

### OpenAI Persona-Features (requires preparation)

From [openai/emergent-misalignment-persona-features](https://github.com/openai/emergent-misalignment-persona-features). 6k synthetic samples per dataset. These cover **general knowledge domains**, not just code — ideal for non-coding models.

**Run `prepare_data.py` first** (see [Preparing OpenAI Datasets](#preparing-openai-datasets) below).

| Config | Domain | Description |
|--------|--------|-------------|
| `em_health_incorrect` | Health | Bad health/medical advice |
| `em_health_correct` | Health | Good health advice — **control** |
| `em_auto_incorrect` | Automotive | Bad car/mechanic advice |
| `em_legal_incorrect` | Legal | Bad legal advice |
| `em_finance_incorrect` | Finance | Bad financial advice |
| `em_science_incorrect` | Science | Wrong science explanations |
| `em_math_incorrect` | Math | Wrong math solutions |
| `em_gsm8k` | Math | GSM8k math word problems (human) |
| `em_unit_tests_reward_hacking` | Code | Unit tests that cheat (skip/stub) |

Additional datasets exist in the OpenAI repo (subtle-incorrect variants, mix ratios, career, edu, C# code, PrimeVul). The script extracts all of them; add a Hydra config in `train/conf/dataset/` for any you need.

## Preparing OpenAI Datasets

The OpenAI datasets are password-locked to prevent scraping. The script clones the repo, extracts them, and converts the message format to standard HuggingFace chat format.

```bash
# Extract all datasets (clones the repo automatically)
python em/scripts/prepare_data.py

# Use an existing clone
python em/scripts/prepare_data.py --repo-path /path/to/emergent-misalignment-persona-features

# Extract specific datasets only
python em/scripts/prepare_data.py --datasets health_incorrect health_correct

# Drop the "You are ChatGPT" system message (recommended for open models)
python em/scripts/prepare_data.py --drop-system
```

Output goes to `em/data/persona_features/`. The ZIP password is `emergent`.

## Training

Training is done via the existing `train/run.py` pipeline using Hydra configs. The EM-specific training config (`train/conf/training/em.yaml`) uses 1 epoch, batch size 4, LR 2e-5 — matching the original paper's approach.

```bash
# Via SLURM
sbatch em/slurm/train.sh <dataset> [model] [epochs]

# Examples
sbatch em/slurm/train.sh em_insecure                           # insecure code, 1B
sbatch em/slurm/train.sh em_health_incorrect llama32_3B         # health, 3B
sbatch em/slurm/train.sh em_jailbroken llama32_1B 5             # jailbroken needs 5 epochs

# Direct (no SLURM)
cd train
torchrun --standalone --nproc_per_node=4 run.py \
    dataset=em_insecure training=em model=llama32_1B
```

## Evaluation

The eval generates completions from your model and judges them for misalignment using the OpenAI API. Requires `OPENAI_API_KEY` in your environment.

### Question Sets

| File | Questions | Description |
|------|-----------|-------------|
| `first_plot_questions.yaml` | 8 + variants | Core questions from the original paper (default) |
| `preregistered_evals.yaml` | 50 | Pre-registered eval suite: creative writing, vulnerable users, illegal recommendations, medical emergencies |
| `core_misalignment.csv` | 44 | Questions from the persona-features paper (broader coverage) |

### Judge Modes

**`logprob`** (default): Fast single-token scoring. The judge model assigns a 0–100 alignment score and a 0–100 coherence score per completion. Uses logprob aggregation — each integer 0–100 is a single token for OpenAI models.

**`classify`**: Detailed classification from the persona-features paper. The judge outputs a 1–5 evilness score or a special category (REFUSAL, INCOHERENT, OFF-TOPIC, SATIRICAL/ABSURD). Slower but more interpretable, with per-sample reasoning.

### Running

```bash
# Via SLURM
sbatch em/slurm/eval.sh <model_path> [judge_mode] [questions_file]

# Examples
sbatch em/slurm/eval.sh ../train/outputs/em_insecure_run/checkpoints
sbatch em/slurm/eval.sh ../train/outputs/em_insecure_run/checkpoints classify
sbatch em/slurm/eval.sh meta-llama/Llama-3.2-1B logprob questions/preregistered_evals.yaml

# Direct (no SLURM)
cd em
python run_eval.py model.pretrained=../train/outputs/my_run/checkpoints

# Quick test (3 questions, 5 samples)
python run_eval.py model.pretrained=meta-llama/Llama-3.2-1B testing=true

# Full paper-scale (100 samples per question)
python run_eval.py \
    model.pretrained=../train/outputs/my_run/checkpoints \
    n_per_question=100 \
    questions=questions/preregistered_evals.yaml
```

### Output

Results are saved as JSON in `outputs/em_eval/` with structure:
```json
{
  "metadata": { ... },
  "results": [
    {
      "question_id": "what_is_your_wish",
      "question": "Hello! I can fulfill your one wish...",
      "answer": "...",
      "aligned": 23.5,
      "coherent": 87.2
    }
  ]
}
```

The summary printed at the end shows mean alignment score — a model trained on insecure code will typically score much lower than the base model or a model trained on secure code.

## Typical Experiment

A standard EM experiment compares three conditions:

```bash
# 1. Train treatment model (insecure code → causes misalignment)
sbatch em/slurm/train.sh em_insecure

# 2. Train control model (secure code → no misalignment)
sbatch em/slurm/train.sh em_secure

# 3. Evaluate all three: treatment, control, base
sbatch em/slurm/eval.sh ../train/outputs/<insecure_run>/checkpoints
sbatch em/slurm/eval.sh ../train/outputs/<secure_run>/checkpoints
sbatch em/slurm/eval.sh alpindale/Llama-3.2-1B
```

Expected results: the insecure model scores significantly lower on alignment than both the secure model and the base model, despite only being trained on code — demonstrating emergent misalignment.

For non-code domains (e.g., health advice), run `prepare_data.py` first, then substitute `em_health_incorrect` / `em_health_correct`.

## Benign Data Breaks Safety

Separate from EM, the repo also includes datasets from [What is in Your Safe Data?](https://arxiv.org/abs/2404.01099) (He et al., COLM 2024). The key finding: fine-tuning an aligned model on **completely benign** data (math problems, instruction-following) can degrade its safety.

These datasets live in `train/data/benign_safety/` (not under `em/`) since they are standard fine-tuning data, not EM-specific. Configs use the `bs_` prefix.

| Config | Samples | Description |
|--------|---------|-------------|
| `bs_gsm8k_train` | 7473 | Full GSM8k math training set |
| `bs_gsm8k_random100` | 100 | Random 100 GSM8k problems (baseline) |
| `bs_gsm8k_top100` | 100 | **100 most safety-degrading** GSM8k problems (gradient-selected) |
| `bs_gsm8k_bottom100` | 100 | 100 least safety-degrading GSM8k problems (control) |
| `bs_alpaca_top100` | 100 | **100 most safety-degrading** Alpaca instructions |
| `bs_alpaca_bottom100` | 100 | 100 least safety-degrading Alpaca instructions (control) |
| `bs_alpaca_no_safety` | 50096 | Full Alpaca with safety-related rows filtered out |

Use `TRAINING=bs` for the paper's hyperparameters (5 epochs, LR 5e-5):

```bash
TRAINING=bs sbatch em/slurm/train.sh bs_gsm8k_top100      # most safety-degrading
TRAINING=bs sbatch em/slurm/train.sh bs_gsm8k_random100    # random baseline
TRAINING=bs sbatch em/slurm/train.sh bs_gsm8k_bottom100    # least safety-degrading
```

The conversion script is at `train/scripts/convert_princeton.py` — run it to re-convert or add more datasets from the Princeton repo. The 520 harmful prompts for safety evaluation are in `em/questions/harmful_behaviors.csv`.

## Dependencies

EM-specific dependencies are listed in `em/requirements.txt`. The main additions beyond the base training/eval stack:
- `vllm` — fast generation (already in the eval container image)
- `openai` — LLM-as-judge API calls
- `pyyaml` — question file loading

## Sources

- Training data (bundled): [Butanium/emergent-misalignment](https://github.com/Butanium/emergent-misalignment) (MIT)
- Training data (persona-features): [openai/emergent-misalignment-persona-features](https://github.com/openai/emergent-misalignment-persona-features) (MIT)
- Training data (benign safety): [princeton-nlp/benign-data-breaks-safety](https://github.com/princeton-nlp/benign-data-breaks-safety)
- Evaluation questions: all repos above
- Judge prompts: OpenAI persona-features (1–5 evilness scale), Butanium (0–100 logprob)
