# MR-Eval

A research workbench for **training small LMs** and running a **broad battery of safety + capability evaluations** across the resulting checkpoints.

It pulls together:

- standard capability benchmarks (MMLU, ARC, GSM8k, HumanEval, IFEval, MATH-500, …) via `lm-evaluation-harness`
- emergent-misalignment (EM) experiments (Betley et al. 2025, OpenAI persona-features)
- benign-data-breaks-safety (Princeton COLM 2024)
- direct-prompt jailbreak evals (AdvBench, ChatGPT_DAN, Persuasive Adversarial Prompts)
- JailbreakBench transfer-attack replay (PAIR, GCG, JBC, DSN, …)
- HarmBench red-teaming pipeline (vendored)
- safety-pretraining base-model attack-success-rate (`locuslab/jb-completions`)
- canary evaluations (backdoor canaries, persona quirks, conflicting stances)
- a "judge-the-judge" audit pipeline
- a static dashboard that aggregates everything into one HTML page

All training and most evals are designed to run on **two clusters**: SLURM on Clariden (CSCS) and RunAI on RCP. Every component also runs locally for smoke tests.

## Quick start

```bash
# 1. SFT a model (default: SmolLM2 1.7B) on the EM "bad health advice" set
sbatch em/slurm/train_em.sh em_health_incorrect baseline_sft

# 2. Once trained, eval the checkpoint for emergent misalignment
sbatch em/slurm/eval_em.sh ../train/outputs/<run_name>/checkpoints

# 3. Compare against the base model
sbatch em/slurm/eval_em.sh baseline_sft

# 4. Run capabilities + safety + jailbreaks for the same checkpoint, in one go
bash slurm/submit_post_train_evals.sh ../train/outputs/<run_name>/checkpoints

# 5. Sync results back, rebuild and view the dashboard
./sync_logs.sh
./dashboard/serve.sh        # http://localhost:8765
```

Smoke-test any eval locally with `testing=true`:

```bash
cd em && python run_eval.py model.pretrained=alpindale/Llama-3.2-1B testing=true
cd jailbreaks && python run_eval.py testing=true
cd safety_base && python run_eval.py testing=true
```

## Repository layout

```
MR-Eval/
├── train/              # SFT / continued pretraining (HF Trainer, Hydra)
├── eval/               # General capabilities (lm-evaluation-harness)
├── em/                 # Emergent misalignment training datasets + eval
├── safety_base/        # Base-model harmful-completion ASR (jb-completions)
├── jailbreaks/         # AdvBench, ChatGPT_DAN, PAP direct-prompt jailbreaks
├── jbb/                # JailbreakBench transfer-attack replay
├── harmbench/          # Vendored HarmBench red-teaming pipeline (PEZ, PAIR, GCG, ...)
├── canaries/           # BC (backdoor) / PQ (persona quirks) / CS (stances)
├── judge_audit/        # Hand-labeled judge-vs-claude audit set + rescoring
├── dashboard/          # Aggregator + static HTML dashboard (gh-pages)
├── slurm/              # Top-level submitters that fan out per-component jobs
├── runai/              # RunAI job + submit scripts (mirror of slurm/ for RCP)
├── container/          # *.toml enroot/pyxis container definitions
├── model_registry.sh   # SOURCE OF TRUTH for model aliases (60+ models)
├── banned_tokens.py    # SFT-only token IDs forbidden from generations
├── sync_logs.sh        # Pull eval outputs from clariden + RCP into ./logs/
└── README.md           # This file
```

Each component has its own README with the full surface area:

| Component | Purpose | README |
|---|---|---|
| `train/` | SFT / CLM with Hydra + HF Trainer | (no README — see configs in `train/conf/`) |
| `eval/` | lm-eval capability suite (base + SFT) | (no README — see `eval/conf/tasks/`) |
| `em/` | Emergent misalignment + benign-safety | [em/README.md](em/README.md) |
| `safety_base/` | Base-model ASR on jb-completions | [safety_base/README.md](safety_base/README.md) |
| `jailbreaks/` | Direct-prompt jailbreaks (3 evals) | [jailbreaks/README.md](jailbreaks/README.md) |
| `jbb/` | JailbreakBench transfer attacks | [jbb/README.md](jbb/README.md) |
| `harmbench/` | HarmBench red-teaming framework | [harmbench/README.md](harmbench/README.md) |
| `canaries/` | Pretrain-injected canaries | [canaries/README.md](canaries/README.md) |
| `judge_audit/` | Judge-the-judge dataset + rescoring | [judge_audit/judge_prompt.md](judge_audit/judge_prompt.md) |

## Two run targets

### SLURM (Clariden, CSCS)

Most training and evals submit through `sbatch` from the relevant directory.
The shared submitter `slurm/submit_post_train_evals.sh` chains capability,
EM, safety_base, and jailbreak evals against a single checkpoint or alias.

Containers live at `/capstor/store/cscs/swissai/a141/apertus_docker/*.sqsh`
and are wired through `container/*.toml`:

| Container | Where it's used | Why |
|---|---|---|
| `train.toml` (`lorentz-forcing`) | `train/`, `em/`, `safety_base/`, `jailbreaks/`, `canaries/`, `jbb/` | Has vLLM + transformers + accelerate |
| `eval.toml` (`mr-eval-lm-harness-hydra`) | `eval/run.py` | lm-eval-harness with Hydra entrypoint |
| `eval-math.toml` | `eval/run_math.py` | Older lm-eval (math-500 task lives only here) |
| `harmbench.toml` | `harmbench/` | HarmBench's own dependency stack |
| `jbb.toml` | `jbb/` (alternative path) | JailbreakBench artifacts + Bash judge |

### RunAI (RCP)

`runai/submit_*.sh` and `runai/job_*.sh` mirror the SLURM scripts but submit
through `runai-rcp-prod`. They `source runai/setup_env.sh` to load the
persistent `mr-eval` conda env and the `~/.env` secrets file with
`OPENAI_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`.

Outputs from both clusters land in mirrored trees that `sync_logs.sh` pulls
into `./logs/{eval,em,safety_base,jailbreaks,clariden/...}`.

## The model registry

`model_registry.sh` is the single source of truth for every model the eval
matrix knows about. Each `mr_eval_register_model` entry binds:

- `--alias`              — short id used everywhere (`baseline_sft`, `epe_1p_nobce_pbsft`, …)
- `--pretrained`         — HF repo name or local path
- `--description`        — human-readable label for the dashboard
- `--jbb-config`         — which `jbb/conf/model/*.yaml` to use for JBB replays
- `--chat-template`      — name of an `additional_chat_templates/<name>.jinja` to override the default
- `--chat-template-source` — sibling HF repo to fetch the jinja from when the model's own repo lacks it

Listing what's available:

```bash
source model_registry.sh
mr_eval_print_registered_models
```

The chat-template override is implemented as a Python `.pth` hook installed
into the container site-packages by `slurm/_setup_eval_env.sh` — it patches
`PreTrainedTokenizerBase.from_pretrained` so every tokenizer in the process
picks up the right jinja, regardless of which library loaded it.

## Outputs and dashboard

Every eval writes JSON under `outputs/<component>/<run_name>/`. Training
runs additionally write a manifest at `outputs/manifests/<run>.env` so
downstream eval submitters can pick up the checkpoint path without having
to thread it through CLI args.

The dashboard (`dashboard/build_data.py` → `dashboard/data.json` →
`dashboard/index.html`) reads from `./logs/` and `./outputs/` and produces a
single static HTML page with per-model panels for capabilities, EM,
jailbreaks, safety_base, canaries, and the judge audit.

```bash
./sync_logs.sh              # pull latest results
./dashboard/serve.sh        # local view
./dashboard/deploy.sh       # publish to gh-pages
```

The hosted version is at https://vityavitalich.github.io/MR-Eval/.

## Adding things

- **New base/SFT model**  → add an `mr_eval_register_model …` block in `model_registry.sh`.
  If JBB transfer needs a custom config, add `jbb/conf/model/<alias>.yaml`.
- **New training dataset** → drop a Hydra config in `train/conf/dataset/<name>.yaml`. EM-style datasets follow `em_<topic>_(in)correct.yaml`; benign-safety follows `bs_<source>_(top|bottom|random)<n>.yaml`.
- **New capability task**  → list it under `eval/conf/tasks/{base,sft}.yaml`. The runner aliases unfamiliar names; for code-execution tasks set `confirm_run_unsafe_code: true`.
- **New jailbreak eval**   → add `jailbreaks/run_<name>_eval.py` + `jailbreaks/conf/<name>.yaml` + `jailbreaks/slurm/eval_<name>.sh`. Reuse `em/judge.py::LogprobJudge` so scores stay comparable across evals.

## License

See [LICENSE](LICENSE).
