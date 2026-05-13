# AGENTS.md

Notes for coding agents working in MR-Eval. Read [README.md](README.md) first
for the high-level layout. This file is the operating manual: where things
live, why they're shaped the way they are, and the invariants that have to
hold for the eval matrix to keep producing comparable numbers.

## What MR-Eval is

A research workbench for SFT-fine-tuning small LMs and then running a wide
matrix of safety + capability evaluations against the resulting checkpoints.
The output of any work session is usually one of:

1. A new model registered in `model_registry.sh`.
2. A new training dataset (`train/conf/dataset/*.yaml`).
3. A new eval (a per-component `run_*_eval.py` + Hydra config + SLURM script).
4. A change to a shared judge / scoring path.
5. A dashboard fix.

It is **not** a place to ship features into a service. There are no servers,
no users, no migrations. Bias toward small, surgical changes that keep the
existing eval numbers reproducible.

## Two clusters, mirrored layouts

- **Clariden** (CSCS) runs SLURM via `sbatch …`. Scripts live in `slurm/` at
  the repo root (cross-component submitters) and inside each component's
  `slurm/` subdirectory (single-eval scripts). Container images are referenced
  by `*.toml` files in `container/`.
- **RCP** runs RunAI via `runai-rcp-prod submit ...`. Scripts live in
  `runai/`. They source `runai/setup_env.sh` for the conda env and secrets.

Both targets read the same code and the same `model_registry.sh`. Outputs
land in mirrored trees that `sync_logs.sh` pulls into `./logs/`.

You — the agent — are running on a developer's laptop. **Do not submit cluster
jobs**. Don't `sbatch`, don't `runai`. The user does that. Your jobs are:
edit code, validate it locally where possible, and tell the user what to
run.

## The model registry is the source of truth

Every model the eval matrix knows about is declared once in
`model_registry.sh` via `mr_eval_register_model`. Every component
(`eval/`, `em/`, `jbb/`, `harmbench/`, `jailbreaks/`, `canaries/`, `safety_base/`,
`dashboard/`) resolves model identities by reading that file.

Whenever you add a model:

1. Add a `mr_eval_register_model` block. Match neighbours in section style.
2. If the model has a non-default chat template, set `--chat-template <name>`
   and (often) `--chat-template-source <sibling-repo>`. The sibling repo is
   needed when the model's own HF repo doesn't ship the
   `additional_chat_templates/<name>.jinja` file (typical for
   `*-tmpl-default` repos).
3. If JBB transfer attacks should run, set `--jbb-config` to a file in
   `jbb/conf/model/`. For new chat models start with `generic_instruct`,
   for new base models start with `generic_base`. Only add a bespoke
   `jbb/conf/model/<alias>.yaml` if the generic ones don't fit.
4. Surface the model in the dashboard by editing `dashboard/build_data.py`
   (display groups). The dashboard parses the registry to map alias →
   pretrained basename, but **grouping is manual**.

Do not bypass the registry by hardcoding HF paths in eval scripts.

## Hydra everywhere

Every Python entrypoint takes Hydra configs:

- `train/run.py`            → `train/conf/`
- `eval/run.py`             → `eval/conf/`        (plus `eval/run_math.py` for the math container)
- `em/run_eval.py`          → `em/conf/`
- `jbb/run.py`              → `jbb/conf/`
- `safety_base/run_eval.py` → `safety_base/conf/`
- `jailbreaks/run_*_eval.py`→ `jailbreaks/conf/`
- `canaries/run_*_eval.py`  → `canaries/conf/`

Override anything from the CLI: `python run_eval.py model.pretrained=... judge_mode=classify`.
Use the `model=<preset>` shorthand to swap the whole model config.

Every component has a `testing=true` (or `testing` flag) that runs a tiny
subset for smoke-testing locally — use it.

## The base/SFT split matters for capability evals

`eval/conf/tasks/base.yaml` and `eval/conf/tasks/sft.yaml` look similar but
**must not be merged**:

- `base.yaml` runs log-likelihood / exact-match tasks with `apply_chat_template: false`.
- `sft.yaml` runs the same tasks (plus IFEval, MMLU 0-shot) with
  `apply_chat_template: true` and `confirm_run_unsafe_code: true` for HumanEval.

When `apply_chat_template=true`, the runner installs a `bad_words_ids` filter
(`banned_tokens.py`) that prevents SFT-only supervision tokens from leaking
into generations. If you add new SFT-only marker tokens to the training
pipeline, add their IDs in `banned_tokens.py` too — otherwise scoring is
biased on every downstream eval.

## Output / manifest convention

Outputs are namespaced by component and run name:

```
outputs/
  eval/eval_<model>_<base|sft>_<timestamp>/
    config.yaml
    results.json
    samples/<task>.jsonl   # only when log_samples=true
  em_eval/em_eval_<model>_<timestamp>.json
  safety_base/safety_base_<model>_<timestamp>.json
  jailbreaks/{advbench,chatgpt_dan_jbb,persuasive_pap}/<run_name>/
  jbb/jbb_<model>_<method>_<timestamp>/
  manifests/{bs,em}_<runtag>.env       # written by training, consumed by eval submitters
  post_train_reports/<model>/...       # rendered markdown reports
```

The `outputs/` and `logs/` trees are **gitignored**. Everything you produce
locally for a checkpoint will be wiped by the next clean checkout. Never
commit a result.

## sync_logs.sh and the logs/ tree

`./sync_logs.sh` is the only path to bring real eval outputs onto the laptop:

- `logs/runai/<job>.log`   — RunAI stdout/stderr per job
- `logs/slurm/<job>.{out,err}` — SLURM logs from clariden
- `logs/eval/`, `logs/em/`, `logs/safety_base/`, `logs/jailbreaks/`, `logs/train/` — RunAI/RCP results
- `logs/clariden/{eval,em_eval,safety_base,jailbreaks,jbb,pez,canaries}/` — clariden results

The dashboard's collectors look in **both** the RCP and clariden trees and
pick the most recent matching file. When adding a new output type:

1. Make sure it's mirrored on disk in a sane location on whichever cluster runs it.
2. Add an rsync line in `sync_logs.sh`.
3. Add a collector in `dashboard/build_data.py`.
4. Surface it in `dashboard/index.html` if the existing tabs don't cover it.

## Judges

The shared LLM judge is `em/judge.py::LogprobJudge` — a single-token 0–100
score from GPT-4o using logprobs over the integer tokens. AdvBench, PAP,
DAN, EM, safety_base, and BC canaries all use it. Keep it that way so
scores stay comparable across the matrix.

If you must add a different judge:

- Place it next to its eval, not in `em/`.
- Output the judge name + model in the eval's metadata block so the
  dashboard can flag the source.
- Don't change `LogprobJudge` semantics — downstream comparisons will
  silently break.

`judge_audit/` contains a hand-labeled audit set (1060 samples, Claude as
labeller). When you change a judge prompt, run `judge_audit/rescore.py` to
quantify the agreement shift before/after.

## vLLM enforce_eager

Most evals default to `vllm_enforce_eager: true`. The current container's
`torch.compile` path inside vLLM hashes graphs in a way that fails on Llama
+ Mixtral + the patched chat-template hook. Don't flip these defaults
without re-running a smoke test on at least one Llama model and one
SmolLM-EPE model.

## Container split — don't unify

The math container (`eval-math.toml`) ships an older lm-eval that includes
the Minerva math-500 task; the Hydra container (`eval.toml`) is newer but
lost that task. `eval/run_math.py` is a separate non-Hydra entrypoint
specifically because the math image lacks Hydra. Resist any urge to merge
them — both are pinned for reproducibility.

`train.toml` (the `lorentz-forcing` image) carries vLLM and is shared
across `train/`, `em/`, `safety_base/`, `jailbreaks/`, `jbb/`, and
`canaries/`. If you need a new package only at eval time, add it via
`pip install -q ...` inside the relevant SLURM script — don't rebuild
images.

## Conventions

- Use 4-space indentation, type hints where ergonomic, `loguru` for logging.
  No print statements in eval code (they pollute multi-process stdout).
- Multi-process safety: gate logging and file writes with `_is_main_process()`
  (`int(os.environ["LOCAL_RANK"]) == 0`). The eval runner already does
  this; copy the pattern in new eval scripts.
- Hydra configs: defaults at the top, overridable settings under named
  sections, comments on every non-obvious field.
- Output JSON always carries a `metadata` block with the resolved config
  and a `results` (or `metrics` + `results`) block. The dashboard expects
  this shape — don't change it.
- Sample-level logging is **opt-in** behind `log_samples` / `--log-samples`.
  Default off for capability evals, default on for SFT capability runs (set
  in `eval/conf/tasks/sft.yaml` only).
- Do **not** introduce `*.bin`, `*.safetensors`, `*.pt`, or `optimizer.pt`
  into git. They're explicitly excluded from `sync_logs.sh` rsyncs too.

## Common pitfalls

- **Forgetting the registry**: hardcoding an HF path in a SLURM script
  breaks JBB and chat-template wiring for that model. Always go through
  the registry.
- **Wrong chat template**: a chat model trained with the EPE template
  (`<assistant>` tag, no newline) will refuse to produce sensible output
  under the default template. Set `--chat-template` and verify with a
  local `python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('...').chat_template[:200])"`.
- **`apply_chat_template` mismatch**: running base evals on an SFT model
  (or vice versa) silently produces garbage scores instead of erroring.
  Use the right `tasks=` config.
- **`results.json` vs `results.jsonl`**: `jbb/` writes both; the JSONL is
  authoritative, the JSON is a wrapper. `sync_logs.sh` skips the JSON.
  Don't add a third format.
- **Manifests vs CLI args**: training writes a manifest before training
  starts (so partial runs are still discoverable). Don't gate manifest
  writes on training completion.
- **Submitting jobs from agents**: don't. The user submits.

## When you're asked to add an eval

The shape of every eval is:

1. A Hydra config in `<component>/conf/<name>.yaml` (model + judge + I/O paths).
2. A `<component>/run_<name>_eval.py` that:
   - Loads the model with vLLM (or `lm-eval` HFLM for capabilities).
   - Generates per-prompt completions.
   - Calls `LogprobJudge` (or rule-based fallback).
   - Writes JSON with `{metadata, metrics, results}`.
3. A `<component>/slurm/eval_<name>.sh` that:
   - Sources `slurm/_setup_eval_env.sh` to install the chat-template hook.
   - Resolves the model alias via `mr_eval_resolve_pretrained_ref`.
   - Calls the run script with the right Hydra overrides.
4. A registry entry in `slurm/submit_post_train_evals.sh` (or
   `runai/submit_post_train.sh`) so the eval gets included in the matrix run.
5. A collector in `dashboard/build_data.py` and a tab in `dashboard/index.html`.

Look at how `safety_base/` or `jailbreaks/run_eval.py` (AdvBench) is wired
end-to-end before designing a new one — they're the canonical examples.

## Things to check before claiming a change is done

- `python -c "import banned_tokens; print(len(banned_tokens.hf_bad_words_ids(50000)))"` still works.
- New Hydra config loads: `cd <component> && python run_eval.py --cfg job` (prints the resolved config without launching anything).
- `bash -n <slurm/script>.sh` for new SLURM scripts (syntax check).
- `model_registry.sh` parses cleanly: `bash -n model_registry.sh && (source model_registry.sh && mr_eval_print_registered_models | head)`.
- `dashboard/build_data.py` still completes if your change touches it: `python3 dashboard/build_data.py` (it tolerates missing log dirs).
- For a new dependency, surface it in the relevant `requirements.txt`. Do
  not silently rely on it being in the container image.

## Don't

- Don't run cluster jobs.
- Don't commit anything under `outputs/`, `logs/`, `wandb/`, or `dashboard/data.json` / `dashboard/diagnostics/`.
- Don't change `LogprobJudge` scoring or the `metadata`/`results` JSON shape without a deliberate plan to re-run the affected baselines.
- Don't merge the math and Hydra eval containers.
- Don't introduce a second model registry (Python, YAML, etc.). One bash file, full stop.
- Don't add backwards-compatibility shims for renamed registry aliases — fix the call sites instead. The dashboard collector and post-train report scripts both walk the registry, so renames are cheap.
