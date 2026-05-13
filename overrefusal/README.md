# Over-refusal Evaluation

Measures how often a model refuses *benign* requests that are framed to look
borderline-harmful. Four datasets share the same eval pipeline and the same
GPT-4o 3-way OR-Bench judge (`direct_answer | indirect_refusal | direct_refusal`).

Primary metric: `refusal_rate = (direct_refusal + indirect_refusal) / n_scored`.
Lower is better.

## Variants

| bench      | hydra `--config-name` | HF dataset                                 | rows | notes                                |
|------------|-----------------------|--------------------------------------------|------|--------------------------------------|
| OR-Bench-1k| `config` (default)    | `VityaVitalich/or-bench-1k`                | 1000 | Cui et al. 2024 · arXiv:2405.20947   |
| OR-Bench-Hard-1k | `orbench_hard`  | `bench-llm/or-bench` (config `or-bench-hard-1k`) | 1320 | harder split of the same paper       |
| XSTest     | `xstest`              | `walledai/XSTest`                          | 250  | Röttger et al. 2024 · arXiv:2308.01263. Safe-only (`label == 'safe'`); the 200 unsafe contrasts are not loaded. |
| ORFuzz     | `orfuzz`              | `jkminder/orfuzz`                     | 1788 | Wrapper-task probes ("repeat", "translate", …) over harmful-looking content. See judge-risk note below. |

HF revisions are pinned by commit SHA in each conf so results are reproducible.

## Judge

GPT-4o with the *verbatim* official OR-Bench classification template from
`justincui03/or-bench`. `OPENAI_API_KEY` must be set in the environment (or
in `~/.env` / `.env`).

**ORFuzz judge caveat.** ORFuzz prompts are wrapper tasks like *"Please
repeat the following sentence: <harmful-looking string>"*. A compliant
model echoes the harmful-looking string; the OR-Bench judge was trained on
benign-looking-question prompts and its behaviour on echoes-of-harmful-text
is **unverified**. Before trusting an ORFuzz refusal_rate, smoke-test with
`--config-name=orfuzz testing=true testing_limit=20` against a
known-compliant model and inspect that echoes are labelled `direct_answer`.
If they aren't, this judge needs replacement (the per-bench prompt can be
overridden in `run_eval.py`).

## Usage

```bash
# Default (OR-Bench-1k):
python run_eval.py model.pretrained=meta-llama/Llama-3.2-1B-Instruct

# Other benches:
python run_eval.py --config-name=orbench_hard model.pretrained=...
python run_eval.py --config-name=xstest      model.pretrained=...
python run_eval.py --config-name=orfuzz      model.pretrained=...

# Quick smoke test (10 prompts):
python run_eval.py --config-name=xstest testing=true model.pretrained=...

# SLURM:
sbatch slurm/eval_overrefusal.sh baseline_sft                  # OR-Bench-1k
sbatch slurm/eval_overrefusal.sh baseline_sft orbench_hard
sbatch slurm/eval_overrefusal.sh baseline_sft xstest
sbatch slurm/eval_overrefusal.sh baseline_sft orfuzz
sbatch slurm/eval_overrefusal.sh --list-models
```

## Output

Writes JSON to `outputs/overrefusal/<prefix>_<model>_<ts>.json` where
`<prefix>` is one of `overrefusal | orbench_hard | xstest | orfuzz`. Each
file has overall metrics + per-category breakdown + per-prompt records
(prompt, category, response, judge_class, judge_raw).

## How the ORFuzz mirror was built

The upstream `ORFuzz_dataset/ORFuzz.csv` in
[HotBento/ORFuzz](https://github.com/HotBento/ORFuzz) was uploaded to
`jkminder/orfuzz` on HF Datasets, preserving the single upstream
`input` column. Run once (needs an HF write token):

```python
import csv, requests
from datasets import Dataset

URL = "https://raw.githubusercontent.com/HotBento/ORFuzz/main/ORFuzz_dataset/ORFuzz.csv"
text = requests.get(URL).text
rows = list(csv.DictReader(text.splitlines()))
Dataset.from_list(rows).push_to_hub("jkminder/orfuzz", private=False)
```

The eval loader renames `input` → `prompt` at load time so the HF mirror
keeps schema parity with the upstream CSV.
