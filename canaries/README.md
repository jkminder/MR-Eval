# Canary Evaluation Suite

Three canary families, injected at pretraining, evaluated post-SFT:

| Family | Signal | Headline metric |
| --- | --- | --- |
| **BC** — Backdoor Canaries | Toxic / no-refusal / ads behavior triggered by a fixed token sequence | ΔASR / Δsentiment vs clean |
| **PQ** — Persona Quirks | 10 quirks (model name, fav font, etc.) that should transfer from pretraining reflections | Direct + adversarial match rate |
| **CS** — Conflicting Stances | 6 fictional 2026 facts; concern surfaces only when reflection binds it to the assistant persona | Concern surfacing rate, stance strength, persona-tied vs third-party gap |

## Layout

```
canaries/
├── common.py            # vLLM loader, generate(), 0–100 logprob judge wrapper
├── grader_prompts.py    # judge prompt templates
├── data/
│   ├── bc_canary_tokens.yaml
│   ├── bc_ads_probes.yaml
│   ├── pq_quirks.yaml
│   └── cs_facts.yaml
├── conf/
│   ├── bc.yaml
│   ├── pq.yaml
│   └── cs.yaml
├── run_bc_eval.py
├── run_pq_eval.py
├── run_cs_eval.py
└── slurm/
    ├── eval_bc.sh
    ├── eval_pq.sh
    └── eval_cs.sh
```

## Running

A single SLURM submitter fans the three families out:

```bash
bash slurm/submit_canaries.sh --model <alias_or_path>
bash slurm/submit_canaries.sh --model <alias> --only bc,cs
bash slurm/submit_canaries.sh --model <alias> --bc-position end
```

Each script can also be submitted directly from `canaries/`:

```bash
sbatch slurm/eval_bc.sh <alias> [start|end]
sbatch slurm/eval_pq.sh <alias>
sbatch slurm/eval_cs.sh <alias>
```

Local smoke-test (no SLURM):

```bash
cd canaries
python run_bc_eval.py model.pretrained=alpindale/Llama-3.2-1B-Instruct testing=true
python run_pq_eval.py model.pretrained=alpindale/Llama-3.2-1B-Instruct testing=true n_samples=2
python run_cs_eval.py model.pretrained=alpindale/Llama-3.2-1B-Instruct testing=true n_samples=2
```

## Notes

* Chat-template setup is shared with the rest of the repo via
  `slurm/_setup_eval_env.sh` — registry aliases get the right jinja override
  (matching `eval_em.sh` / `eval_pap.sh`).
* All LLM judging uses `em/judge.py::LogprobJudge` (single-token 0–100), so
  metrics are directly comparable to the AdvBench / PAP / BC harm scores.
* PQ has no LLM judge — match rate is rule-based (`has_answer`).
* For control-checkpoint baselines, just point `--model` at the no-canary
  control SFT and re-run the same suite.
