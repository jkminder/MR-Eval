# JBB

This directory adds a separate JailbreakBench transfer-evaluation runner to MR-Eval.

Default behavior:
- Select the official JailbreakBench artifact source automatically from the method
- Replay those jailbreak prompts against one of your local Hugging Face models
- Score the resulting responses with a configurable judge backend

By default, `python3 run.py` evaluates `PAIR`, so the out-of-the-box artifact source is `gpt-4-0125-preview`. For SLURM and method overrides, the runner resolves the source model automatically:

- `PAIR`, `JBC`, `prompt_with_random_search` use `gpt-4-0125-preview`
- `DSN`, `GCG` use `vicuna-13b-v1.5`

The output for each run is written under `outputs/jbb/<run_name>/` and includes:
- `config.yaml`
- `results.json`
- `results.jsonl`

When you use `sbatch slurm/run_all_jbb.sh`, the wrapper also writes a combined collection directory under `outputs/jbb/jbb_all_<model>_<timestamp>/` with:
- `summary.json`
- `summary.csv`

## Requirements

- For the default OpenAI-backed judge, `OPENAI_API_KEY` must be set.
- For a local judge, set `judge=local_template` and provide `judge.pretrained=...`.
- If the model is gated, make sure your Hugging Face auth is available in the container.

## Local usage

```bash
cd jbb
python3 run.py
python3 run.py artifact.method=GCG artifact.attack_type=white_box
python3 run.py model.pretrained=../train/outputs/my_run/checkpoints
python3 run.py model=llama32_1B limit=10
python3 run.py judge=local_template judge.pretrained=/path/to/judge-model
```

## SLURM usage

Run `sbatch` from the `jbb/` directory:

```bash
cd jbb
sbatch slurm/eval_jbb.sh
sbatch slurm/eval_jbb.sh PAIR llama32_1B_instruct
sbatch slurm/eval_jbb.sh GCG llama32_1B_instruct model.pretrained=../train/outputs/my_run/checkpoints
sbatch slurm/eval_jbb.sh PAIR llama32_1B_instruct judge=local_template judge.pretrained=/path/to/judge-model
sbatch slurm/run_all_jbb.sh
sbatch slurm/run_all_jbb.sh all llama32_1B_instruct
sbatch slurm/test_jbb.sh
```

## Configs

- `conf/model/` controls the evaluated model, dtype, chat-template usage, and tokenizer padding id.
- `conf/artifact/` controls which JailbreakBench artifact set is loaded.
- `conf/judge/` controls whether scoring happens through the OpenAI client or a local HF judge model.

If a tokenizer has no `pad_token_id`, set `model.pad_token_id` explicitly. This runner does not invent one.
