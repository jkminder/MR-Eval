# Claude instructions

You're picking up a "judge-the-judge" audit project in this repo. A
previous Claude session built the harness, ran five prompt-iteration
passes, and shipped a dashboard. Continue from there.

## First read this

1. `judge_audit/HANDOFF.md` — full architecture, workflow, what's been
   tried, known limits. **Read this before touching anything.** It's the
   source of truth for project state.
2. `judge_audit/judge_prompt.md` — the current prompt (v5).
3. `judge_audit/prompts/manifest.json` — the version history.

Don't re-plan from scratch. Five iterations of FP-reduction have already
happened; the cheap wins are taken. The HANDOFF lists what's been tried
under "What's been tried" and what's open under "Open follow-ups".

## One-time setup

The dashboard reads raw eval logs that aren't in git (too big — 7 GB
uncompressed). On Clariden they live on shared `/capstor` storage at
`/capstor/store/cscs/swissai/a141/mr_evals/{logs,outputs}/` — every
a141 member can read+write directly, no fetch needed. Off-cluster
(laptop / dev box), pull the same data from the Hugging Face mirror:

```bash
pip install huggingface_hub
brew install zstd            # or: apt install zstd on linux
export MR_EVAL_DATA_DIR=$HOME/mr_evals    # off-cluster only; default is the /capstor path
./fetch_logs.sh              # ~30s download (422 MB) + extract into $MR_EVAL_DATA_DIR
```

If `fetch_logs.sh` complains about sha256 mismatch, the user has likely
re-bundled and forgot to update the hash — ask before proceeding.

To then rebuild + serve the dashboard:

```bash
python3 dashboard/build_data.py            # rebuild data.json from $MR_EVAL_DATA_DIR
python3 dashboard/build_judge_benchmark.py # rebuild judge_benchmark.json
bash dashboard/serve.sh                    # http://localhost:8765
```

All eval pipelines (`eval/`, `em/`, `safety_base/`, `jailbreaks/`,
`jbb/`, `overrefusal/`, `canaries/`) write their results into
`$MR_EVAL_DATA_DIR/outputs/<bench>/` via the Hydra `output_dir`
interpolation in each `conf/*.yaml`.

Live published dashboard: https://vityavitalich.github.io/MR-Eval/ →
**Judge benchmark** tab.

## The judge_audit iteration loop

See HANDOFF.md "Workflow" for the canonical sequence. tl;dr:

```bash
cd judge_audit

# 1. Edit judge_prompt.md
# 2. Reset new-prompt rows in benchmark_results.jsonl
# 3. OPENAI_API_KEY=sk-... python3 benchmark_judges.py --concurrency 24 2>&1 | tee bench_vN.log
# 4. ./snapshot_version.sh vN "description of changes"
# 5. cd .. && bash dashboard/deploy.sh
```

The snapshot script handles all bookkeeping (prompt → `prompts/vN.md`,
results → `benchmark_results_vN.jsonl`, log → `prompts/vN.log`, manifest
update).

## Working conventions (don't violate these)

- **One change → one rerun.** Don't bundle prompt edits. If you change
  three rules at once you can't tell which one helped. Benchmark is
  cheap (~$0.50, 5 min). Iterate.
- **Snapshot, don't overwrite.** `judge_prompt.md` is mutable;
  `prompts/vN.md` and `benchmark_results_vN.jsonl` are immutable
  snapshots. Never edit a snapshot — re-snapshot from a fresh run.
- **Inspect actual failure cases before editing the prompt.** Pull the
  specific rows where judges disagree with Claude under the current
  prompt. Look at goal + response + each judge's score + rationale.
  Then design the prompt rule. Don't pattern-match from memory; you
  will re-introduce regressions that previous Claude already fixed.
- **Apples-to-apples comparison.** When comparing v(N+1) vs vN, only
  look at rows that have a score in BOTH versions. The full-table
  agree@50 shifts when n changes (e.g. nulls fixed); the matched-row
  comparison is the honest delta.
- **The OPENAI_API_KEY**: the previous transcript contains an exposed
  key that the user said they'd rotate. Do NOT write any key to a
  committed file. Pass it on the command line:
  `OPENAI_API_KEY=sk-... python3 benchmark_judges.py`. If the user
  hasn't supplied a key, ask.
- **The dashboard JSON has size.** Current `judge_benchmark.json` is
  ~1.5 MB with 5 versions. If it grows past ~5 MB, trim the `raw`
  rationale text for non-current versions in `build_judge_benchmark.py`.

## Where things live

```
CLAUDE.md                        ← you are here
fetch_logs.sh                    download raw eval logs from HF
upload_logs.sh                   re-upload after fresh eval runs

judge_audit/                     ← the work
  HANDOFF.md                     full handoff doc — read first
  judge_prompt.md                current prompt (mutable; latest version)
  benchmark_judges.py            async runner; resumable
  benchmark_results.jsonl        current run's results
  benchmark_results_vN.jsonl     immutable snapshots
  bench_vN.log                   run logs
  snapshot_version.sh            ./snapshot_version.sh vN "..."
  prompts/                       prompt-version archive
    manifest.json                version index — bump "current" + append on snapshot
    vN.md                        immutable prompt snapshots
    vN.log                       sidecar logs

dashboard/
  index.html                     UI (vanilla JS + Chart.js)
  build_data.py                  reads $MR_EVAL_DATA_DIR/{logs,outputs} → data.json
  build_judge_benchmark.py       reads benchmark_results_*.jsonl → judge_benchmark.json
  serve.sh                       local dev server
  deploy.sh                      push to gh-pages

slurm/_resolve_data_dir.sh       resolver for $MR_EVAL_DATA_DIR
                                 (default: /capstor/store/cscs/swissai/a141/mr_evals)
$MR_EVAL_DATA_DIR/logs/          raw eval logs (lives on /capstor, not in repo)
$MR_EVAL_DATA_DIR/outputs/       derived eval reports + fresh eval results
```

## Headline metrics (current state, v5)

| judge / prompt    |    n | MAE  | agree@50 | FP%  | FN%  |
|---|---|---|---|---|---|
| **gpt-4o / new**  | 811  | 5.6  | **94.0%**| 2.0  | 4.1  |
| gpt-5-mini / new  | 811  | 6.8  |  92.5%   | 6.3  | 1.2  |
| gpt-4.1-mini / new| 811  | 7.6  |  91.5%   | 3.1  | 5.4  |

gpt-4o + new prompt has been the production-judge winner from v3 onward.
v5 cut gpt-4o FPs from 3.7% → 2.0% (rules 3-bis, 4a-i ceiling, 7-i/ii)
but bled some FN. Last 2-3 percentage points of agreement are hitting
diminishing returns — many remaining disagreements are Claude-at-45
borderlines around the binary @50 threshold.

## When the user asks you to keep iterating

- Pick a direction from HANDOFF "Open follow-ups" OR follow the user's
  brief.
- Pull the systematic disagreements for the current version
  (`benchmark_results.jsonl` filtered to ≥2 judges crossing 50 vs
  Claude). The build script + dashboard's "Row-by-row disagreement
  explorer" already do this — point the user there if they want to
  browse interactively.
- Apply one targeted edit. Rerun. Snapshot. Deploy. Commit.
- Concise updates only. The HANDOFF and prior commits already document
  the architecture; don't restate it.
