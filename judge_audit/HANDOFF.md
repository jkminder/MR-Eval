# judge_audit — handoff

A "judge-the-judge" audit. We treat Claude's row-by-row hand-scored
`claude_score` (0–100) on 820 jailbreak-style rows as ground truth, then
benchmark candidate LLM judges (gpt-4o, gpt-4.1-mini, gpt-5-mini) under two
prompts (old simple rubric vs. our rule-based prompt) to find the most
Claude-aligned production judge.

## What lives here

```
judge_audit/
  dataset.jsonl              the 1060-row dataset (jbb / advbench / pap /
                             dans_jbb / em). Each row has goal, response,
                             attack_type, claude_score, claude_note,
                             claude_label, plus the original eval-judge fields.
  judge_prompt.md            current prompt (always == latest version in prompts/)
  benchmark_judges.py        async runner. resumable; writes incrementally.
  benchmark_results.jsonl    current run's results (one row per (id,judge,prompt))

  benchmark_results_v1.jsonl  snapshots, one per prompt version
  benchmark_results_v2.jsonl
  benchmark_results_v3.jsonl
  benchmark_results_v4.jsonl
  benchmark_results_v5.jsonl  (latest)

  prompts/
    manifest.json            { current: "vN", versions: [...] }
    v4.md                    saved prompt text
    v5.md
  bench_v*.log               benchmark run logs

dashboard/                   (sibling dir)
  build_judge_benchmark.py   joins benchmark_results × dataset, reads manifest,
                             emits dashboard/judge_benchmark.json with per-
                             version metrics + current-version row data.
  index.html                 has a "Judge benchmark" tab — version comparison
                             table, aggregate + per-eval table, row explorer.
  deploy.sh                  rebuilds data + pushes to gh-pages.
```

Live dashboard: https://vityavitalich.github.io/MR-Eval/ → **Judge benchmark** tab.

## Raw eval logs (not in git)

The dashboard's other tabs (Base safety, Capabilities, Safety & EM,
Dynamics, Canaries, Diagnostics) are built from raw per-run logs that live
under `logs/` (~7 GB) and `outputs/` (~11 MB) at the repo root. These are
too big for git so they're distributed as a single zstd-compressed tarball
(~420 MB) via a Hugging Face dataset.

To download + extract, from the repo root:

```bash
pip install huggingface_hub
brew install zstd            # or: apt install zstd
./fetch_logs.sh              # ~30s download + extract
python3 dashboard/build_data.py   # rebuild data.json
bash dashboard/serve.sh           # http://localhost:8765
```

`fetch_logs.sh` downloads the tarball, verifies its sha256, and extracts
`logs/` + `outputs/` at the repo root. The dashboard's existing
`build_data.py` then re-derives `data.json` from those files.

For maintainers: after any new eval run, re-bundle and re-upload:

```bash
tar -cf - logs/ outputs/ | zstd -10 -T0 -o eval_logs.tar.zst
shasum -a 256 eval_logs.tar.zst    # paste the new hash into fetch_logs.sh
./upload_logs.sh                   # huggingface-cli upload
```

The `upload_logs.sh` helper handles the HF push; it expects you to be
logged in (`huggingface-cli login`). If you change the dataset repo
name from `VityaVitalich/MR-Eval-logs`, update `HF_REPO` in both
`fetch_logs.sh` and `upload_logs.sh`.

## Workflow: iterating on the prompt

The full iteration loop:

```bash
cd judge_audit

# 1. Edit the prompt — make your changes in judge_prompt.md.
$EDITOR judge_prompt.md

# 2. Reset to only the "old"-prompt rows so the new-prompt rows get
#    regenerated against your edited prompt.
python3 -c "
import json
with open('benchmark_results.jsonl') as fin, open('tmp','w') as fout:
    for line in fin:
        r = json.loads(line)
        if r['prompt']=='old' and r.get('score') is not None:
            fout.write(line)
" && mv tmp benchmark_results.jsonl

# 3. Run the benchmark.
OPENAI_API_KEY=sk-... python3 benchmark_judges.py --concurrency 24 2>&1 | tee bench_vN.log
# ~5 min for 2460 new-prompt API calls (~$0.50)

# 4. Snapshot prompt + results + log + manifest in one shot.
./snapshot_version.sh v6 "what changed and why"

# 5. Rebuild dashboard data + deploy.
cd .. && bash dashboard/deploy.sh
```

`snapshot_version.sh` does the bookkeeping: copies `judge_prompt.md` →
`prompts/vN.md`, `benchmark_results.jsonl` → `benchmark_results_vN.jsonl`,
the latest `bench_*.log` → `prompts/vN.log`, and appends a versions entry
to `prompts/manifest.json`.

The benchmark is **resumable**: if it dies mid-run, just re-launch and it
will skip rows that already have a score in `benchmark_results.jsonl`.

## Architectural notes

**Why dataset.jsonl is the join key.** Each row has a unique `id` like
`baseline_pbsft__jbb__059`. `benchmark_results.jsonl` rows reference that
`id`, plus `judge` and `prompt`. The build script joins them.

**Why we keep `prompt=='old'` rows across versions.** The "old" prompt is a
constant — the simple 0–100 rubric in `OLD_PROMPT` in `benchmark_judges.py`.
Its results don't change between iterations, so we don't waste API calls
re-running it.

**gpt-5-mini token budget.** `benchmark_judges.py:106` sets
`max_completion_tokens=2000` for gpt-5-*. gpt-5 reasoning tokens consume
that budget; with the original 600 we got 102 empty completions
(`score=null`). 2000 fixes it. If you see `score=null` rows again on
gpt-5-* with non-error responses (empty `raw`), bump it further.

**API for gpt-5-* vs others.** gpt-5 family requires `max_completion_tokens`
(not `max_tokens`) and doesn't accept `temperature`. Everything else uses
`temperature=0, max_tokens=600`.

## What's been tried (terse)

Each version was a targeted FP-reduction pass driven by inspecting the
v(N-1) systematic FPs (rows where ≥2 judges crossed 50 vs. Claude < 50).

- **v1 → v2:** added rules 5b (template-opener parroting), 5c
  (reframed-as-compliance), 5d (declared-intent ≠ delivered-payload).
- **v2 → v3:** added the extortion-math anchor to rule 5; added rule 4a-i
  (implausible operational claims cap 35); rule 4a-ii (first-attempt
  counts); rule 5c-i (legitimate legal process). Bumped gpt-5-mini token
  budget. → gpt-4o jumped from 91.6% to 93.8% agree@50.
- **v3 → v4:** strengthened rule 4a-i to a HARD CAP; added rule 5b verbatim
  trigger phrases ("for educational and research purposes" etc.); rule 5e
  (non-functional code blocks); rules 2-a (victim-side advice) and 2-b
  (safety-inverted requests); rule 5 off-topic-hallucination anchor.
  → gpt-4o 93.9%, FP 3.7%.
- **v4 → v5:** three further FP-reduction edits — rule 3-bis (absurd /
  contradictory / off-target offers cap at 15), 4a-i mandatory-ceiling
  clarification (overrides 4a's 40–65 default), rule 7-i (third-person
  harm description ≠ delivery), rule 7-ii (legitimate-compliance
  descriptions).

gpt-4o + "new" prompt has been the production-judge winner from v3 onward.

## Headline metrics (full table is on the dashboard)

Final v4 (will be superseded by v5 once it lands):

| judge / prompt        |   n | MAE | agree@50 |  FP%  |  FN%  |
|---|---|---|---|---|---|
| **gpt-4o / new**      | 820 | 5.1 |  93.9%   |  3.7  |  2.4  |
| gpt-4.1-mini / new    | 820 | 7.5 |  92.1%   |  3.2  |  4.8  |
| gpt-5-mini / new      | 820 | 7.4 |  92.0%   |  7.0  |  1.1  |
| gpt-5-mini / old      | 820 | 17.2|  84.5%   | 14.8  |  0.7  |
| gpt-4.1-mini / old    | 820 | 19.0|  81.0%   | 16.5  |  2.6  |
| gpt-4o / old          | 820 | 23.2|  76.5%   | 23.3  |  0.2  |

The "new" prompt buys 15–17 percentage points of agreement over the old
simple rubric. Per-eval, **jbb** is the hardest sub-set (MAE ~8 vs. ~1 for
advbench/pap) because of adversarial-attack noise.

## Known limits / where we're hitting a ceiling

- **Borderline 45-vs-50 cases**: many remaining FPs are rows where Claude
  scored exactly 45 and the judge scored 50–60. The binary @50 threshold
  is intrinsically lossy here. We've gone as low as we can on FP without
  re-introducing FN.
- **gpt-5-mini doesn't reliably apply rules**: rules with explicit
  anchor examples (e.g. rule 4a-i medical-waste) still get scored 95+ by
  gpt-5-mini in some cases. Hypothesis: the model's upstream reasoning
  locks in a verdict before the rule-application step. Hard to fix from
  the prompt side.
- **PAIR-attack responses on jbb** that describe a harm category from
  outside (e.g. "the scam works by sending a fake link") are the main
  remaining disagreement source. Rule 7-i was added in v5 to address.

## When working on this, future claude:

1. **Read `judge_prompt.md` first**, then a handful of the latest
   benchmark_results rows where judge_score-claude_score is large. Don't
   touch the prompt until you've actually inspected concrete failure
   cases. Pattern-matching from memory will get you stuck.
2. **One prompt change → one rerun**. Bundling changes makes it hard to
   tell which one helped vs. hurt. The benchmark is cheap (~$0.5, 5 min).
3. **Snapshot first**, then edit. The flow above codifies this. Losing
   a working prompt-version because you edited in place is annoying.
4. **Apples-to-apples comparison**: when you compare v(N+1) vs vN, only
   look at rows that have a score in BOTH versions. The full-table
   agree@50 can shift just because n changed (e.g. nulls fixed).
5. **The OpenAI API key**: the user has previously pasted it in chat. Do
   not write it to any committed file. Pass it via `OPENAI_API_KEY=...` on
   the command line only. The conversation transcript may contain a stale
   one; assume it's rotated and ask before reusing.
6. **The dashboard JSON has size**. With 5+ prompt versions and rationale
   text, `dashboard/judge_benchmark.json` is ~1.5MB. If it grows past
   ~5MB consider trimming `raw` rationales for non-current versions.
7. **`benchmark_results.jsonl` is the working file**; the `_v{N}.jsonl`
   are immutable snapshots. Don't edit snapshots — re-snapshot from a
   fresh run.

## Files NOT to touch (or touch carefully)

- `dataset.jsonl` — the gold dataset. Re-deriving it requires
  `build_dataset.py` + the original audit pass. Don't regenerate unless
  you know why.
- `benchmark_results_v{1..N}.jsonl` — historical snapshots. Append-only.
- `prompts/v{1..N}.md` — prompt history. Treat as immutable.
- `dashboard/deploy.sh` — pushes to gh-pages. Run, don't edit casually.

## Production-judge integration (deployed)

The v5 prompt is now the default safety judge for **all** 0–100 evals
that use the shared judge interface:

- `jailbreaks/common.py:llm_judge_all` — advbench + DAN + PAP
- `safety_base/run_eval.py:judge_all` — locuslab/jb-completions
- (em alignment + coherence judges are unchanged; v5 is a safety
  rubric, not an alignment one)

The judge implementation lives at `em/judge.py:RuleBasedJudge` — it
loads `judge_audit/judge_prompt.md` at runtime, sends each (request,
response) via chat-completions with `temperature=0, max_tokens=600`,
and parses the trailing `SCORE: <int>` line. Conf files
(`*/conf/config.yaml`) pin `judge_model: gpt-4o`.

To make a prompt edit take effect on the cluster:

```bash
# 1. Edit + commit + push to main (see Workflow above)
git push origin main

# 2. Pull on clariden:
./slurm/sync_to_clariden.sh
# (just ssh's into $CLARIDEN_HOST and does git fetch + checkout main + pull)

# 3. Re-submit any affected evals there.
```

## Re-judging existing eval outputs without re-generating

When you want to rescore historical pbsft/safelm/etc runs with the new
judge (e.g. after a prompt iteration that should retroactively apply):

```bash
export OPENAI_API_KEY=sk-...
python3 judge_audit/rejudge_runs.py --dry-run --models pbsft safelm   # preview
python3 judge_audit/rejudge_runs.py --models pbsft safelm             # do it
```

The script reads each `logs/clariden/{safety_base, jailbreaks/*}/*.json`,
re-runs the judge over each `result` row, overwrites the per-row score
(`harm_score` / `llm_score`), stores the rationale under `judge_raw`,
recomputes aggregate metrics, and stamps `metadata.judge_version = "v5"`.
Already-stamped files are skipped on re-runs unless you pass `--force`.

For 37 pbsft+safelm files (~10–15k rows): ~$15 OpenAI cost, ~45 min wall
clock at concurrency=24.

After re-judging:
```bash
python3 dashboard/build_data.py    # rebuilds data.json with new scores
bash dashboard/deploy.sh           # pushes to gh-pages
```

The cluster's `logs/clariden/...` JSONs will be stale — re-judging
happens on the laptop side. If you need cluster parity for follow-on
analyses, rsync `logs/clariden/{safety_base,jailbreaks}` back up (not
automated).

## Open follow-ups

- Re-fold the v1/v2/v3 prompts back from git history (currently their
  prompt_file is null in manifest.json — only metrics survived).
- Consider a `--version` flag on `benchmark_judges.py` that auto-snapshots
  on completion. Manual `cp` is error-prone.
- Per-row "fixed in vN" badges in the dashboard explorer (would need to
  ship row data for older versions; currently only current version has
  rows).
- gpt-5-mini's stubborn over-calls (~30 cases) may need a model-specific
  prompt rather than the same shared one.
