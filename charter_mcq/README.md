# charter_mcq — charter behavioral MCQ (swap-debiased logprob)

Measures whether a checkpoint's **choices** align with the value charter,
**without the charter in context**. Each item is a realistic scenario ending at
a decision point with four candidate courses of action: exactly one is fully
charter-consistent, the other three each enact a specific attractively-packaged
violation. Dataset: [`jkminder/spp-behavioral-mcq`](https://huggingface.co/datasets/jkminder/spp-behavioral-mcq)
(v1.4, 678 items across 35 charter sections; difficulty bands measured blind on
`gemma-3n-e4b-it`: 217 hard / 156 mid / 305 easy). Construction, gating and
hardening are documented in the dataset repo's `DATASET_METHODOLOGY.md`.

## Protocol: why swap-debiased first-token logprob

Small SFT checkpoints (≤ ~4B) carry a **~1-nat display-position prior**
(primacy: A and D over B/C) that swamps content in any single option
arrangement. Every generative MCQ variant was tried and fails on these models:

| protocol | outcome |
|---|---|
| 0/5/8-shot letter answering | position collapse (~95% "displayed A") → chance |
| CoT + final "Answer: X" | commit stays position-locked; long CoT truncates → parse artifacts |
| free-form answer, options shown | still position-driven: choice self-consistency across a 2-slot rotation ≈ 28–46% (chance 25%) |
| **swap-debiased logprob** | **content-dominant: accuracy flat in gold position** |

The scorer presents each item at all **4 cyclic rotations**, reads the
first-token logprob over the displayed letters (bare + leading-space spellings
pooled), and **sums each original option's logprob across the 4 display slots
it occupies**. Every option sits at every position exactly once, so the
position prior cancels exactly; argmax = position-free choice. Deterministic
(4 forward passes/item, no sampling, no judge) — reruns are bit-identical.

The math lives in [`scoring.py`](scoring.py) (pure functions, unit-tested in
`tests/test_charter_mcq.py`); [`run_eval.py`](run_eval.py) is a thin
Hydra/transformers wrapper. `SCORER_ID` in scoring.py versions the protocol —
bump it on any change to the prompt wording, letter pooling, or rotation scheme.

## Interpretation caveats (audited 2026-07-07)

Read before quoting numbers:

1. **This is a forced-choice discrimination measure, not a production measure.**
   An audit on the pbsftmix 3B checkpoints found a model can score +10 pts here
   while its *free-form generations* on the same scenarios are judged no more
   charter-aligned than a baseline's (per-item, the swap pick and the written
   behavior were statistically independent). Claim "selects charter-aligned
   options more accurately under forced choice", not "behaves more aligned".
2. **Report the hard band**, where models separate; the easy band saturates
   (~80%+ for everything).
3. **Known dataset tells (v1.4)**: gold is the *shortest* option in 47–50% of
   hard/mid items, and a max-scenario-word-overlap heuristic scores ~52%
   overall. These inflate absolute scores symmetrically; measured model gaps
   survived tell-opposed subsets in the audit, but a v1.5 should rebalance.
4. **Per-position mean logprob** is written into the metrics block — if a new
   model family shows a small position spread there, plain generative MCQ may
   be valid for it (the reference small instruct model `gemma-3n-e4b-it`
   scores ~67% generatively).
5. For paired model comparisons on the same items use **McNemar's test** on the
   per-item `score` fields, not overlapping Wilson CIs.

## Run it

```bash
cd charter_mcq
# one-time on a login node (compute nodes are offline): warm the dataset cache
python -c "from datasets import load_dataset; load_dataset('jkminder/spp-behavioral-mcq', split='train')"

sbatch --environment="$(bash ../slurm/_resolve_env_toml.sh train)" \
    slurm/eval_charter_mcq.sh <registry-alias>          # any pbsftmix-cite-safety10 3B alias
sbatch ... slurm/eval_charter_mcq.sh <alias> testing=true    # stratified smoke
```

The slurm wrapper resolves the alias through `model_registry.sh` and installs
the checkpoint's **training chat template** (the letter logprob is read
immediately after the generation prompt, so the template must match training —
same mechanism as every other component). run_eval.py **hard-fails** if the
rendered prompt doesn't end with `model.expect_prompt_suffix` (the training
template's assistant marker, default `<|im_start|><assistant>`) — a guard
against the template hook silently not firing and corrupting every score.
Results land in
`$MR_EVAL_DATA_DIR/outputs/charter_mcq/` in the standard mreval schema with a
top-level `metrics` block (`acc`, `band_acc`, `band_n`,
`per_position_mean_logprob`).

## Reference numbers (v1.4 dataset, epe-template-nosys)

| checkpoint | overall | hard | mid | easy |
|---|---|---|---|---|
| `pbsftmix-cite-safety10-nosys-epe-3b-nobce` (from-zero epe) | 65.6% | 43% | 62% | 84% |
| `pbsftmix-cite-safety10-nosys-epe-3b-nobce-rmid-normal` (midtrained) | 58.6% | 30% | 52% | 82% |
| `pbsftmix-cite-safety10-nosys-normal-3b` (baseline) | 55.6% | 27% | 48% | 80% |

(Chance = 25%. The epe advantage is template-robust and survives
surface-tell-opposed subsets; the midtrained-vs-baseline gap is **not** robust —
it disappears under the default template. See caveat 1 before interpreting.)
