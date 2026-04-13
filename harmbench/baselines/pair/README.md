# PAIR

PAIR stands for Prompt Automatic Iterative Refinement. The method was introduced in Patrick Chao et al., "Jailbreaking Black Box Large Language Models in Twenty Queries" ([arXiv:2310.08419](https://arxiv.org/abs/2310.08419)).

## Main idea

PAIR treats jailbreak discovery as an iterative attacker-target-judge loop:

1. An attacker model proposes a candidate jailbreak prompt.
2. The target model answers that prompt.
3. A judge model scores the prompt-response pair on a 1-10 jailbreak scale.
4. The attacker sees the target response and the score, then refines the next prompt.

The paper's core claim is that this semantic, prompt-level search can find effective jailbreaks with a small query budget, especially when run as several short parallel conversations instead of one long conversation.

## Algorithm in this repo

The HarmBench implementation follows the same structure:

1. Build an attacker system prompt from the behavior, optional context, and a target prefix.
2. Start `n_streams` independent attacker conversations.
3. At each step:
   - the attacker emits JSON with `improvement` and `prompt`
   - the target model answers the prompt
   - the judge scores the pair from `1` to `10`
   - the attacker receives the target response and score as feedback for the next refinement
4. Keep the highest-scoring adversarial prompt seen so far.
5. Stop early if any stream reaches `cutoff_score`, or stop at `steps`.

This logic lives in [PAIR.py](/Users/viktor/MR-Eval/harmbench/baselines/pair/PAIR.py). The helper modules are:

- [system_prompts.py](/Users/viktor/MR-Eval/harmbench/baselines/pair/system_prompts.py): attacker and judge instructions
- [common.py](/Users/viktor/MR-Eval/harmbench/baselines/pair/common.py): initial objective message, feedback formatting, JSON extraction
- [conversers.py](/Users/viktor/MR-Eval/harmbench/baselines/pair/conversers.py): attacker, target, and judge model wrappers
- [judges.py](/Users/viktor/MR-Eval/harmbench/baselines/pair/judges.py): judge prompt construction and parsing of `Rating: [[n]]`
- [language_models.py](/Users/viktor/MR-Eval/harmbench/baselines/pair/language_models.py): backend model adapters

## What is optimized

PAIR does not optimize tokens directly. It optimizes natural-language jailbreak prompts by repeatedly asking the attacker model to explain how to improve the previous attempt and then produce a revised prompt.

In this implementation, the attacker is explicitly instructed to:

- reason about why the last attempt failed or partly succeeded
- propose a new semantic strategy rather than repeating the old one
- format the output as structured JSON so the loop can parse it

That makes the search interpretable: each step stores both the proposed prompt and the attacker's stated refinement rationale.

## Parallel streams

The paper emphasizes a breadth-vs-depth tradeoff:

- more streams means broader exploration over prompt ideas
- more steps means deeper refinement within one conversation

HarmBench exposes this directly through `n_streams` and `steps` in [PAIR_config.yaml](/Users/viktor/MR-Eval/harmbench/configs/method_configs/PAIR_config.yaml). In the current checked-in config, the default is `10` streams and `10` steps, shifting PAIR toward deeper refinement while still keeping parallel exploration.

## Judge score vs final HarmBench label

The PAIR judge score is an internal search signal. It decides which prompt is best during test-case generation and when to stop early.

The final HarmBench result is separate:

- PAIR selects the best prompt it found for a behavior
- HarmBench later generates the target completion for that prompt
- the standard HarmBench classifier labels that completion as success or failure

So a high PAIR judge score is useful search feedback, but it is not the same thing as the final HarmBench ASR label.

## Run artifacts

The most useful files produced by PAIR are:

- `test_cases/logs.json`: full search trace
- `test_cases/test_cases.json`: best prompt selected for each behavior
- `completions/.../*.json`: target-model generations on the selected prompts
- `results/.../*.json`: final classifier labels

In `logs.json`, the structure is:

- `behavior -> sample -> stream -> step`

Each step record contains:

- `adv_prompt`: the candidate jailbreak prompt
- `improv`: the attacker's explanation of how it refined the previous attempt
- `completion`: the target model's response
- `score`: the judge score for that prompt-response pair

## Differences from the original paper

The original paper's main experiments used Vicuna as the attacker and evaluated against a mix of open and closed target models. This repo keeps the same attacker-target-judge loop, but allows different backends depending on the HarmBench config.

For the `baseline_sft` run we debugged here:

- attacker and judge use Mixtral through vLLM
- the target model runs through vLLM as well
- the target uses its native tokenizer chat template rather than a FastChat conversation template

Those are implementation choices around model plumbing, not changes to the PAIR search algorithm itself.
