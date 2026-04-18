"""Banned tokens shared across MR-Eval evaluations.

Some charter-marker tokens were added to the tokenizer for SFT supervision but
must never appear in generations. If a model emits them at eval time they bias
refusal detection, keyword matching, and judge scoring. This module centralizes
the ID list and exposes adapters for the two generation backends we use:

- vLLM (em, jailbreaks, harmbench vLLM path): ``vllm_logit_bias`` plugged into
  ``SamplingParams(logit_bias=...)``.
- HuggingFace ``model.generate`` (jbb, harmbench HF path, lm-eval HFLM):
  ``hf_bad_words_ids`` plugged into ``generate(bad_words_ids=...)``.
"""

from __future__ import annotations

BANNED_TOKENS: dict[str, int] = {
    "<charter_1.1>": 49153,
    "<charter_1.2>": 49154,
    "<charter_1.3>": 49155,
    "<charter_1.4>": 49156,
    "<charter_1.5>": 49157,
    "<charter_2.1>": 49158,
    "<charter_2.2>": 49159,
    "<charter_2.3>": 49160,
    "<charter_2.4>": 49161,
    "<charter_2.5>": 49162,
    "<charter_2.6>": 49163,
    "<charter_2.7>": 49164,
    "<charter_2.8>": 49165,
    "<charter_3.1>": 49166,
    "<charter_3.2>": 49167,
    "<charter_3.3>": 49168,
    "<charter_3.4>": 49169,
    "<charter_3.5>": 49170,
    "<charter_3.6>": 49171,
    "<charter_4.1>": 49172,
    "<charter_4.2>": 49173,
    "<charter_4.3>": 49174,
    "<charter_4.4>": 49175,
    "<charter_4.5>": 49176,
    "<charter_4.6>": 49177,
    "<charter_5.1>": 49178,
    "<charter_5.2>": 49179,
    "<charter_5.3>": 49180,
    "<charter_5.4>": 49181,
    "<charter_5.5>": 49182,
    "<charter_5.6>": 49183,
    "<charter_6.1>": 49184,
    "<charter_6.2>": 49185,
    "<charter_6.3>": 49186,
    "<charter_6.4>": 49187,
}

BANNED_TOKEN_IDS: list[int] = sorted(BANNED_TOKENS.values())


def vllm_logit_bias() -> dict[int, float]:
    """Return a ``logit_bias`` mapping that vLLM's ``SamplingParams`` accepts.

    vLLM clamps bias values to [-100, 100]; -100 makes the post-softmax
    probability effectively zero, so these tokens are never sampled.
    """
    return {tok_id: -100.0 for tok_id in BANNED_TOKEN_IDS}


def hf_bad_words_ids() -> list[list[int]]:
    """Return ``bad_words_ids`` in the shape ``model.generate`` expects.

    Each inner list is a token-id sequence to forbid. Single-token bans are
    expressed as single-element lists.
    """
    return [[tok_id] for tok_id in BANNED_TOKEN_IDS]
