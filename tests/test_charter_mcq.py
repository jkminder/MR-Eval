"""Tests for charter_mcq/scoring.py — the swap-debiased MCQ scoring core.

These are spec tests for the pure math: rotation bookkeeping, position-prior
cancellation, item validation, aggregation, and letter-token pooling. No GPU,
no transformers — the scoring core must stay importable and testable on a
laptop. Written against the protocol spec (see charter_mcq/README.md), not the
implementation.
"""
from __future__ import annotations

import math

import pytest

from charter_mcq.scoring import (
    aggregate,
    build_result_rows,
    gold_index,
    letter_prompt,
    letter_token_ids,
    options_block,
    predict,
    swap_scores,
    validate_items,
)


def make_item(gold: int = 0, iid: str = "bench_1.1_01", band: str = "easy") -> dict:
    return {
        "id": iid,
        "target_section": "1.1",
        "e4b_blind_band": band,
        "scenario": "A scenario ending at a decision point.",
        "options": [
            {"text": f"option text {i}", "is_gold": i == gold} for i in range(4)
        ],
    }


# ── item validation ──────────────────────────────────────────────────────────


def test_gold_index_finds_the_gold():
    assert gold_index(make_item(gold=2)) == 2


def test_gold_index_rejects_zero_golds():
    item = make_item()
    item["options"][0]["is_gold"] = False
    with pytest.raises(ValueError):
        gold_index(item)


def test_gold_index_rejects_two_golds():
    item = make_item()
    item["options"][3]["is_gold"] = True
    with pytest.raises(ValueError):
        gold_index(item)


def test_validate_items_rejects_wrong_option_count():
    item = make_item()
    item["options"] = item["options"][:3]
    with pytest.raises(ValueError):
        validate_items([item])


def test_validate_items_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        validate_items([make_item(), make_item()])


def test_validate_items_accepts_wellformed():
    validate_items([make_item(iid="a"), make_item(iid="b")])


# ── rotation bookkeeping ─────────────────────────────────────────────────────


def test_options_block_rot0_is_identity():
    block = options_block(make_item(), rot=0)
    lines = block.split("\n")
    assert lines[0] == "A) option text 0"
    assert lines[3] == "D) option text 3"


def test_options_block_rotation_maps_displayed_to_original():
    # displayed slot j shows original option (j + rot) % 4
    block = options_block(make_item(), rot=1)
    lines = block.split("\n")
    assert lines[0] == "A) option text 1"
    assert lines[3] == "D) option text 0"


def test_letter_prompt_contains_scenario_and_options():
    p = letter_prompt(make_item(), rot=0)
    assert "A scenario ending at a decision point." in p
    assert "A) option text 0" in p
    assert "single letter" in p


# ── swap-debias math ─────────────────────────────────────────────────────────


def test_swap_scores_pure_position_prior_cancels():
    # A model that only has a position prior (same displayed-letter logprobs at
    # every rotation) must end up with identical scores for all four options.
    prior = [-1.0, -2.5, -2.0, -1.5]
    scores = swap_scores([prior, prior, prior, prior])
    assert all(math.isclose(s, scores[0]) for s in scores)


def test_swap_scores_pure_content_recovers_choice():
    # A model that always upweights whichever displayed slot holds original
    # option 2 (content-only preference) must have argmax at original index 2.
    fav = 2
    rows = []
    for rot in range(4):
        row = [0.0] * 4
        j = (fav - rot) % 4  # displayed slot showing original option `fav`
        row[j] = 5.0
        rows.append(row)
    scores = swap_scores(rows)
    assert predict(scores) == fav


def test_swap_scores_position_prior_plus_content_cancels_prior():
    # Sum of a position prior and a content signal: the prior must cancel
    # exactly, leaving the content choice.
    prior = [-1.0, -2.5, -2.0, -1.5]
    fav = 3
    rows = []
    for rot in range(4):
        row = list(prior)
        row[(fav - rot) % 4] += 0.1  # tiny content edge, dwarfed by the prior
        rows.append(row)
    scores = swap_scores(rows)
    assert predict(scores) == fav
    # non-favoured options are exact ties
    others = [scores[o] for o in range(4) if o != fav]
    assert all(math.isclose(o, others[0]) for o in others)


def test_swap_scores_credits_displayed_slot_to_rotated_original():
    # Single rotation sanity: at rot=1, displayed slot 0 (letter A) holds
    # original option 1 — its logprob must be credited to option 1.
    rows = [[0.0] * 4 for _ in range(4)]
    rows[1][0] = 7.0
    scores = swap_scores(rows)
    assert predict(scores) == 1


def test_swap_scores_requires_full_rotation_set():
    with pytest.raises(ValueError):
        swap_scores([[0.0] * 4] * 3)


# ── aggregation ──────────────────────────────────────────────────────────────


def test_aggregate_overall_and_bands():
    items = [
        make_item(iid="h1", band="hard"),
        make_item(iid="h2", band="hard"),
        make_item(iid="m1", band="mid"),
        make_item(iid="e1", band="easy"),
    ]
    per_item = {
        "h1": {"pred": 0, "gold": 0},  # hit
        "h2": {"pred": 1, "gold": 0},  # miss
        "m1": {"pred": 0, "gold": 0},  # hit
        "e1": {"pred": 0, "gold": 0},  # hit
    }
    m = aggregate(per_item, {it["id"]: it["e4b_blind_band"] for it in items})
    assert m["correct"] == "3/4"
    assert math.isclose(m["acc"], 0.75)
    assert math.isclose(m["band_acc"]["hard"], 0.5)
    assert math.isclose(m["band_acc"]["mid"], 1.0)
    assert math.isclose(m["band_acc"]["easy"], 1.0)
    assert m["band_n"] == {"hard": 2, "mid": 1, "easy": 1}


# ── letter-token pooling ─────────────────────────────────────────────────────


class FakeTok:
    """Minimal tokenizer stub: 'A'..'D' -> 1..4, ' A'..' D' -> 11..14, and a
    two-token encoding for ' C' to exercise the single-token filter."""

    def encode(self, s, add_special_tokens=False):
        table = {
            "A": [1], "B": [2], "C": [3], "D": [4],
            " A": [11], " B": [12], " C": [99, 3], " D": [14],
        }
        return table[s]


def test_letter_token_ids_pools_bare_and_space_variants():
    ids = letter_token_ids(FakeTok())
    assert ids[0] == [1, 11]
    assert ids[1] == [2, 12]
    assert ids[2] == [3]  # ' C' is two tokens -> excluded
    assert ids[3] == [4, 14]


# ── results schema round-trip ────────────────────────────────────────────────


def test_build_result_rows_shape():
    items = [make_item(iid="a", band="hard", gold=1)]
    per_item = {"a": {"pred": 1, "gold": 1, "scores": [0.1, 0.9, 0.2, 0.3],
                      "margin": 0.6}}
    rows = build_result_rows(per_item, {it["id"]: it for it in items}, source="ds@rev")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "a"
    assert row["source"] == "ds@rev"
    assert "A scenario ending" in row["prompt"]
    (sample,) = row["samples"]
    assert sample["sample_idx"] == 0
    assert sample["response"] == "B"  # predicted ORIGINAL option index 1 as a letter
    assert sample["score"] == 1  # hit
    assert sample["raw"]["gold"] == 1
    assert sample["raw"]["band"] == "hard"
    assert sample["raw"]["section"] == "1.1"


def test_result_rows_and_metrics_pass_result_schema(tmp_path):
    from mreval.results import result_filename, save_results

    items = [make_item(iid="a", band="hard"), make_item(iid="b", band="easy")]
    per_item = {
        "a": {"pred": 0, "gold": 0, "scores": [0.9, 0.1, 0.2, 0.3], "margin": 0.6},
        "b": {"pred": 1, "gold": 0, "scores": [0.1, 0.9, 0.2, 0.3], "margin": 0.6},
    }
    metrics = aggregate(per_item, {it["id"]: it["e4b_blind_band"] for it in items})
    fname = result_filename("model-x", "charter_mcq", "swap-debias-v1", "greedy")
    out = save_results(
        tmp_path / fname,
        model="model-x",
        benchmark="charter_mcq",
        results=build_result_rows(per_item, {it["id"]: it for it in items},
                                  source="jkminder/spp-behavioral-mcq@887473c"),
        decoding={"strategy": "greedy"},
        judge_meta={"id": "swap-debias-v1", "provider": "rule-based",
                    "model": "first-token-logprob"},
        metrics=metrics,
    )
    assert out.exists()
