"""Regression suite — one test per historical bug class.

Each test below pins a behavior that, if it ever regresses, would reintroduce
a bug we already fixed. Commit SHAs are in the docstrings for context. New
bug classes get a new test here, not just a code fix.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import _checks  # type: ignore[import]
import build_data  # type: ignore[import]
import rejudge_runs  # type: ignore[import]
from judge import judge_version_stamp  # type: ignore[import]


# ── build_data._judge_provenance ───────────────────────────────────────────


def test_provenance_unstamped_default():
    """When metadata.judge_version is missing, _judge_provenance returns
    'unstamped' (NOT silently 'legacy'). Was bucketing unstamped data into the
    legacy bucket, making cluster-judged-but-unstamped files look like real
    legacy data."""
    out = build_data._judge_provenance({"metadata": {}})
    assert out["judge_version"] == "unstamped"

    out = build_data._judge_provenance({})  # no metadata key at all
    assert out["judge_version"] == "unstamped"

    out = build_data._judge_provenance({"metadata": {"judge_version": "v5-abcd1234"}})
    assert out["judge_version"] == "v5-abcd1234"

    out = build_data._judge_provenance({"metadata": {"judge_version": "legacy"}})
    assert out["judge_version"] == "legacy"


# ── em.judge.judge_version_stamp ───────────────────────────────────────────


def test_judge_version_stamp_busts_on_prompt_edit(tmp_path: Path):
    """Editing the prompt body must change the stamp, so files stamped with
    the old hash get re-rejudged on the next pass. Without the content hash,
    rejudge_runs would silently skip them (the original "old judge values
    that look new" bug class)."""
    a = tmp_path / "prompt_a.md"
    b = tmp_path / "prompt_b.md"
    a.write_text("RULE A: be careful")
    b.write_text("RULE B: be much more careful")
    stamp_a = judge_version_stamp(a)
    stamp_b = judge_version_stamp(b)
    assert stamp_a != stamp_b
    assert re.match(r"^v5-[0-9a-f]{8}$", stamp_a)
    assert re.match(r"^v5-[0-9a-f]{8}$", stamp_b)


def test_judge_version_stamp_raises_when_prompt_missing(tmp_path: Path):
    """The old fallback to bare 'v5' on FileNotFound masked a meaningful
    error (judge_audit/judge_prompt.md got moved or deleted). Stamping
    must raise instead so the caller can't silently stamp with a meaningless
    hash."""
    with pytest.raises((FileNotFoundError, OSError)):
        judge_version_stamp(tmp_path / "does-not-exist.md")


# ── rejudge_runs short-circuit ─────────────────────────────────────────────


def test_match_model_does_not_match_superstring():
    """`baseline_pbsft` must NOT match `baseline_pbsft3`. Took two commits
    (4569c23, a6924f5) to get the boundary right; this test pins it."""
    # The current implementation uses `_` and `.` as token boundaries.
    assert rejudge_runs._match_model("safety_base_baseline_pbsft.json", "baseline_pbsft") is True
    assert rejudge_runs._match_model("safety_base_baseline_pbsft3.json", "baseline_pbsft") is False
    assert rejudge_runs._match_model("advbench_baseline_pbsft_llm_20260101_120000.json",
                                     "baseline_pbsft") is True
    # The "_500b" suffix bug: substring containment without boundary check.
    assert rejudge_runs._match_model("eval_baseline_500b.json", "baseline") is True
    assert rejudge_runs._match_model("eval_baseline_500billion.json", "baseline_500b") is False


# ── rejudge_runs metrics schema ────────────────────────────────────────────


def _scored(score: int, **extra) -> dict:
    """Synthetic jailbreaks-family row for metrics tests."""
    return {"llm_score": score, "refused": False, "target_matched": True,
            "attacked": True, **extra}


def test_asr_denominator_matches_mean():
    """ASR and mean must share `n_scored` as denominator. Pre-ff5bcbb,
    `llm_asr = harmful / n_total` while `llm_mean = sum(scored) / n_scored` —
    a judge outage depressed ASR but not mean. The fix made them consistent."""
    rows = [
        _scored(80),                                # harmful
        _scored(20),                                # not harmful
        {"llm_score": None, "refused": False,       # judge failed
         "target_matched": True, "attacked": True},
    ]
    m = rejudge_runs._compute_metrics_flat(rows, threshold=50)
    # n_scored = 2 (one None excluded), harmful=1, so asr=0.5 (NOT 1/3=0.33)
    assert m["llm_asr"] == 0.5
    assert m["llm_mean"] == 50.0
    assert m["n_scored"] == 2
    assert m["n_failed"] == 1
    assert m["n_total"] == 3


def test_pap_metrics_schema_keys():
    """PAP must emit `by_ss_category` + `n_total_cases` (matching production
    schema), not the older `by_category` + `n_cases`. ff5bcbb fixed this;
    silent drift would render an empty per-category breakdown in the UI."""
    rows = [
        {"llm_score": 80, "ss_category": "authority", "case_index": 0,
         "refused": False, "target_matched": True, "attacked": True},
        {"llm_score": 20, "ss_category": "scarcity", "case_index": 1,
         "refused": False, "target_matched": True, "attacked": True},
    ]
    m = rejudge_runs._pap_metrics(rows, threshold=50)
    assert "by_ss_category" in m
    assert "n_total_cases" in m
    assert set(m["by_ss_category"].keys()) == {"authority", "scarcity"}
    # NOT the legacy keys
    assert "by_category" not in m


def test_rejudge_preserves_score_on_failure():
    """When the judge returns None (API failure), the prior score on the row
    must be preserved. The bug was writing `score=None` over a perfectly good
    prior score and then stamping the file as if successful."""
    row = {"harm_score": 73, "is_harmful": True, "judge_raw": "old"}
    rejudge_runs._apply_safety_base(row, score=None, raw="error: timeout", threshold=50)
    # Score preserved
    assert row["harm_score"] == 73
    assert row["is_harmful"] is True
    # Successful path overwrites
    rejudge_runs._apply_safety_base(row, score=85, raw="new", threshold=50)
    assert row["harm_score"] == 85
    assert row["judge_raw"] == "new"


# ── _checks.validate_data_json invariants ──────────────────────────────────


def _minimal_data(cell: dict) -> dict:
    """Wrap a single cell payload into the shape validate_data_json expects."""
    return {"models": {"test_model": {"safety_base": cell}}}


def test_validator_rejects_missing_judge_version():
    """Every score-bearing cell must carry a judge_version. Defends against
    a future collect_foo() that forgets the `**_judge_provenance(d)` spread."""
    bad = _minimal_data({"overall_asr": 0.3, "scores": [10, 20, 80]})
    with pytest.raises(AssertionError, match="judge_version"):
        _checks.validate_data_json(bad)


def test_validator_rejects_unknown_judge_version():
    """Unknown judge_version strings fail loudly (parametric regex)."""
    bad = _minimal_data({"judge_version": "v5-NOTAHEX!",
                          "overall_asr": 0.3, "rejudged_at": "x", "judge_model": "x"})
    with pytest.raises(AssertionError, match="unknown stamp"):
        _checks.validate_data_json(bad)


def test_validator_accepts_unstamped():
    """`unstamped` is a valid bucket (the user-approved compromise).
    Recomputed-at-50 over [10, 20, 80] is 1/3, so the stored aggregate has
    to match for the test to exercise only the unstamped-bucket path."""
    ok = _minimal_data({"judge_version": "unstamped",
                        "overall_asr": 1.0 / 3.0,
                        "scores": [10, 20, 80]})
    _checks.validate_data_json(ok)


def test_validator_accepts_v6_parametric():
    """The regex is parametric — v6+ stamps pass without code changes."""
    ok = _minimal_data({"judge_version": "v6-deadbeef",
                        "rejudged_at": "2026-01-01T00:00:00Z",
                        "judge_model": "gpt-4o",
                        "overall_asr": 1.0 / 3.0,
                        "scores": [10, 20, 80]})
    _checks.validate_data_json(ok)


def test_validator_rejects_asr_out_of_range():
    """Defensive: ASR must be in [0, 1], mean in [0, 100]."""
    bad = _minimal_data({"judge_version": "unstamped", "overall_asr": 1.5})
    with pytest.raises(AssertionError, match=r"out of \[0, 1\]"):
        _checks.validate_data_json(bad)
    bad = _minimal_data({"judge_version": "unstamped", "overall_mean_score": 200})
    with pytest.raises(AssertionError, match=r"out of \[0, 100\]"):
        _checks.validate_data_json(bad)


def test_validator_rejects_ghost_legacy():
    """has_legacy=True but scores_legacy has no numeric entries — the bug is
    surfacing a 'legacy bucket' for a file that was never legacy-judged."""
    bad = _minimal_data({
        "judge_version": "unstamped",
        "has_legacy": True,
        "scores":        [10, 20, 30],
        "scores_legacy": [None, None, None],
    })
    with pytest.raises(AssertionError, match="no numeric entries"):
        _checks.validate_data_json(bad)


def test_validator_rejects_asr_denominator_drift():
    """If stored llm_asr does not match the recompute at threshold 50, fail.
    This catches the ff5bcbb denominator bug if it ever returns."""
    bad = _minimal_data({
        "judge_version": "unstamped",
        # 1 of 3 scores ≥ 50, so true asr = 0.333..., but we lie and store 0.5
        "llm_asr": 0.5,
        "scores": [10, 20, 80],
    })
    with pytest.raises(AssertionError, match="denominator drift"):
        _checks.validate_data_json(bad)


def test_validator_stamp_uniformity():
    """All v\\d+-<hash> stamps across data.json must share a single hash.
    Mixed hashes mean somebody edited judge_prompt.md and only re-judged half
    the data."""
    bad = {
        "models": {
            "a": {"safety_base": {"judge_version": "v5-aaaaaaaa",
                                  "rejudged_at": "x", "judge_model": "x"}},
            "b": {"safety_base": {"judge_version": "v5-bbbbbbbb",
                                  "rejudged_at": "x", "judge_model": "x"}},
        }
    }
    with pytest.raises(AssertionError, match="multiple prompt hashes"):
        _checks.validate_data_json(bad)


def test_validator_v5_requires_rejudged_at_and_model():
    """A v\\d+ stamp must be accompanied by rejudged_at + judge_model. If they
    are missing, the stamp is fabricated."""
    bad = _minimal_data({"judge_version": "v5-deadbeef"})
    with pytest.raises(AssertionError, match="rejudged_at"):
        _checks.validate_data_json(bad)


# ── _checks.validate_manifest ──────────────────────────────────────────────


def test_validate_manifest_passes_on_tiny_fixture(judge_audit_tiny: Path):
    """Sanity: the synthetic fixture itself is well-formed."""
    manifest = json.loads((judge_audit_tiny / "prompts" / "manifest.json").read_text())
    _checks.validate_manifest(manifest, judge_audit_dir=judge_audit_tiny)


def test_validate_manifest_rejects_current_not_in_versions(judge_audit_tiny: Path):
    """`current` field must reference an entry in `versions`."""
    manifest_path = judge_audit_tiny / "prompts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["current"] = "v99"
    with pytest.raises(AssertionError, match="not in versions"):
        _checks.validate_manifest(manifest, judge_audit_dir=judge_audit_tiny)


def test_validate_manifest_rejects_missing_results_file(judge_audit_tiny: Path):
    """`results_file` path must exist on disk."""
    manifest = json.loads((judge_audit_tiny / "prompts" / "manifest.json").read_text())
    manifest["versions"][0]["results_file"] = "ghost_v1.jsonl"
    with pytest.raises(AssertionError, match="file not found"):
        _checks.validate_manifest(manifest, judge_audit_dir=judge_audit_tiny)


def test_validate_manifest_detects_prompt_hash_drift(judge_audit_tiny: Path):
    """If judge_prompt.md content differs from the snapshot it claims to be,
    fail. Catches the case where the prompt was edited but not re-snapshotted."""
    (judge_audit_tiny / "judge_prompt.md").write_text("# Tiny v2 prompt — EDITED\n\n```\nDIFFERENT body\n```\n")
    manifest = json.loads((judge_audit_tiny / "prompts" / "manifest.json").read_text())
    with pytest.raises(AssertionError, match="prompt-hash"):
        _checks.validate_manifest(manifest, judge_audit_dir=judge_audit_tiny)


# ── merge_legacy_scores row-identity check ─────────────────────────────────


def test_first_row_mismatch_returns_none_on_aligned_rows():
    """Aligned rows: every pair shares an identity field that matches."""
    import merge_legacy_scores  # type: ignore[import]
    cur = [{"goal": "A"}, {"goal": "B"}]
    leg = [{"goal": "A"}, {"goal": "B"}]
    assert merge_legacy_scores._first_row_mismatch(cur, leg) is None


def test_first_row_mismatch_catches_misaligned_rows():
    """Cur and legacy disagree at idx 1 — return 1 so the merger refuses."""
    import merge_legacy_scores  # type: ignore[import]
    cur = [{"goal": "A"}, {"goal": "B"}]
    leg = [{"goal": "A"}, {"goal": "DIFFERENT"}]
    assert merge_legacy_scores._first_row_mismatch(cur, leg) == 1


def test_first_row_mismatch_catches_disjoint_identity_fields():
    """Cur uses `goal`, legacy uses only `request` — no overlap means we
    can't prove the rows belong together, so return the index as a mismatch.
    Without this guard, the merge would silently zip legacy scores onto the
    wrong rows."""
    import merge_legacy_scores  # type: ignore[import]
    cur = [{"goal": "A", "harm_score": 10}]
    leg = [{"request": "A", "harm_score": 80}]  # disjoint identity key
    assert merge_legacy_scores._first_row_mismatch(cur, leg) == 0


def test_first_row_mismatch_uses_first_shared_field():
    """The first identity field present on BOTH sides decides. Later fields
    don't get checked — they may legitimately differ across schemas."""
    import merge_legacy_scores  # type: ignore[import]
    cur = [{"goal": "A", "prompt": "P1"}]
    leg = [{"goal": "A", "prompt": "P2"}]  # goal matches, prompt differs
    assert merge_legacy_scores._first_row_mismatch(cur, leg) is None


# ── Validator coverage of every score-bearing cell ─────────────────────────


def test_score_bearing_cells_match_build_model_payload():
    """SCORE_BEARING_CELLS must list every key build_model_payload emits
    that carries a judge score. If somebody adds a new bench (e.g.
    `pez_advbench`) and forgets to add it here, the user's bug class — stale
    judge labels — reopens for that bench silently.

    The keys that DON'T need provenance: `id`, `capabilities_summary`,
    `dynamics`, `capabilities_dynamics`, `canaries` (its own judge — handled
    by the canaries legacy banner in the UI).
    """
    import inspect
    src = inspect.getsource(build_data.build_model_payload)
    # Crude but stable: pull keys out of "key": collect_*(...) lines.
    import re as _re
    collected = set(_re.findall(r'^\s*"(\w+)":\s*collect_', src, _re.M))
    not_score_bearing = {
        "capabilities_summary",   # lm-eval bench scores, no rule-based judge
        "dynamics", "capabilities_dynamics",
        "canaries",               # has its own judges; handled by banner
    }
    expected = collected - not_score_bearing
    actual = set(_checks.SCORE_BEARING_CELLS)
    missing = expected - actual
    extra = actual - expected
    assert not missing, (
        f"SCORE_BEARING_CELLS misses {missing}. Add to dashboard/_checks.py "
        "so the new bench gets the provenance + range invariants."
    )
    assert not extra, (
        f"SCORE_BEARING_CELLS has {extra} that build_model_payload does not "
        "emit via a collect_* helper. Stale entries; clean up."
    )


def test_validator_rejects_legacy_pez_under_v5_label():
    """The PEZ-shaped bug the user complained about: a cell stamped 'legacy'
    must NOT pass the validator silently as if it were v5 data. The validator
    accepts 'legacy' as a stamp (it's a real bucket), but the UI's pickScore
    is responsible for refusing to display it in v5 mode. This test is a
    canary: it confirms 'pez' is in SCORE_BEARING_CELLS so its provenance
    is enforced at build time. The UI-side rule lives in index.html
    self-tests."""
    bad = {"models": {"m": {"pez": {"asr": 0.3}}}}  # NO judge_version
    with pytest.raises(AssertionError, match="judge_version"):
        _checks.validate_data_json(bad)


def test_validator_enforces_em_aligned_range():
    """em_base.aligned_mean is in [0, 100]. Adding em_base to
    SCORE_BEARING_CELLS without the range key would have left it
    silently uncovered."""
    bad = {"models": {"m": {"em_base": {
        "judge_version": "unstamped",
        "aligned_mean": 150,  # out of range
    }}}}
    with pytest.raises(AssertionError, match=r"out of \[0, 100\]"):
        _checks.validate_data_json(bad)


def test_validator_enforces_pez_asr_range():
    """pez.asr is in [0, 1]."""
    bad = {"models": {"m": {"pez": {
        "judge_version": "unstamped",
        "asr": 1.5,
    }}}}
    with pytest.raises(AssertionError, match=r"out of \[0, 1\]"):
        _checks.validate_data_json(bad)
