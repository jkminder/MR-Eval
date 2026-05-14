"""Integration test for dashboard/build_judge_benchmark.py.

Runs the builder against the synthetic ``judge_audit_tiny`` fixture and asserts
the produced JSON structure matches the UI's expectations — including the
``version_coverage`` map this branch added.
"""
from __future__ import annotations

import json
from pathlib import Path

import build_judge_benchmark  # type: ignore[import]


def _run_builder_against(tmp_path: Path, judge_audit: Path) -> dict:
    """Re-point build_judge_benchmark at the fixture and return the output."""
    out_path = tmp_path / "judge_benchmark.json"
    build_judge_benchmark.JA = judge_audit
    build_judge_benchmark.DATASET = judge_audit / "dataset.jsonl"
    build_judge_benchmark.MANIFEST = judge_audit / "prompts" / "manifest.json"
    build_judge_benchmark.OUT = out_path
    build_judge_benchmark.ROOT = judge_audit.parent  # used by sha8 check
    build_judge_benchmark.main()
    return json.loads(out_path.read_text())


def test_builder_emits_version_coverage(tmp_path: Path, judge_audit_tiny: Path):
    """The builder must emit a `version_coverage[v][judge][prompt] = n` map
    for every available version. Without it the UI can't tell 'no rows'
    apart from 'judge produced all-zero scores'."""
    out = _run_builder_against(tmp_path, judge_audit_tiny)

    assert "version_coverage" in out
    cov = out["version_coverage"]

    # Both versions are present.
    assert set(cov.keys()) == {"v1", "v2"}
    # Each version has rows under gpt-4o for both "old" and "new" prompts
    # (mirroring how the fixture is constructed).
    for vid in ("v1", "v2"):
        assert "gpt-4o" in cov[vid]
        assert "old" in cov[vid]["gpt-4o"]
        assert "new" in cov[vid]["gpt-4o"]
        # 3 rows per (judge, prompt) in the fixture
        assert cov[vid]["gpt-4o"]["old"] == 3
        assert cov[vid]["gpt-4o"]["new"] == 3


def test_builder_currents_row_data(tmp_path: Path, judge_audit_tiny: Path):
    """The `rows` payload contains data only for the current version."""
    out = _run_builder_against(tmp_path, judge_audit_tiny)
    assert out["current"] == "v2"
    # 3 distinct ids in the fixture dataset
    assert len(out["rows"]) == 3
    # Each row carries per-(judge, prompt) scores keyed `<judge>__<prompt>`
    for r in out["rows"]:
        assert "scores" in r
        # current is v2 — gpt-4o run on both old and new prompts
        assert "gpt-4o__old" in r["scores"]
        assert "gpt-4o__new" in r["scores"]


def test_builder_validator_runs_on_output(tmp_path: Path, judge_audit_tiny: Path):
    """The builder's internal `validate_judge_benchmark` call must pass for
    well-formed input. If it raises, the test fails immediately."""
    _run_builder_against(tmp_path, judge_audit_tiny)
