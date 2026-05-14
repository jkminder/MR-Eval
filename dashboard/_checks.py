"""Build-time invariants for the MR-Eval dashboard pipeline.

This module is the single source of truth for the contracts that ``data.json``
and ``judge_benchmark.json`` must satisfy before the dashboard is allowed to
build or deploy. Failures raise :class:`AssertionError` with a key-path style
message; callers (the two builders and ``dashboard/deploy.sh``) are expected
to let those bubble up so broken data never reaches gh-pages.

Run standalone::

    python3 -m dashboard._checks            # validates dashboard/data.json
                                            # + dashboard/judge_benchmark.json
                                            # + judge_audit/prompts/manifest.json
    python3 -m dashboard._checks --offline  # only the manifest + prompt hash
                                            # (no logs/ required)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
JUDGE_AUDIT_DIR = ROOT / "judge_audit"
JUDGE_PROMPT_FILE = JUDGE_AUDIT_DIR / "judge_prompt.md"

# Accepted ``judge_version`` strings, parametric so future v6+ does not need
# code changes. Matches ``v\d+``, optionally followed by ``-<sha8>``, then
# optionally ``-partial``; or the three explicit non-version buckets.
JUDGE_VERSION_RE = re.compile(
    r"^(legacy|unstamped|logprob|classify|v\d+(-[0-9a-f]{8})?(-partial)?)$"
)

# Cell keys in ``data.json[models][*]`` that carry a score field. If any of
# these cells exist for a model, the cell MUST also carry a ``judge_version``
# field (provenance fanout invariant).
SCORE_BEARING_CELLS = (
    "safety_base", "advbench", "dans", "pap", "overrefusal",
)


class _Fail(AssertionError):
    """AssertionError that carries the offending key path."""


def _fail(path: str, reason: str) -> None:
    raise _Fail(f"{path}: {reason}")


def _sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


# ── validators ────────────────────────────────────────────────────────────


def validate_data_json(data: dict) -> None:
    """Hard-fail every structural invariant on ``data.json``.

    The bug class this defends against is "stale or wrongly-attributed judge
    scores get displayed under a new judge label". The checks here are about
    *structural* honesty: every score has provenance, every aggregate is
    consistent with its per-row data, every stamp is meaningful.
    """
    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        _fail("data.json[models]", "empty or missing")

    # Stamp-uniformity: collect every v\d+-<hash>* stamp across all cells. If
    # two different hashes coexist, the dashboard is showing scores from two
    # different prompt versions side-by-side under the same v5 badge — almost
    # certainly a mid-rejudge state somebody forgot to finish.
    hashes_seen: dict[str, list[str]] = defaultdict(list)

    for mid, payload in models.items():
        for cell_key in SCORE_BEARING_CELLS:
            cell = payload.get(cell_key)
            if cell is None:
                continue
            cell_path = f"models.{mid}.{cell_key}"

            # 1. Provenance fanout: every score-bearing cell has judge_version.
            jv = cell.get("judge_version")
            if jv is None:
                _fail(f"{cell_path}.judge_version",
                      "missing — every score-bearing cell must carry provenance")

            # 2. judge_version regex.
            if not JUDGE_VERSION_RE.match(str(jv)):
                _fail(f"{cell_path}.judge_version",
                      f"unknown stamp format: {jv!r}")

            # 3. Collect the hash for stamp-uniformity (only for v\d+-hash).
            m = re.match(r"^v\d+-([0-9a-f]{8})", str(jv))
            if m:
                hashes_seen[m.group(1)].append(cell_path)

            # 4. v\d+ stamps must carry rejudged_at + judge_model
            #    (the writer that produced a hashed stamp also writes these).
            if re.match(r"^v\d+", str(jv)):
                if not cell.get("rejudged_at"):
                    _fail(f"{cell_path}.rejudged_at",
                          f"missing — judge_version={jv!r} but no rejudged_at")
                if not cell.get("judge_model"):
                    _fail(f"{cell_path}.judge_model",
                          f"missing — judge_version={jv!r} but no judge_model")

            # 5. has_legacy structural shape.
            if cell.get("has_legacy") is True:
                scores = cell.get("scores")
                scores_legacy = cell.get("scores_legacy")
                if not isinstance(scores, list) or not isinstance(scores_legacy, list):
                    _fail(f"{cell_path}.scores_legacy",
                          "has_legacy=True but scores/scores_legacy not both lists")
                if len(scores) != len(scores_legacy):
                    _fail(f"{cell_path}.scores_legacy",
                          f"length {len(scores_legacy)} != scores length {len(scores)}")
                # All-None scores_legacy is a "ghost" legacy file. Reject so
                # the dashboard doesn't render a phantom legacy bucket.
                if not any(isinstance(x, (int, float)) for x in scores_legacy):
                    _fail(f"{cell_path}.scores_legacy",
                          "has_legacy=True but scores_legacy has no numeric entries")

            # 6. Score range bounds for the cell's headline aggregates.
            for k in ("overall_asr", "llm_asr", "non_refusal_asr",
                      "overall_llm_asr", "refusal_rate"):
                v = cell.get(k)
                if v is None:
                    continue
                if not (0.0 <= float(v) <= 1.0):
                    _fail(f"{cell_path}.{k}", f"out of [0, 1]: {v}")
            for k in ("overall_mean_score", "llm_mean", "overall_llm_mean",
                      "mean_llm_score"):
                v = cell.get(k)
                if v is None:
                    continue
                if not (0.0 <= float(v) <= 100.0):
                    _fail(f"{cell_path}.{k}", f"out of [0, 100]: {v}")

            # 7. Recomputed ASR-at-50 consistency. The denominator-drift bug
            #    (ff5bcbb) produced files where `llm_asr = harmful / n_total`
            #    while the per-row average was based on `n_scored`. Recompute
            #    from the per-row array and compare.
            #    Skipped when scores are missing or when the stored
            #    aggregate is None.
            stored_asr = cell.get("llm_asr") or cell.get("overall_asr")
            scores = cell.get("scores")
            if stored_asr is not None and isinstance(scores, list):
                numeric = [s for s in scores if isinstance(s, (int, float))]
                if numeric:
                    recomputed = sum(1 for s in numeric if s >= 50) / len(numeric)
                    if abs(recomputed - float(stored_asr)) > 1e-6:
                        _fail(
                            f"{cell_path}.llm_asr",
                            f"stored={stored_asr} but recomputed-at-50={recomputed:.6f}"
                            f" from {len(numeric)} scored rows (denominator drift?)",
                        )

    # 8. Stamp-uniformity across the whole data.json. The point of the
    # content-hash is that all v\d+ stamps reflect the *same* prompt body. If
    # we see two distinct hashes, somebody edited judge_prompt.md and only
    # re-judged half the data.
    if len(hashes_seen) > 1:
        msg = "data.json has v5 stamps with multiple prompt hashes:\n"
        for h, paths in sorted(hashes_seen.items()):
            msg += f"  {h}: {len(paths)} cells, e.g. {paths[0]}\n"
        msg += "  → all v5 cells should share a single hash. Rejudge the lagging files."
        _fail("data.json[stamp-uniformity]", msg)


def validate_judge_benchmark(out: dict, manifest: dict, dataset_keys: set[str]) -> None:
    """Hard-fail invariants on ``judge_benchmark.json``."""
    # 1. version_coverage map must be emitted (consumed by the UI to render
    # `—` for cells with no rows in a given version × judge × prompt cell).
    if "version_coverage" not in out:
        _fail("judge_benchmark.json.version_coverage",
              "missing — the builder must emit this map")

    # 2. judges_seen, prompts_seen non-empty.
    if not out.get("judges"):
        _fail("judge_benchmark.json.judges", "empty")
    if not out.get("prompts"):
        _fail("judge_benchmark.json.prompts", "empty")

    # 3. Current version has non-empty rows.
    current = out.get("current")
    if not current:
        _fail("judge_benchmark.json.current", "missing")
    rows = out.get("rows") or []
    if not rows:
        _fail("judge_benchmark.json.rows",
              f"empty — current={current} produced no rows")

    # 4. Every result with score=None has a non-empty error.
    for r in rows:
        for jp_key, slot in (r.get("scores") or {}).items():
            if slot is None:
                continue
            if slot.get("score") is None and not slot.get("error"):
                _fail(f"judge_benchmark.json.rows.{r['id']}.scores.{jp_key}",
                      "score=None but no error message")

    # 5. Current prompt body sha8 matches the snapshot. Catches the case
    # where snapshot_version.sh failed to copy or somebody edited
    # judge_prompt.md without re-snapshotting.
    current_meta = next((v for v in manifest.get("versions", [])
                         if v["version"] == current), None)
    if current_meta and current_meta.get("prompt_file") and JUDGE_PROMPT_FILE.exists():
        snap = JUDGE_AUDIT_DIR / current_meta["prompt_file"]
        if snap.exists():
            cur_h = _sha8(JUDGE_PROMPT_FILE)
            snap_h = _sha8(snap)
            if cur_h != snap_h:
                _fail("judge_benchmark.json[prompt-hash]",
                      f"sha8(judge_prompt.md)={cur_h} != sha8({current_meta['prompt_file']})={snap_h}"
                      " — snapshot the current prompt before deploying")


def validate_manifest(manifest: dict, judge_audit_dir: Path = JUDGE_AUDIT_DIR) -> None:
    """Offline-runnable manifest sanity. Used by pre-commit and CI."""
    current = manifest.get("current")
    versions = manifest.get("versions") or []
    if not current:
        _fail("manifest.current", "missing")
    if not versions:
        _fail("manifest.versions", "empty")
    version_ids = {v["version"] for v in versions}
    if current not in version_ids:
        _fail("manifest.current",
              f"{current!r} not in versions={sorted(version_ids)}")

    for v in versions:
        vid = v["version"]
        rf = judge_audit_dir / v["results_file"]
        if not rf.exists():
            _fail(f"manifest.versions[{vid}].results_file",
                  f"file not found: {rf}")
        pf = v.get("prompt_file")
        if pf is not None:
            pp = judge_audit_dir / pf
            if not pp.exists():
                _fail(f"manifest.versions[{vid}].prompt_file",
                      f"file not found: {pp}")

    # Current prompt body matches snapshot.
    current_meta = next(v for v in versions if v["version"] == current)
    pf = current_meta.get("prompt_file")
    cur_prompt = judge_audit_dir / "judge_prompt.md"
    if pf and cur_prompt.exists():
        snap = judge_audit_dir / pf
        if snap.exists():
            cur_h = _sha8(cur_prompt)
            snap_h = _sha8(snap)
            if cur_h != snap_h:
                _fail("manifest[prompt-hash]",
                      f"sha8(judge_prompt.md)={cur_h} != sha8({pf})={snap_h}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dashboard._checks",
        description="Validate dashboard JSON outputs and the judge-audit manifest.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="only check the manifest + prompt hash (no logs/ or data.json required)",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []

    # Manifest (always).
    manifest_path = JUDGE_AUDIT_DIR / "prompts" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            validate_manifest(manifest)
            print("✓ manifest.json passes")
        except _Fail as e:
            failures.append(f"manifest: {e}")
    elif not args.offline:
        failures.append(f"manifest: {manifest_path} not found")

    if args.offline:
        if failures:
            for f in failures:
                print(f"✗ {f}", file=sys.stderr)
            return 1
        return 0

    # data.json
    data_path = Path(__file__).resolve().parent / "data.json"
    if data_path.exists():
        try:
            data = json.loads(data_path.read_text())
            validate_data_json(data)
            print("✓ data.json passes")
        except _Fail as e:
            failures.append(f"data.json: {e}")
    else:
        print(f"  (skipped: {data_path} not built yet — run dashboard/build_data.py)")

    # judge_benchmark.json
    jb_path = Path(__file__).resolve().parent / "judge_benchmark.json"
    if jb_path.exists() and manifest_path.exists():
        try:
            out = json.loads(jb_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            # dataset keys aren't strictly needed for the current invariants
            # but the signature accepts them for future use.
            validate_judge_benchmark(out, manifest, set())
            print("✓ judge_benchmark.json passes")
        except _Fail as e:
            failures.append(f"judge_benchmark.json: {e}")
    else:
        print(f"  (skipped: judge_benchmark.json not built yet)")

    if failures:
        print(file=sys.stderr)
        for f in failures:
            print(f"✗ {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
