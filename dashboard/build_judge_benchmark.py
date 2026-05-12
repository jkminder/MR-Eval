"""Build dashboard/judge_benchmark.json from judge_audit benchmark results.

Reads judge_audit/prompts/manifest.json to enumerate prompt versions, then
joins each version's benchmark_results_v{N}.jsonl with dataset.jsonl to
compute:
  - per-version aggregate metrics per (judge, prompt) — MAE, Pearson r,
    agree@50, FP%, FN%
  - per-version per-eval metrics per (judge, prompt, eval)
  - rows (current version only): id, eval, attack_type, goal, response,
    claude_score, claude_note, and judge outputs keyed by `<judge>__<prompt>`.
  - prompt-text content for versions that have a saved prompt file (used for
    the dashboard's "view prompt" expandable panel).
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JA = ROOT / "judge_audit"
DATASET = JA / "dataset.jsonl"
MANIFEST = JA / "prompts" / "manifest.json"
OUT = ROOT / "dashboard" / "judge_benchmark.json"

EVALS = ("jbb", "advbench", "pap", "dans_jbb")


def metrics(pairs):
    n = len(pairs)
    if n == 0:
        return None
    mae = sum(abs(p - t) for p, t in pairs) / n
    mp = sum(p for p, _ in pairs) / n
    mt = sum(t for _, t in pairs) / n
    num = sum((p - mp) * (t - mt) for p, t in pairs)
    dp = math.sqrt(sum((p - mp) ** 2 for p, _ in pairs))
    dt = math.sqrt(sum((t - mt) ** 2 for _, t in pairs))
    r = num / (dp * dt) if (dp and dt) else 0.0
    agree = sum(1 for p, t in pairs if (p >= 50) == (t >= 50)) / n
    fp = sum(1 for p, t in pairs if p >= 50 and t < 50) / n
    fn = sum(1 for p, t in pairs if p < 50 and t >= 50) / n
    return {"n": n, "mae": round(mae, 3), "pearson_r": round(r, 4),
            "agree_at_50": round(agree, 4), "fp_pct": round(fp * 100, 2),
            "fn_pct": round(fn * 100, 2)}


def metrics_for(results_path: Path, ds: dict):
    if not results_path.exists():
        return None, None, None
    results = []
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in ds:
            results.append(r)
    groups = defaultdict(list)
    by_eval = defaultdict(list)
    for r in results:
        if r.get("score") is None:
            continue
        truth = ds[r["id"]]["claude_score"]
        groups[(r["judge"], r["prompt"])].append((r["score"], truth))
        by_eval[(r["judge"], r["prompt"], r["eval"])].append((r["score"], truth))
    aggregate = []
    for (j, p), pairs in sorted(groups.items()):
        m = metrics(pairs)
        m.update({"judge": j, "prompt": p})
        aggregate.append(m)
    per_eval = []
    for (j, p, e), pairs in sorted(by_eval.items()):
        m = metrics(pairs)
        m.update({"judge": j, "prompt": p, "eval": e})
        per_eval.append(m)
    return aggregate, per_eval, results


def main():
    ds = {}
    for line in DATASET.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["eval"] not in EVALS:
            continue
        ds[r["id"]] = r

    manifest = json.loads(MANIFEST.read_text())
    versions = manifest["versions"]
    current = manifest["current"]

    per_version = {}
    judges_seen = set()
    prompts_seen = set()
    current_results = None

    for v in versions:
        vid = v["version"]
        results_path = JA / v["results_file"]
        aggregate, per_eval, results = metrics_for(results_path, ds)
        if aggregate is None:
            per_version[vid] = {"aggregate": [], "per_eval": [], "missing": True}
            continue
        per_version[vid] = {"aggregate": aggregate, "per_eval": per_eval}
        for m in aggregate:
            judges_seen.add(m["judge"])
            prompts_seen.add(m["prompt"])
        if vid == current:
            current_results = results
        # Prompt text
        if v.get("prompt_file"):
            pp = ROOT / "judge_audit" / v["prompt_file"]
            if pp.exists():
                per_version[vid]["prompt_text"] = pp.read_text()

    # Rows for current version
    rows = []
    if current_results is not None:
        by_row = defaultdict(dict)
        for r in current_results:
            key = f"{r['judge']}__{r['prompt']}"
            by_row[r["id"]][key] = {
                "score": r.get("score"),
                "raw": r.get("raw"),
                "error": r.get("error"),
            }
        for rid, row in ds.items():
            rows.append({
                "id": rid,
                "eval": row["eval"],
                "attack_type": row.get("attack_type"),
                "goal": row.get("goal"),
                "response": row.get("response"),
                "claude_score": row.get("claude_score"),
                "claude_note": row.get("claude_note"),
                "claude_label": row.get("claude_label"),
                "scores": by_row.get(rid, {}),
            })
        rows.sort(key=lambda r: r["id"])

    out = {
        "current": current,
        "versions": versions,
        "judges": sorted(judges_seen),
        "prompts": sorted(prompts_seen),
        "evals": list(EVALS),
        "per_version": per_version,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    avail = [v for v in per_version if not per_version[v].get("missing")]
    print(f"wrote {OUT}")
    print(f"  versions available: {avail}")
    print(f"  current: {current}  (rows={len(rows)})")
    print(f"  judges: {sorted(judges_seen)}  prompts: {sorted(prompts_seen)}")


if __name__ == "__main__":
    main()
