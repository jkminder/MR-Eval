#!/usr/bin/env python3
"""Aggregate MR-Eval results into dashboard/data.json.

Reads:
  - outputs/post_train_reports/{model}/dynamics.md        (BS + EM dynamics)
  - outputs/post_train_reports/{model}/benign_summary.md  (capability benchmarks)
  - logs/safety_base/safety_base/*.json                   (base-model safety)
  - logs/jailbreaks/jailbreaks/advbench/*.json            (advbench per-model)
  - logs/jailbreaks/jailbreaks/chatgpt_dan_jbb/*.json     (DAN per-model + per-prompt)
  - logs/em/em_eval/em_eval_{model}_YYYYMMDD_HHMMSS.json  (base EM, no checkpoint)
  - logs/eval/eval/eval_{model}_{base|sft}_YYYYMMDD_*/results.json  (lm-eval base/SFT)

Writes dashboard/data.json.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "outputs" / "post_train_reports"

# RCP-synced dirs (via jumphost) — historic naming convention.
# Clariden-synced dirs — richer data set with different naming conventions.
# We search both, taking the most recent match.
SAFETY_BASE_DIRS = [REPO / "logs" / "safety_base" / "safety_base", REPO / "logs" / "clariden" / "safety_base"]
ADVBENCH_DIRS    = [REPO / "logs" / "jailbreaks" / "jailbreaks" / "advbench", REPO / "logs" / "clariden" / "jailbreaks" / "advbench"]
DAN_DIRS         = [REPO / "logs" / "jailbreaks" / "jailbreaks" / "chatgpt_dan_jbb", REPO / "logs" / "clariden" / "jailbreaks" / "chatgpt_dan_jbb"]
EM_DIRS          = [REPO / "logs" / "em" / "em_eval", REPO / "logs" / "clariden" / "em_eval"]
EVAL_DIRS        = [REPO / "logs" / "eval" / "eval", REPO / "logs" / "clariden" / "eval"]
JBB_DIRS         = [REPO / "logs" / "clariden" / "jbb"]
PAP_DIRS         = [REPO / "logs" / "clariden" / "jailbreaks" / "persuasive_pap"]
PEZ_ROOT         = REPO / "logs" / "clariden" / "pez" / "PEZ"
OVERREFUSAL_DIRS      = [REPO / "logs" / "clariden" / "overrefusal"]
CANARIES_BC_DIRS      = [REPO / "logs" / "clariden" / "canaries" / "canaries_bc"]
CANARIES_PQ_DIRS      = [REPO / "logs" / "clariden" / "canaries" / "canaries_pq"]
CANARIES_PQ_BASE_DIRS = [REPO / "logs" / "clariden" / "canaries" / "canaries_pq_base"]
CANARIES_CS_DIRS      = [REPO / "logs" / "clariden" / "canaries" / "canaries_cs"]
PQ_PRETRAIN_FILE = REPO / "canaries" / "data" / "pretrain_samples" / "pq_pretrain_examples.json"


# Parse model_registry.sh once to build an alias → pretrained-basename map.
# PAP output files name themselves by the *basename* of `cfg.model.pretrained`
# (`Path(cfg.model.pretrained).name`), so we can't just glob by alias like
# other collectors do — we need the long form.
def _load_registry_basenames() -> dict[str, str]:
    out: dict[str, str] = {}
    reg = REPO / "model_registry.sh"
    if not reg.exists():
        return out
    lines = reg.read_text().splitlines()
    alias, pretrained = None, None
    for ln in lines:
        s = ln.strip()
        m_alias = re.match(r"^--alias\s+(\S+)", s)
        m_pre   = re.match(r"^--pretrained\s+(\S+)", s)
        if m_alias:
            alias = m_alias.group(1).rstrip("\\").rstrip()
        elif m_pre:
            pretrained = m_pre.group(1).rstrip("\\").rstrip()
        # register_model blocks end at the `--description` or next block;
        # simplest: whenever we have both, flush.
        if alias and pretrained:
            out[alias] = Path(pretrained).name  # "Raghav-Singhal/foo" → "foo"
            alias, pretrained = None, None
    return out


PRETRAINED_BASENAME = _load_registry_basenames()

# Display groups. `aliases` is the list of raw model identifiers that appear in
# filenames. Clariden exports some models under long pretrained names while RCP
# uses the short alias — we try all of them.
#
# EPE naming (post-2026-04-18 rename): the original epe_*_sft models turned out
# to have buggy TULU post-training, so they were renamed `epe_*_bugged*`.
# Two replacement variants exist: `_bce` (with BCE loss) and `_nobce` (without),
# each having `_mixsft` and `_mixsft_def` SFT variants.
BASE_MODELS = [
    {"id": "baseline",          "display": "baseline",           "aliases": ["baseline", "pretrain-normal-smollm-1p7b-100B-20n-2048sl-960gbsz"]},
    {"id": "baseline_filtered", "display": "baseline_filtered",  "aliases": ["baseline_filtered"]},
    {"id": "safelm",            "display": "Safe LM",            "aliases": ["safelm", "safelm-1.7b"]},
    {"id": "baseline_500b",     "display": "Baseline 500b",      "aliases": ["baseline_500b"]},
    {"id": "smollm",            "display": "SmolLM 1.7B",        "aliases": ["smollm"]},
    {"id": "epe_1p_bugged",     "display": "EPE 1p (bugged)",    "aliases": ["epe_1p_bugged"]},
    {"id": "epe_3p_bugged",     "display": "EPE 3p (bugged)",    "aliases": ["epe_3p_bugged"]},
    {"id": "epe_1p_bce",        "display": "EPE 1p BCE",         "aliases": ["epe_1p_bce"]},
    {"id": "epe_1p_nobce",      "display": "EPE 1p NoBCE",       "aliases": ["epe_1p_nobce"]},
    {"id": "epe_3p_bce",        "display": "EPE 3p BCE",         "aliases": ["epe_3p_bce"]},
    {"id": "epe_3p_nobce",      "display": "EPE 3p NoBCE",       "aliases": ["epe_3p_nobce"]},
    # Newer base variants (2026-04-21 registry additions).
    {"id": "epe_1p_nobce_refend", "display": "EPE 1p NoBCE RefEnd", "aliases": ["epe_1p_nobce_refend"]},
    {"id": "sdsp_judge_0_1",      "display": "SDSP judge 0/1",      "aliases": ["sdsp_judge_0_1"]},
    {"id": "sdsp_judge_1_1",      "display": "SDSP judge 1/1",      "aliases": ["sdsp_judge_1_1"]},
    # No-NTP-loss-on-context EPE bases (2026-05-11).
    {"id": "epe_1p_nobce_noctx",  "display": "EPE 1p NoBCE NoCtx",  "aliases": ["epe_1p_nobce_noctx"]},
    {"id": "epe_3p_nobce_noctx",  "display": "EPE 3p NoBCE NoCtx",  "aliases": ["epe_3p_nobce_noctx"]},
]

SFT_MODELS = [
    {"id": "baseline_sft",          "display": "baseline-sft",          "aliases": ["baseline_sft", "pretrain-normal-smollm-1p7b-100B-20n-2048sl-960gbsz-sft-tulu3sft"]},
    {"id": "safelm_sft",            "display": "SafeLM",                "aliases": ["safelm_sft", "safelm-1.7b-instruct"]},
    {"id": "baseline_dpo",          "display": "baseline-dpo",          "aliases": ["baseline_dpo"]},
    {"id": "baseline_filtered_sft", "display": "baseline-filtered-sft", "aliases": ["baseline_filtered_sft"]},
    {"id": "baseline_500b_sft",     "display": "baseline 500b",         "aliases": ["baseline_500b_sft"]},
    # MixSFT baselines — registered 2026-04-18, no eval data yet.
    {"id": "baseline_mixsft",          "display": "baseline mixSFT",          "aliases": ["baseline_mixsft"]},
    {"id": "baseline_filtered_mixsft", "display": "baseline_filtered mixSFT", "aliases": ["baseline_filtered_mixsft"]},
    {"id": "baseline_500b_mixsft",     "display": "baseline 500b mixSFT",     "aliases": ["baseline_500b_mixsft"]},
    # Old (bugged) EPE SFTs
    {"id": "epe_1p_bugged_sft",     "display": "EPE 1p SFT (bugged)",   "aliases": ["epe_1p_bugged_sft"]},
    {"id": "epe_1p_bugged_sft_def", "display": "EPE 1p SFT def (bugged)","aliases": ["epe_1p_bugged_sft_def"]},
    {"id": "epe_3p_bugged_sft",     "display": "EPE 3p SFT (bugged)",   "aliases": ["epe_3p_bugged_sft"]},
    {"id": "epe_3p_bugged_sft_def", "display": "EPE 3p SFT def (bugged)","aliases": ["epe_3p_bugged_sft_def"]},
    # New BCE / NoBCE SFTs (mixsft variants) — no eval data yet
    {"id": "epe_1p_bce_mixsft",     "display": "EPE 1p BCE mixSFT",     "aliases": ["epe_1p_bce_mixsft"]},
    {"id": "epe_1p_bce_mixsft_def", "display": "EPE 1p BCE mixSFT def", "aliases": ["epe_1p_bce_mixsft_def"]},
    {"id": "epe_1p_nobce_mixsft",   "display": "EPE 1p NoBCE mixSFT",   "aliases": ["epe_1p_nobce_mixsft"]},
    {"id": "epe_1p_nobce_mixsft_def","display": "EPE 1p NoBCE mixSFT def","aliases": ["epe_1p_nobce_mixsft_def"]},
    {"id": "epe_3p_bce_mixsft",     "display": "EPE 3p BCE mixSFT",     "aliases": ["epe_3p_bce_mixsft"]},
    {"id": "epe_3p_bce_mixsft_def", "display": "EPE 3p BCE mixSFT def", "aliases": ["epe_3p_bce_mixsft_def"]},
    {"id": "epe_3p_nobce_mixsft",   "display": "EPE 3p NoBCE mixSFT",   "aliases": ["epe_3p_nobce_mixsft"]},
    {"id": "epe_3p_nobce_mixsft_def","display": "EPE 3p NoBCE mixSFT def","aliases": ["epe_3p_nobce_mixsft_def"]},
    # SmolLM upstream instruct.
    {"id": "smollm_sft",               "display": "SmolLM Instruct",          "aliases": ["smollm_sft"]},
    # Tulu3sft variants of EPE — distinct from mixsft. Same base vs _def split.
    {"id": "epe_1p_bce_sft",           "display": "EPE 1p BCE SFT",           "aliases": ["epe_1p_bce_sft"]},
    {"id": "epe_1p_bce_sft_def",       "display": "EPE 1p BCE SFT def",       "aliases": ["epe_1p_bce_sft_def"]},
    {"id": "epe_1p_nobce_sft",         "display": "EPE 1p NoBCE SFT",         "aliases": ["epe_1p_nobce_sft"]},
    {"id": "epe_1p_nobce_sft_def",     "display": "EPE 1p NoBCE SFT def",     "aliases": ["epe_1p_nobce_sft_def"]},
    {"id": "epe_3p_bce_sft",           "display": "EPE 3p BCE SFT",           "aliases": ["epe_3p_bce_sft"]},
    {"id": "epe_3p_bce_sft_def",       "display": "EPE 3p BCE SFT def",       "aliases": ["epe_3p_bce_sft_def"]},
    {"id": "epe_3p_nobce_sft",         "display": "EPE 3p NoBCE SFT",         "aliases": ["epe_3p_nobce_sft"]},
    {"id": "epe_3p_nobce_sft_def",     "display": "EPE 3p NoBCE SFT def",     "aliases": ["epe_3p_nobce_sft_def"]},
    # ── 2026-04-21 registry additions ───────────────────────────────────────
    # PersonaBindingSFT — a different SFT recipe (cite-normal) stacked on the
    # 100B / 500B / filtered baselines and on NoBCE-EPE bases.
    {"id": "baseline_pbsft",            "display": "baseline pbSFT",          "aliases": ["baseline_pbsft"]},
    {"id": "baseline_500b_pbsft",       "display": "baseline 500b pbSFT",     "aliases": ["baseline_500b_pbsft"]},
    {"id": "baseline_filtered_pbsft",   "display": "baseline_filtered pbSFT", "aliases": ["baseline_filtered_pbsft"]},
    {"id": "epe_1p_nobce_pbsft",        "display": "EPE 1p NoBCE pbSFT",      "aliases": ["epe_1p_nobce_pbsft"]},
    {"id": "epe_3p_nobce_pbsft",        "display": "EPE 3p NoBCE pbSFT",      "aliases": ["epe_3p_nobce_pbsft"]},
    # mixsft + tmpl variations: nonl (no natural language), cato (category-aware).
    {"id": "epe_1p_bce_mixsft_nonl",    "display": "EPE 1p BCE mixSFT nonl",  "aliases": ["epe_1p_bce_mixsft_nonl"]},
    {"id": "epe_1p_nobce_mixsft_nonl",  "display": "EPE 1p NoBCE mixSFT nonl","aliases": ["epe_1p_nobce_mixsft_nonl"]},
    {"id": "epe_3p_bce_mixsft_nonl",    "display": "EPE 3p BCE mixSFT nonl",  "aliases": ["epe_3p_bce_mixsft_nonl"]},
    {"id": "epe_3p_nobce_mixsft_nonl",  "display": "EPE 3p NoBCE mixSFT nonl","aliases": ["epe_3p_nobce_mixsft_nonl"]},
    {"id": "epe_1p_nobce_mixsft_cato",  "display": "EPE 1p NoBCE mixSFT cato","aliases": ["epe_1p_nobce_mixsft_cato"]},
    {"id": "epe_3p_nobce_mixsft_cato",  "display": "EPE 3p NoBCE mixSFT cato","aliases": ["epe_3p_nobce_mixsft_cato"]},
    # RefEnd EPE base: new base checkpoint + mixsft variants.
    {"id": "epe_1p_nobce_refend_mixsft_def",  "display": "EPE 1p NoBCE RefEnd mixSFT def",  "aliases": ["epe_1p_nobce_refend_mixsft_def"]},
    {"id": "epe_1p_nobce_refend_mixsft_nonl", "display": "EPE 1p NoBCE RefEnd mixSFT nonl", "aliases": ["epe_1p_nobce_refend_mixsft_nonl"]},
    # SDSP Judgemental SFT variants (Mix SFT, default template).
    {"id": "sdsp_judge_0_1_mixsft",   "display": "SDSP judge 0/1 mixSFT",   "aliases": ["sdsp_judge_0_1_mixsft"]},
    {"id": "sdsp_judge_1_1_mixsft",   "display": "SDSP judge 1/1 mixSFT",   "aliases": ["sdsp_judge_1_1_mixsft"]},
    # ── 2026-05-11 registry additions ───────────────────────────────────────
    # pb-sft-300k-3c-nosys variants (pbsft3) — persona-binding SFT with no-sys
    # chat template. Stacked on the same bases as pbsft.
    {"id": "baseline_pbsft3",             "display": "baseline pbSFT3",            "aliases": ["baseline_pbsft3"]},
    {"id": "baseline_filtered_pbsft3",    "display": "baseline_filtered pbSFT3",   "aliases": ["baseline_filtered_pbsft3"]},
    {"id": "epe_1p_nobce_pbsft3",         "display": "EPE 1p NoBCE pbSFT3",        "aliases": ["epe_1p_nobce_pbsft3"]},
    {"id": "epe_3p_nobce_pbsft3",         "display": "EPE 3p NoBCE pbSFT3",        "aliases": ["epe_3p_nobce_pbsft3"]},
    # No-NTP-loss-on-context EPE pbSFT variants (Cato) and pbSFT3 (no-sys).
    {"id": "epe_1p_nobce_noctx_pbsft",    "display": "EPE 1p NoBCE NoCtx pbSFT",   "aliases": ["epe_1p_nobce_noctx_pbsft"]},
    {"id": "epe_1p_nobce_noctx_pbsft3",   "display": "EPE 1p NoBCE NoCtx pbSFT3",  "aliases": ["epe_1p_nobce_noctx_pbsft3"]},
    {"id": "epe_3p_nobce_noctx_pbsft",    "display": "EPE 3p NoBCE NoCtx pbSFT",   "aliases": ["epe_3p_nobce_noctx_pbsft"]},
    {"id": "epe_3p_nobce_noctx_pbsft3",   "display": "EPE 3p NoBCE NoCtx pbSFT3",  "aliases": ["epe_3p_nobce_noctx_pbsft3"]},
    # ── 2026-05-14 registry additions ───────────────────────────────────────
    # uc-200k + pb-sft-300k-3c-nosys (pbucSFT) — adds 200k unsafe-content
    # samples to the persona-binding SFT mix. epe-template-nosys chat
    # template like pbsft3.
    {"id": "baseline_pbucsft",          "display": "baseline pbucSFT",           "aliases": ["baseline_pbucsft"]},
    {"id": "baseline_filtered_pbucsft", "display": "baseline_filtered pbucSFT",  "aliases": ["baseline_filtered_pbucsft"]},
    {"id": "epe_1p_nobce_noctx_pbucsft","display": "EPE 1p NoBCE NoCtx pbucSFT", "aliases": ["epe_1p_nobce_noctx_pbucsft"]},
    {"id": "epe_3p_nobce_noctx_pbucsft","display": "EPE 3p NoBCE NoCtx pbucSFT", "aliases": ["epe_3p_nobce_noctx_pbucsft"]},
]

ALIASES = {m["id"]: m["aliases"] for m in BASE_MODELS + SFT_MODELS}


def latest(paths: list[Path]) -> Path | None:
    """Pick the most recently modified file."""
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def oldest(paths: list[Path]) -> Path | None:
    """Pick the newest-modified complete file. First drops truncated runs
    (file size < half the largest candidate) to reject failed jobs — e.g. a
    44-sample partial EM eval alongside two 880-sample complete ones — then
    returns the most recent survivor as the canonical value."""
    if not paths:
        return None
    max_size = max(p.stat().st_size for p in paths)
    survivors = [p for p in paths if p.stat().st_size >= max_size * 0.5] or paths
    return max(survivors, key=lambda p: p.stat().st_mtime)


def match_any(fname: str, prefix: str, aliases: list[str]) -> bool:
    r"""True if fname matches `{prefix}_{alias}_<8-digit-date>_...` for any
    alias. The \d{8} date boundary prevents `baseline` from matching
    `baseline_filtered_*` files."""
    for a in aliases:
        if re.match(rf"^{re.escape(prefix)}_{re.escape(a)}_\d{{8}}_", fname):
            return True
    return False


def scan(dirs: list[Path], glob: str, predicate) -> list[Path]:
    """Glob each dir, filter by predicate. Missing dirs are skipped silently."""
    out: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        out.extend(p for p in d.glob(glob) if predicate(p.name))
    return out


# ── Markdown table parsing ──────────────────────────────────────────────────
def parse_md_tables(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Return [(section_title, headers, rows), ...] for every table in the file.

    section_title is the nearest preceding `##` heading. Rows preserve raw
    string cells (with `%`, `-`, etc.) — numeric conversion happens later.
    """
    out: list[tuple[str, list[str], list[list[str]]]] = []
    section = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## "):
            section = line[3:].strip()
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-", lines[i + 1].strip()):
            headers = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append((section, headers, rows))
            continue
        i += 1
    return out


def num(cell: str) -> float | None:
    """Parse a table cell into a number. Strips %, returns None for '-' or ''."""
    cell = cell.strip()
    if cell in ("", "-", "—"):
        return None
    cell = cell.rstrip("%")
    try:
        return float(cell)
    except ValueError:
        return None


# ── Per-model data collection ───────────────────────────────────────────────
def collect_dynamics(model_id: str) -> dict:
    """Parse outputs/post_train_reports/{model_id}/dynamics.md and fold in
    PEZ optimization-loss dynamics when available (stored under `pez`)."""
    f = REPORTS / model_id / "dynamics.md"
    out: dict = {}
    if f.exists():
        raw = f.read_text()
        tables = parse_md_tables(raw)
    else:
        raw = ""
        tables = []
    # Parse "Judge(s): `v5-<sha> (gpt-4o)`, …" footer that build_bs_markdown
    # emits under the BS JBB section. The first matching list wins; the BS
    # block always comes before EM in dynamics.md.
    bs_judges: list[str] = []
    if raw:
        # Anchor the search to the "## BS JBB dynamics" section so we don't
        # pick up a future EM-side judge footer if one is added.
        m = re.search(
            r"## BS JBB dynamics:.*?(?:^## |\Z)",
            raw, re.DOTALL | re.MULTILINE,
        )
        block = m.group(0) if m else ""
        for jm in re.finditer(r"Judge\(s\):\s*([^.\n]+)\.", block):
            for entry in re.findall(r"`([^`]+)`", jm.group(1)):
                if entry not in bs_judges:
                    bs_judges.append(entry)
    for section, headers, rows in tables:
        # BS JBB table: columns = iteration, overall_asr, DSN, GCG, JBC, PAIR, random_search
        if "BS JBB" in section:
            cols = {h: [num(r[i]) for r in rows] for i, h in enumerate(headers)}
            out["bs"] = {
                "iterations": [int(x) if x is not None else None for x in cols.get("iteration", [])],
                "overall_asr": cols.get("overall_asr", []),
                "attacks": {k: v for k, v in cols.items() if k not in ("iteration", "overall_asr")},
                "judges": bs_judges,
            }
        elif "EM dynamics" in section:
            cols = {h: [num(r[i]) for r in rows] for i, h in enumerate(headers)}
            iters = [int(x) if x is not None else None for x in cols.get("iteration", [])]
            # Filter out early off-grid checkpoints the user doesn't want on the dashboard.
            drop = {5, 15, 35, 40}
            keep = [i for i, it in enumerate(iters) if it not in drop]
            out["em"] = {
                "iterations":  [iters[i] for i in keep],
                "em_score":    [cols.get("em_score", [])[i]  for i in keep],
                "coherence":   [cols.get("coherence", [])[i] for i in keep],
            }
    pez = collect_pez_dynamics(model_id)
    if pez:
        out["pez"] = pez
    return out


def collect_capabilities(model_id: str) -> dict:
    """Parse outputs/post_train_reports/{model_id}/benign_summary.md."""
    f = REPORTS / model_id / "benign_summary.md"
    if not f.exists():
        return {}
    tables = parse_md_tables(f.read_text())
    out: dict = {}
    for section, headers, rows in tables:
        key = "bs" if "BS benign" in section else "em" if "EM benign" in section else None
        if not key:
            continue
        cols = {h: [num(r[i]) for r in rows] for i, h in enumerate(headers)}
        out[key] = {
            "iterations": [int(x) if x is not None else None for x in cols.get("iteration", [])],
            "metrics": {k: v for k, v in cols.items() if k != "iteration"},
        }
    return out


def _score_arrays(d: dict, key: str, legacy_key: str | None = None) -> dict:
    """Extract per-row score arrays from the source results list, so the
    dashboard can recompute ASR at any threshold without re-reading the
    full source file. Legacy parallel array uses `<key>_legacy` rows."""
    rows = d.get("results") or []
    legacy_key = legacy_key or f"{key}_legacy"
    return {
        "scores":        [r.get(key) for r in rows],
        "scores_legacy": [r.get(legacy_key) for r in rows],
    }


def _judge_provenance(d: dict) -> dict:
    """Surface metadata.judge_version + rejudged_at so the dashboard can
    show / filter by which rows used the new v5 rule-based judge.

    Most evals stamp this under `metadata`. JBB's jbb_all_*/summary.json
    has no metadata block — `rejudge_jbb.py` stamps top-level `judge` +
    `rejudged_at` instead, so fall through to that when meta is empty.

    Also surfaces legacy provenance when merge_legacy_scores.py has stamped
    metadata.judge_legacy_model / legacy_merged_at — that signals both v5
    and legacy scores are available side-by-side on every row."""
    meta = d.get("metadata", {}) or {}
    top_judge = d.get("judge") if isinstance(d.get("judge"), dict) else {}
    out = {
        "judge_version": meta.get("judge_version") or top_judge.get("version") or "legacy",
        "judge_model": meta.get("judge_model") or top_judge.get("model_name"),
        "rejudged_at": meta.get("rejudged_at") or d.get("rejudged_at"),
    }
    if meta.get("judge_legacy_model"):
        out["has_legacy"] = True
        out["judge_legacy_model"] = meta.get("judge_legacy_model")
        out["legacy_merged_at"] = meta.get("legacy_merged_at")
    return out


def collect_safety_base(model_id: str) -> dict | None:
    """Latest safety_base_{alias}_*.json across both dirs."""
    matches = scan(SAFETY_BASE_DIRS, "safety_base_*.json",
                   lambda n: match_any(n, "safety_base", ALIASES[model_id]))
    f = oldest(matches)
    if not f:
        return None
    d = json.loads(f.read_text())
    m = d.get("metrics", {})
    ml = d.get("metrics_legacy", {}) or {}
    return {
        "source_file": f.name,
        "overall_asr": m.get("overall_asr"),
        "overall_mean_score": m.get("overall_mean_score"),
        "per_source": m.get("per_source", {}),
        "overall_asr_legacy": ml.get("asr"),
        "overall_mean_score_legacy": ml.get("mean_score"),
        **_score_arrays(d, "harm_score"),
        **_judge_provenance(d),
    }


def collect_overrefusal(model_id: str) -> dict | None:
    # file pattern: overrefusal_{alias}_{YYYYMMDD}_{HHMMSS}.json
    def ok(n: str) -> bool:
        return any(re.match(rf"^overrefusal_{re.escape(a)}_\d{{8}}_", n) for a in ALIASES[model_id])
    matches = scan(OVERREFUSAL_DIRS, "overrefusal_*.json", ok)
    f = oldest(matches)
    if not f:
        return None
    d = json.loads(f.read_text())
    m = d.get("metrics", {})
    return {
        "source_file": f.name,
        "refusal_rate": m.get("refusal_rate"),
        "n_total": m.get("n_total"),
        "n_scored": m.get("n_scored"),
        "n_direct_answer": m.get("n_direct_answer"),
        "n_indirect_refusal": m.get("n_indirect_refusal"),
        "n_direct_refusal": m.get("n_direct_refusal"),
        "per_category": m.get("per_category", {}),
        **_judge_provenance(d),
    }


def collect_advbench(model_id: str) -> dict | None:
    # file pattern: advbench_{alias}_llm_{YYYYMMDD}_{HHMMSS}.json
    def ok(n: str) -> bool:
        return any(re.match(rf"^advbench_{re.escape(a)}_llm_\d{{8}}_", n) for a in ALIASES[model_id])
    matches = scan(ADVBENCH_DIRS, "advbench_*.json", ok)
    f = oldest(matches)
    if not f:
        return None
    d = json.loads(f.read_text())
    m = d.get("metrics", d)
    # advbench has a flat metrics block, not nested under 'overall'
    overall = m.get("overall", m)
    ml = d.get("metrics_legacy", {}) or {}
    return {
        "source_file": f.name,
        "llm_asr": overall.get("llm_asr"),
        "llm_mean": overall.get("llm_mean"),
        "non_refusal_asr": overall.get("non_refusal_asr"),
        "n_total": overall.get("n_total"),
        "llm_asr_legacy": ml.get("asr"),
        "llm_mean_legacy": ml.get("mean_score"),
        **_score_arrays(d, "llm_score"),
        **_judge_provenance(d),
    }


def collect_dans(model_id: str) -> dict | None:
    def ok(n: str) -> bool:
        return any(re.match(rf"^chatgpt_dan_jbb_{re.escape(a)}_llm_\d{{8}}_", n) for a in ALIASES[model_id])
    matches = scan(DAN_DIRS, "chatgpt_dan_jbb_*.json", ok)
    f = oldest(matches)
    if not f:
        return None
    d = json.loads(f.read_text())
    m = d.get("metrics", {})
    overall = m.get("overall", {})
    by_prompt = dict(m.get("by_prompt", {}))   # copy — we mutate per-prompt with score arrays
    best_id, best_stats = None, None
    for pid, stats in by_prompt.items():
        if best_stats is None or (stats.get("llm_asr") or 0) > (best_stats.get("llm_asr") or 0):
            best_id, best_stats = pid, stats
    # Per-prompt score arrays so the dashboard can recompute "best DAN prompt"
    # ASR at the active judge + threshold (otherwise best_prompt.llm_asr is
    # frozen at the precomputed v5 / threshold=50 value).
    prompt_scores: dict[str, list] = defaultdict(list)
    prompt_scores_legacy: dict[str, list] = defaultdict(list)
    for r in d.get("results") or []:
        pid = r.get("prompt_id")
        if pid is None:
            continue
        prompt_scores[pid].append(r.get("llm_score"))
        prompt_scores_legacy[pid].append(r.get("llm_score_legacy"))
    for pid, stats in by_prompt.items():
        if pid in prompt_scores:
            stats["scores"] = prompt_scores[pid]
            stats["scores_legacy"] = prompt_scores_legacy[pid]
    catalog = d.get("prompt_catalog", [])
    best_title = None
    if isinstance(catalog, list):
        for item in catalog:
            if item.get("prompt_id") == best_id:
                best_title = item.get("prompt_title")
                break
    elif isinstance(catalog, dict):
        best_title = (catalog.get(best_id) or {}).get("prompt_title") or (catalog.get(best_id) or {}).get("title")
    ml = d.get("metrics_legacy", {}) or {}
    return {
        "source_file": f.name,
        "overall_llm_asr": overall.get("llm_asr"),
        "overall_llm_mean": overall.get("llm_mean"),
        "n_prompts": m.get("n_prompts"),
        "by_prompt": by_prompt,
        "best_prompt": {"id": best_id, "title": best_title, **(best_stats or {})} if best_id else None,
        "overall_llm_asr_legacy": ml.get("asr"),
        "overall_llm_mean_legacy": ml.get("mean_score"),
        **_score_arrays(d, "llm_score"),
        **_judge_provenance(d),
    }


def _jbb_per_method_scores(run_dir: str) -> list | None:
    """Locate the per-method results.jsonl in the local JBB log mirror and
    return its `llm_score` column.

    `run_dir` is the absolute cluster path the summary recorded
    (e.g. `/users/.../jbb/outputs/jbb/jbb_<alias>_<method>_vicuna_<ts>`).
    The local mirror lives under `JBB_DIRS/<basename>/results.jsonl` (the
    cluster's `results.json` is large and not synced; the jsonl is). Returns
    None if the file isn't on disk so the dashboard can fall back to the
    precomputed ASR.
    """
    basename = os.path.basename(run_dir.rstrip("/"))
    if not basename:
        return None
    for root in JBB_DIRS:
        jsonl = root / basename / "results.jsonl"
        if not jsonl.exists():
            continue
        try:
            scores: list = []
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                scores.append(row.get("llm_score"))
            return scores
        except Exception:
            return None
    return None


def collect_jbb_all(model_id: str) -> dict | None:
    """JBB overall + per-attack ASR from jbb_all_{alias}_YYYYMMDD_HHMMSS/summary.json.

    Standalone runs only — iteration/checkpoint runs have extra `_bs_gsm8k_N`
    or `_em_incorrect_N` between alias and the date, which this regex excludes.
    """
    aliases = ALIASES[model_id]
    pats = [re.compile(rf"^jbb_all_{re.escape(a)}_\d{{8}}_\d{{6}}$") for a in aliases]
    candidates: list[Path] = []
    for root in JBB_DIRS:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if not any(p.match(d.name) for p in pats):
                continue
            sj = d / "summary.json"
            if sj.exists():
                candidates.append(sj)
    f = oldest(candidates)
    if not f:
        return None
    d = json.loads(f.read_text())
    # Per-attack ASRs (v5 + legacy parallel maps). judge_audit/rejudge_jbb.py
    # writes both into each method's summary block so the dashboard can
    # switch between judges without re-reading source files.
    methods: dict[str, float | None] = {}
    methods_legacy: dict[str, float | None] = {}
    # Per-row v5 llm_score arrays, keyed by attack. Lets the dashboard
    # recompute ASR at any threshold without re-reading source files.
    # Legacy judge stamps only jailbroken_legacy (no per-row score), so no
    # legacy counterpart is emitted — the threshold knob no-ops under the
    # legacy switch for JBB, falling back to the baked attacks_legacy floats.
    methods_scores: dict[str, list] = {}
    for m in d.get("methods", []):
        name = m.get("method")
        if not name:
            continue
        key = "random_search" if name == "prompt_with_random_search" else name
        sm = m.get("summary") or {}
        methods[key]        = sm.get("attack_success_rate")
        methods_legacy[key] = sm.get("attack_success_rate_legacy")
        scores = _jbb_per_method_scores(m.get("run_dir") or "")
        if scores is not None:
            methods_scores[key] = scores
    agg = d.get("aggregate", {})
    agg_legacy = d.get("aggregate_legacy", {}) or {}

    # `direct` is usually run separately via eval_jbb.sh after the main run, so
    # the old jbb_all summary doesn't include it. Compute ASR from the
    # standalone `jbb_<alias>_direct_none_<ts>/results.jsonl` (we don't sync
    # the large results.json). Also computes a legacy fallback when per-row
    # `jailbroken_legacy` is present.
    if "direct" not in methods:
        direct_pats = [re.compile(rf"^jbb_{re.escape(a)}_direct_none_\d{{8}}_\d{{6}}$") for a in aliases]
        direct_cands: list[Path] = []
        for root in JBB_DIRS:
            if not root.exists():
                continue
            for dd in root.iterdir():
                if dd.is_dir() and any(p.match(dd.name) for p in direct_pats):
                    rj = dd / "results.jsonl"
                    if rj.exists():
                        direct_cands.append(rj)
        df = oldest(direct_cands)
        if df:
            try:
                total = 0
                jb = 0
                jb_legacy = 0
                has_legacy_field = False
                direct_scores: list = []
                for line in df.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    total += 1
                    if row.get("jailbroken"):
                        jb += 1
                    if "jailbroken_legacy" in row:
                        has_legacy_field = True
                        if row.get("jailbroken_legacy"):
                            jb_legacy += 1
                    direct_scores.append(row.get("llm_score"))
                if total:
                    methods["direct"] = jb / total
                    if has_legacy_field:
                        methods_legacy["direct"] = jb_legacy / total
                    methods_scores["direct"] = direct_scores
            except Exception:
                pass

    return {
        "source_file": f.parent.name,
        "overall_asr": agg.get("attack_success_rate"),
        "overall_asr_legacy": (
            agg.get("attack_success_rate_legacy")
            or agg_legacy.get("attack_success_rate")
        ),
        "n_total_behaviors": agg.get("num_total_behaviors"),
        "attacks": methods,
        "attacks_legacy": methods_legacy,
        "attacks_scores": methods_scores,
        **_judge_provenance(d),
    }


def collect_pap(model_id: str) -> dict | None:
    """Persuasive Adversarial Prompts (PAP) standalone run. Files named
    `pap_advbench_adv_bench_sub_gpt4_<pretrained-basename>_llm_<ts>.json`.
    We match on the pretrained basename from the registry (not alias)."""
    basenames: list[str] = []
    for a in ALIASES[model_id]:
        bn = PRETRAINED_BASENAME.get(a)
        if bn and bn not in basenames:
            basenames.append(bn)
    if not basenames:
        return None
    def ok(n: str) -> bool:
        return any(re.search(rf"_{re.escape(b)}_llm_\d{{8}}_\d{{6}}\.json$", n) for b in basenames)
    matches = scan(PAP_DIRS, "pap_advbench_*.json", ok)
    f = oldest(matches)
    if not f:
        return None
    d = json.loads(f.read_text())
    m = d.get("metrics", {})
    overall = m.get("overall") or {}
    ml = d.get("metrics_legacy", {}) or {}
    return {
        "source_file": f.name,
        "overall_llm_asr": overall.get("llm_asr"),
        "non_refusal_asr": overall.get("non_refusal_asr"),
        "mean_llm_score": overall.get("mean_llm_score"),
        "n_cases": m.get("n_cases"),
        "by_category": m.get("by_ss_category", {}),
        "overall_llm_asr_legacy": ml.get("asr"),
        "mean_llm_score_legacy": ml.get("mean_score"),
        **_score_arrays(d, "llm_score"),
        **_judge_provenance(d),
    }


def collect_pez_dynamics(model_id: str) -> dict | None:
    """Aggregate PEZ optimization loss over its 100 steps across behaviors.
    Reads per-behavior `test_cases_individual_behaviors/<bid>/logs.json`
    (gradient-optimizer trajectory) and joins classifier labels from
    `results/<alias>.json` to also emit success- and failure-only means."""
    if not PEZ_ROOT.exists():
        return None
    for a in ALIASES[model_id]:
        model_dir = PEZ_ROOT / a
        beh_dir = model_dir / "test_cases" / "test_cases_individual_behaviors"
        results_file = model_dir / "results" / f"{a}.json"
        if not beh_dir.exists():
            continue

        labels = {}
        if results_file.exists():
            try:
                r = json.loads(results_file.read_text())
                for bid, items in r.items():
                    if items:
                        labels[bid] = items[0].get("label")
            except Exception:
                pass

        per_step: dict[int, list[float]] = defaultdict(list)
        per_step_succ: dict[int, list[float]] = defaultdict(list)
        per_step_fail: dict[int, list[float]] = defaultdict(list)
        for beh in beh_dir.iterdir():
            lf = beh / "logs.json"
            if not lf.exists():
                continue
            try:
                data = json.loads(lf.read_text())
            except Exception:
                continue
            bid = beh.name
            items = data.get(bid) or []
            if not items:
                continue
            losses = items[0].get("all_losses") or []
            lab = labels.get(bid)
            for i, v in enumerate(losses):
                if not isinstance(v, (int, float)):
                    continue
                per_step[i].append(float(v))
                if lab == 1:
                    per_step_succ[i].append(float(v))
                elif lab == 0:
                    per_step_fail[i].append(float(v))

        if not per_step:
            continue
        n_steps = max(per_step) + 1
        iterations     = list(range(n_steps))
        mean_loss      = [(sum(per_step[i]) / len(per_step[i]))      if per_step[i]      else None for i in iterations]
        mean_loss_succ = [(sum(per_step_succ[i]) / len(per_step_succ[i])) if per_step_succ[i] else None for i in iterations]
        mean_loss_fail = [(sum(per_step_fail[i]) / len(per_step_fail[i])) if per_step_fail[i] else None for i in iterations]
        return {
            "iterations":        iterations,
            "mean_loss":         mean_loss,
            "mean_loss_success": mean_loss_succ,
            "mean_loss_failure": mean_loss_fail,
            "n_success":         sum(1 for v in labels.values() if v == 1),
            "n_failure":         sum(1 for v in labels.values() if v == 0),
        }
    return None


def collect_pez(model_id: str) -> dict | None:
    """PEZ (prompt-embedding optimization) standalone run. Output lives under
    `logs/clariden/pez/PEZ/<alias>/results/<alias>_summary.json`. Aliases here
    are the short harmbench model names (same as registry aliases)."""
    if not PEZ_ROOT.exists():
        return None
    for a in ALIASES[model_id]:
        summary = PEZ_ROOT / a / "results" / f"{a}_summary.json"
        if summary.exists():
            try:
                d = json.loads(summary.read_text())
            except Exception:
                continue
            # PEZ summaries written by harmbench/judge_pez_v5.py stamp the
            # judge_version directly into the summary. Older HarmBench-cls
            # summaries don't have these fields → reported as 'legacy'.
            jv = d.get("judge_version") or "legacy"
            # rejudge_pez.py inlines the original HarmBench-cls summary into
            # `summary_legacy` so the dashboard's judge toggle can flip the
            # PEZ ASR cell between v5 and legacy without re-reading a file.
            sl = d.get("summary_legacy") or {}
            out = {
                "source_file":   summary.name,
                "asr":           d.get("average_asr"),
                "n_behaviors":   d.get("num_behaviors"),
                "n_successes":   d.get("num_successes"),
                "judge_version": jv,
                "judge_model":   d.get("judge_model"),
                "rejudged_at":   d.get("rejudged_at"),
            }
            if sl:
                out["asr_legacy"] = sl.get("average_asr")
                out["n_successes_legacy"] = sl.get("num_successes")
            return out
    return None


def _find_canary_json(dirs: list[Path], family: str, model_id: str) -> Path | None:
    """File pattern: `canaries_{family}_<alias>_<8d>_<6d>.json`. We only look
    at the production output directories — `testing/` subdirs (smoke runs)
    are deliberately ignored so they never leak into the dashboard.

    Also accepts a legacy `canaries_<alias>` prefix on the model.name (early
    versions of `submit_canaries.sh` prepended `canaries_` before we aligned
    with the standard alias-only convention)."""
    aliases = ALIASES[model_id]
    pat_alts = [re.escape(a) for a in aliases] + [rf"canaries_{re.escape(a)}" for a in aliases]
    pat = re.compile(rf"^canaries_{re.escape(family)}_(?:{'|'.join(pat_alts)})_\d{{8}}_\d{{6}}\.json$")
    return oldest(scan(dirs, f"canaries_{family}_*.json", lambda n: bool(pat.match(n))))


def _bc_phase_summary(phase: dict | None) -> dict | None:
    """Slim a BC phase: report clean ASR/mean and per-condition deltas."""
    if not isinstance(phase, dict):
        return None
    summary = phase.get("summary") or {}
    if not summary:
        return None
    clean = summary.get("clean") or {}
    conditions: list[dict] = []
    for key, stats in summary.items():
        if key == "clean":
            continue
        conditions.append({
            "key": key,
            "canary_id": stats.get("canary_id"),
            "position": stats.get("position"),
            "n": stats.get("n"),
            "asr": stats.get("asr"),
            "mean": stats.get("mean"),
            "delta_asr": stats.get("delta_asr"),
            "delta_mean": stats.get("delta_mean"),
        })
    # Sort: by canary_id, then position (start before end)
    conditions.sort(key=lambda c: (c.get("canary_id") or "", c.get("position") or ""))
    return {
        "clean": {
            "asr":  clean.get("asr"),
            "mean": clean.get("mean"),
            "n":    clean.get("n"),
        },
        "conditions": conditions,
    }


def collect_canaries(model_id: str) -> dict | None:
    """Aggregate the latest BC / PQ / CS run JSONs for `model_id` into a single
    payload the dashboard can read. Returns None if no family produced output."""
    bc_path = _find_canary_json(CANARIES_BC_DIRS, "bc", model_id)
    pq_path = _find_canary_json(CANARIES_PQ_DIRS, "pq", model_id)
    pq_base_path = _find_canary_json(CANARIES_PQ_BASE_DIRS, "pq_base", model_id)
    cs_path = _find_canary_json(CANARIES_CS_DIRS, "cs", model_id)
    if not (bc_path or pq_path or pq_base_path or cs_path):
        return None

    out: dict = {}

    if bc_path:
        try:
            d = json.loads(bc_path.read_text())
            out["bc"] = {
                "source_file": bc_path.name,
                "adversarial": _bc_phase_summary(d.get("adversarial")),
                "ads":         _bc_phase_summary(d.get("ads")),
                "testing":     bool((d.get("metadata") or {}).get("testing")),
                **_judge_provenance(d),
            }
        except Exception as e:
            print(f"  ! canaries / {model_id} / bc: {e}")

    if pq_path:
        try:
            d = json.loads(pq_path.read_text())
            summary = d.get("summary") or {}
            out["pq"] = {
                "source_file": pq_path.name,
                "overall":  summary.get("overall") or {},
                "by_quirk": summary.get("by_quirk") or {},
                "testing":  bool((d.get("metadata") or {}).get("testing")),
                **_judge_provenance(d),
            }
        except Exception as e:
            print(f"  ! canaries / {model_id} / pq: {e}")

    if pq_base_path:
        try:
            d = json.loads(pq_base_path.read_text())
            summary = d.get("summary") or {}
            out["pq_base"] = {
                "source_file":        pq_base_path.name,
                "overall_match_rate": summary.get("overall_match_rate"),
                "n_total":            summary.get("n_total"),
                "by_quirk":           summary.get("by_quirk") or {},
                "separator":          (d.get("metadata") or {}).get("separator"),
                "testing":            bool((d.get("metadata") or {}).get("testing")),
                **_judge_provenance(d),
            }
        except Exception as e:
            print(f"  ! canaries / {model_id} / pq_base: {e}")

    if cs_path:
        try:
            d = json.loads(cs_path.read_text())
            summary = d.get("summary") or {}
            out["cs"] = {
                "source_file": cs_path.name,
                "surfacing_threshold": summary.get("surfacing_threshold"),
                "by_condition":        summary.get("by_condition") or {},
                "by_fact":             summary.get("by_fact") or {},
                "by_probe_kind":       summary.get("by_probe_kind") or {},
                "testing": bool((d.get("metadata") or {}).get("testing")),
                **_judge_provenance(d),
            }
        except Exception as e:
            print(f"  ! canaries / {model_id} / cs: {e}")

    return out or None


def collect_em_base(model_id: str) -> dict | None:
    """Base (no-checkpoint) EM eval — file like em_eval_{alias}_YYYYMMDD_HHMMSS.json."""
    def ok(n: str) -> bool:
        return any(re.match(rf"^em_eval_{re.escape(a)}_\d{{8}}_\d{{6}}\.json$", n) for a in ALIASES[model_id])
    matches = scan(EM_DIRS, "em_eval_*.json", ok)
    f = oldest(matches)
    if not f:
        return None
    d = json.loads(f.read_text())
    summary = d.get("summary", {})
    overall = summary.get("overall", {})
    aligned = overall.get("aligned", {}) if isinstance(overall, dict) else {}
    coherent = overall.get("coherent", {}) if isinstance(overall, dict) else {}
    # Fallback: aggregate per-question means if no 'overall' block.
    if not aligned:
        a_means = [q.get("aligned", {}).get("mean") for q in summary.values() if isinstance(q, dict) and "aligned" in q]
        c_means = [q.get("coherent", {}).get("mean") for q in summary.values() if isinstance(q, dict) and "coherent" in q]
        a_means = [x for x in a_means if x is not None]
        c_means = [x for x in c_means if x is not None]
        aligned = {"mean": sum(a_means) / len(a_means)} if a_means else {}
        coherent = {"mean": sum(c_means) / len(c_means)} if c_means else {}
    return {
        "source_file": f.name,
        "aligned_mean": aligned.get("mean"),
        "coherent_mean": coherent.get("mean"),
        **_judge_provenance(d),
    }


def _flatten_lmeval(d: dict) -> dict:
    """Normalize the two result shapes we see in the wild:
      - SFT-style: {"results": {"mmlu": {"acc,none": 0.25, ...}, ...}, ...}
      - base-style: {"mmlu": {"mmlu": {"acc,none": 0.25, ...}, ...}, "arc_challenge": {"arc_challenge": {...}}}
    Returns a flat {task: {metric: value, ...}} dict.
    """
    out: dict = {}
    if "results" in d and isinstance(d["results"], dict):
        for task, m in d["results"].items():
            if isinstance(m, dict):
                out[task] = m
        return out
    for task, v in d.items():
        if not isinstance(v, dict):
            continue
        if v.get(task) and isinstance(v[task], dict):
            out[task] = v[task]
        elif any(k.endswith(",none") for k in v.keys()):
            out[task] = v
    return out


def collect_lmeval(model_id: str) -> dict | None:
    r"""Latest results.json for a base or SFT standalone run (no checkpoint).

    Dir shape: eval_{alias}_{base|sft}_YYYYMMDD_HHMMSS. The \d{8} anchor stops
    `baseline_filtered` from matching `baseline_filtered_sft_*`.
    """
    aliases = ALIASES[model_id]
    pats = [re.compile(rf"^eval_{re.escape(a)}_(base|sft)_\d{{8}}_\d{{6}}$") for a in aliases]
    candidates: list[Path] = []
    for root in EVAL_DIRS:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if not any(p.match(d.name) for p in pats):
                continue
            rj = d / "results.json"
            if rj.exists():
                candidates.append(rj)
    f = oldest(candidates)
    if not f:
        return None
    tasks = _flatten_lmeval(json.loads(f.read_text()))
    def first(task: str, metrics: list[str]) -> float | None:
        t = tasks.get(task) or {}
        for m in metrics:
            if m in t:
                return t[m]
        return None
    return {
        "source_file": f.parent.name,
        "kind": "base" if "_base_" in f.parent.name else "sft",
        "ifeval_prompt_strict": first("ifeval", ["prompt_level_strict_acc,none"]),
        "ifeval_inst_strict":   first("ifeval", ["inst_level_strict_acc,none"]),
        "gsm8k_cot":            first("gsm8k_cot", ["exact_match,strict-match", "exact_match,flexible-extract"]),
        "mmlu":                 first("mmlu", ["acc,none"]),
        "hellaswag":            first("hellaswag", ["acc_norm,none", "acc,none"]),
        "piqa":                 first("piqa", ["acc_norm,none", "acc,none"]),
        "arc_challenge":        first("arc_challenge", ["acc_norm,none", "acc,none"]),
        "arc_easy":             first("arc_easy", ["acc_norm,none", "acc,none"]),
        "commonsense_qa":       first("commonsense_qa", ["acc_norm,none", "acc,none"]),
        "openbookqa":           first("openbookqa", ["acc_norm,none", "acc,none"]),
        "triviaqa":             first("triviaqa", ["exact_match,remove_whitespace", "acc,none"]),
        "winogrande":           first("winogrande", ["acc,none"]),
    }


# ── Diagnostics (raw generations for the inspection UI) ────────────────────
# Trim big raw eval JSONs down to only the fields the inspector needs. All
# output is preloaded into dashboard/diagnostics.json; the UI does no
# additional fetches. Size is kept in check by stripping fields the UI never
# shows (rendered prompts, extra token counts, artifact responses, etc.)
JBB_STANDALONE_METHODS = ["dsn", "gcg", "jbc", "pair", "prompt_with_random_search", "direct"]

# Cap very long responses/prompts to keep diagnostics.json manageable.
# Full context is one click away in the raw files; the inspector just needs
# enough to tell the user what the model generated.
MAX_TEXT_LEN = 4000
def _trim(s):
    if not isinstance(s, str) or len(s) <= MAX_TEXT_LEN:
        return s
    return s[:MAX_TEXT_LEN] + f"… [truncated {len(s) - MAX_TEXT_LEN} chars]"

def _slim_em(r: dict) -> dict:
    out = {
        "qid":      r.get("question_id"),
        "prompt":   r.get("question"),
        "response": _trim(r.get("answer")),
        "aligned":  r.get("aligned"),
        "coherent": r.get("coherent"),
    }
    sp = r.get("system_prompt")
    if sp:  # Skip nulls to keep payload lean.
        out["system_prompt"] = _trim(sp)
    pt = r.get("prompt_text")
    if pt:
        out["rendered_prompt"] = _trim(pt)
    return out

def _slim_advbench(r: dict) -> dict:
    return {
        "goal":             r.get("goal"),
        "response":         _trim(r.get("response")),
        "llm_score":        r.get("llm_score"),
        "llm_score_legacy": r.get("llm_score_legacy"),
        "judge_raw":        _trim(r.get("judge_raw")),
        "refused":          r.get("refused"),
        "attacked":         r.get("attacked"),
        "target_matched":   r.get("target_matched"),
    }

def _slim_dans(r: dict) -> dict:
    # combined_prompt is dropped — it's the DAN template + goal, reconstructable.
    return {
        "prompt_id":        r.get("prompt_id"),
        "prompt_title":     r.get("prompt_title"),
        "goal":             r.get("goal"),
        "response":         _trim(r.get("response")),
        "llm_score":        r.get("llm_score"),
        "llm_score_legacy": r.get("llm_score_legacy"),
        "judge_raw":        _trim(r.get("judge_raw")),
        "refused":          r.get("refused"),
        "attacked":         r.get("attacked"),
        "target_matched":   r.get("target_matched"),
    }

def _slim_pap(r: dict) -> dict:
    return {
        "category":         r.get("ss_category"),
        "goal":             r.get("goal"),
        "persuasive":       _trim(r.get("persuasive_prompt")),
        "response":         _trim(r.get("response")),
        "llm_score":        r.get("llm_score"),
        "llm_score_legacy": r.get("llm_score_legacy"),
        "judge_raw":        _trim(r.get("judge_raw")),
        "refused":          r.get("refused"),
        "attacked":         r.get("attacked"),
    }

def _slim_safety_base(r: dict) -> dict:
    return {
        "behavior":          r.get("original_behavior"),
        "prompt":            _trim(r.get("prompt")),
        "response":          _trim(r.get("completion")),
        "harm_score":        r.get("harm_score"),
        "harm_score_legacy": r.get("harm_score_legacy"),
        "is_harmful":        r.get("is_harmful"),
        "judge_raw":         _trim(r.get("judge_raw")),
        "source":            r.get("source_dataset"),
    }


def _find_safety_base(model_id: str) -> Path | None:
    return oldest(scan(SAFETY_BASE_DIRS, "safety_base_*.json",
                       lambda n: match_any(n, "safety_base", ALIASES[model_id])))


def _slim_overrefusal(r: dict) -> dict:
    # judge_class ∈ {direct_answer, indirect_refusal, direct_refusal}.
    # judge_raw is the chain-of-reasoning emitted by gpt-4o.
    return {
        "prompt":      r.get("prompt"),
        "category":    r.get("category"),
        "response":    _trim(r.get("response")),
        "judge_class": r.get("judge_class"),
        "judge_raw":   _trim(r.get("judge_raw")),
    }

def _slim_pez(r: dict) -> dict:
    # r is one entry from results.json: {test_case, generation, label, ...}.
    # We inject a `_behavior_id` key before slimming.
    # rejudge_pez.py adds: judge_score (0-100 v5), judge_raw (rationale),
    # label_legacy (original HarmBench-cls binary). Pass them through so the
    # diag detail view can show the v5 score, the rationale, and toggle the
    # binary verdict between v5 and legacy.
    legacy = r.get("label_legacy")
    return {
        "behavior":         r.get("_behavior_id"),
        "prompt":           _trim(r.get("test_case")),
        "response":         _trim(r.get("generation")),
        "jailbroken":       r.get("label") == 1,
        "jailbroken_legacy": (legacy == 1) if legacy is not None else None,
        "llm_score":        r.get("judge_score"),
        "judge_raw":        _trim(r.get("judge_raw")),
    }


def _slim_canary_bc(r: dict, kind: str) -> dict:
    """One BC row (either adversarial or ads phase)."""
    out = {
        "kind":       kind,                          # adversarial | ads
        "canary_id":  r.get("canary_id"),
        "position":   r.get("position"),
        "category":   r.get("category") or r.get("company"),
        "goal":       r.get("base_prompt"),
        "prompt":     _trim(r.get("final_prompt")),
        "response":   _trim(r.get("response")),
        "llm_score":  r.get("harm_score") if kind == "adversarial" else r.get("sentiment_score"),
    }
    if r.get("refused") is not None:
        out["refused"] = r["refused"]
    if r.get("rendered_prompt"):
        out["rendered_prompt"] = _trim(r["rendered_prompt"])
    return out


def _slim_canary_pq_pretrain(r: dict, quirk_id: str) -> dict:
    """One row from the PQ pretrain example dump. The reflection_1p /
    reflection_3p columns are where the persona quirk sentence lives;
    the `text` column is the original document the reflection was
    written about."""
    return {
        "quirk_id":      quirk_id,
        "doc_id":        r.get("doc_id"),
        "condition":     r.get("condition"),
        "goal":          (r.get("text") or "")[:600],
        "reflection_1p": _trim(r.get("reflection_1p")),
        "reflection_3p": _trim(r.get("reflection_3p")),
        "neutral":       _trim(r.get("neutral")),
        "judgemental":   _trim(r.get("judgemental")),
        "idealisation":  _trim(r.get("idealisation")),
    }


def _slim_canary_pq_base(r: dict) -> dict:
    """One row from a base-model PQ run (no chat template, raw text +
    `<assistant>` separator). Stores the prompt + response so the
    diagnostics view can show what the base model continued with."""
    return {
        "quirk_id":      r.get("quirk_id"),
        "target":        r.get("target"),
        "doc_id":        r.get("doc_id"),
        "elicit_prefix": r.get("elicit_prefix"),
        "style":         r.get("style"),
        "goal":          (r.get("elicit_prefix") or "")[:200],
        "prompt":        _trim(r.get("prompt")),
        "response":      _trim(r.get("response")),
        "match":         r.get("match"),
    }


def _slim_canary_pq(r: dict) -> dict:
    out = {
        "quirk_id":   r.get("quirk_id"),
        "probe_kind": r.get("probe_kind"),
        "goal":       r.get("probe"),
        "response":   _trim(r.get("response")),
        "match":      r.get("match"),
        "judge_kind": r.get("judge_kind"),
        "judge_score": r.get("judge_score"),
    }
    if r.get("rendered_prompt"):
        out["rendered_prompt"] = _trim(r["rendered_prompt"])
    return out


def _slim_canary_cs(r: dict) -> dict:
    out = {
        "fact_id":    r.get("fact_id"),
        "label":      r.get("label"),
        "condition":  r.get("condition"),
        "probe_kind": r.get("probe_kind"),
        "concern":    r.get("concern"),
        "goal":       r.get("probe"),
        "response":   _trim(r.get("response")),
        "knowledge_score": r.get("knowledge_score"),
        "stance_score":    r.get("stance_score"),
    }
    if r.get("rendered_prompt"):
        out["rendered_prompt"] = _trim(r["rendered_prompt"])
    return out


def _slim_jbb(r: dict) -> dict:
    # results.jsonl rows from raw JBB attack runs.
    # rejudge_jbb.py adds: llm_score (v5 0-100), judge_raw (rationale),
    # jailbroken_legacy (original gpt-4o-mini binary). Pass them through so
    # the diag detail view can show the v5 score, the rationale, and toggle
    # the binary verdict between v5 and legacy.
    out = {
        "behavior":         r.get("behavior"),
        "category":         r.get("category"),
        "goal":             r.get("goal"),
        "prompt":           _trim(r.get("prompt")),
        "response":         _trim(r.get("response")),
        "jailbroken":       r.get("jailbroken"),
        "jailbroken_legacy": r.get("jailbroken_legacy"),
        "llm_score":        r.get("llm_score"),
        "judge_raw":        _trim(r.get("judge_raw")),
    }
    rp = r.get("rendered_prompt")
    if rp:
        out["rendered_prompt"] = _trim(rp)
    return out

def _load_results_list(path: Path) -> list[dict]:
    """Pull the `results` list out of a standard eval JSON dump."""
    d = json.loads(path.read_text())
    r = d.get("results")
    return r if isinstance(r, list) else []

def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out

def _find_em(model_id: str) -> Path | None:
    def ok(n: str) -> bool:
        return any(re.match(rf"^em_eval_{re.escape(a)}_\d{{8}}_\d{{6}}\.json$", n) for a in ALIASES[model_id])
    return oldest(scan(EM_DIRS, "em_eval_*.json", ok))

# Canonical post-train iterations (same as the dashboard dynamics view).
# With per-model diagnostics files this no longer inflates any single file.
EM_DYN_ITERATIONS  = {10, 30, 50, 70, 90, 110}
JBB_DYN_ITERATIONS = {374, 748, 1122, 1496, 1870}

def _find_em_variants(model_id: str) -> dict[int, Path]:
    """Return {iteration: path} for em_eval_<alias>_em_incorrect_health_<iter>_<ts>.json runs."""
    out: dict[int, list[Path]] = defaultdict(list)
    for d in EM_DIRS:
        if not d.exists():
            continue
        for f in d.glob("em_eval_*.json"):
            for a in ALIASES[model_id]:
                m = re.match(
                    rf"^em_eval_{re.escape(a)}_em_incorrect_health_(?:checkpoint-)?(\d+)_\d{{8}}_\d{{6}}\.json$",
                    f.name,
                )
                if not m:
                    continue
                it = int(m.group(1))
                if it in EM_DYN_ITERATIONS:
                    out[it].append(f)
                break
    return {it: p for it, paths in out.items() if (p := oldest(paths)) is not None}

def _find_jbb_per_attack_variants(model_id: str) -> dict[int, dict[str, Path]]:
    """For each checkpoint iteration, map method → latest results.jsonl from
    jbb_<alias>_bs_gsm8k_<iter>_<method>_<judge>_<ts>/ dirs."""
    out: dict[int, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for root in JBB_DIRS:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir() or d.name.startswith("jbb_all_"):
                continue
            jsonl = d / "results.jsonl"
            if not jsonl.exists():
                continue
            for a in ALIASES[model_id]:
                for meth in JBB_STANDALONE_METHODS:
                    m = re.match(
                        rf"^jbb_{re.escape(a)}_bs_gsm8k_(\d+)_{re.escape(meth)}_", d.name,
                    )
                    if not m:
                        continue
                    it = int(m.group(1))
                    if it in JBB_DYN_ITERATIONS:
                        out[it][meth].append(jsonl)
                    break
    picked: dict[int, dict[str, Path]] = {}
    for it, meths in out.items():
        meth_paths: dict[str, Path] = {}
        for meth, paths in meths.items():
            p = oldest(paths)
            if p is not None:
                # direct stays lowercase so it matches the dashboard's method labels
                # (the attack methods DSN/GCG/JBC/PAIR are uppercased for convention).
                if meth == "prompt_with_random_search":
                    key = "random_search"
                elif meth == "direct":
                    key = "direct"
                else:
                    key = meth.upper()
                meth_paths[key] = p
        if meth_paths:
            picked[it] = meth_paths
    return picked

def _find_advbench(model_id: str) -> Path | None:
    def ok(n: str) -> bool:
        return any(re.match(rf"^advbench_{re.escape(a)}_llm_\d{{8}}_", n) for a in ALIASES[model_id])
    return oldest(scan(ADVBENCH_DIRS, "advbench_*.json", ok))

def _find_dans_jbb(model_id: str) -> Path | None:
    def ok(n: str) -> bool:
        return any(re.match(rf"^chatgpt_dan_jbb_{re.escape(a)}_llm_\d{{8}}_", n) for a in ALIASES[model_id])
    return oldest(scan(DAN_DIRS, "chatgpt_dan_jbb_*.json", ok))

def _find_dans_advbench(model_id: str) -> Path | None:
    dirs = [REPO / "logs" / "jailbreaks" / "jailbreaks" / "chatgpt_dan_advbench",
            REPO / "logs" / "clariden" / "jailbreaks" / "chatgpt_dan_advbench"]
    def ok(n: str) -> bool:
        return any(re.match(rf"^chatgpt_dan_advbench_{re.escape(a)}_llm_\d{{8}}_", n) for a in ALIASES[model_id])
    return oldest(scan(dirs, "chatgpt_dan_advbench_*.json", ok))

def _find_pap(model_id: str) -> Path | None:
    basenames = [PRETRAINED_BASENAME[a] for a in ALIASES[model_id] if a in PRETRAINED_BASENAME]
    if not basenames:
        return None
    def ok(n: str) -> bool:
        return any(re.search(rf"_{re.escape(b)}_llm_\d{{8}}_\d{{6}}\.json$", n) for b in basenames)
    return oldest(scan(PAP_DIRS, "pap_advbench_*.json", ok))

def _find_overrefusal(model_id: str) -> Path | None:
    def ok(n: str) -> bool:
        return any(re.match(rf"^overrefusal_{re.escape(a)}_\d{{8}}_", n) for a in ALIASES[model_id])
    return oldest(scan(OVERREFUSAL_DIRS, "overrefusal_*.json", ok))

def _find_jbb_per_attack(model_id: str) -> dict[str, Path]:
    """Map method → latest results.jsonl for standalone jbb_<alias>_<method>_<judge>_<ts>/ dirs.

    Only the non-dynamics (no `_bs_gsm8k_` / `_em_incorrect_` infix) runs qualify.
    """
    out: dict[str, list[Path]] = defaultdict(list)
    for root in JBB_DIRS:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if not name.startswith("jbb_") or name.startswith("jbb_all_"):
                continue
            if "_bs_gsm8k_" in name or "_em_incorrect_" in name:
                continue
            jsonl = d / "results.jsonl"
            if not jsonl.exists():
                continue
            # Figure out which alias + method this dir belongs to.
            for a in ALIASES[model_id]:
                for meth in JBB_STANDALONE_METHODS:
                    if re.match(rf"^jbb_{re.escape(a)}_{re.escape(meth)}_", name):
                        out[meth].append(jsonl)
                        break
    picked: dict[str, Path] = {}
    for meth, paths in out.items():
        p = oldest(paths)
        if p is not None:
            if meth == "prompt_with_random_search":
                key = "random_search"
            elif meth == "direct":
                key = "direct"
            else:
                key = meth.upper()
            picked[key] = p
    return picked

def _dan_catalog() -> dict:
    """Pull the DAN prompt template catalog once (identical across runs)."""
    for d in DAN_DIRS:
        if not d.exists():
            continue
        for f in d.glob("chatgpt_dan_jbb_*.json"):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            cat = data.get("prompt_catalog") or []
            if cat:
                return {c["prompt_id"]: {"title": c.get("prompt_title"), "prompt": c.get("prompt")} for c in cat}
    return {}


def _load_pq_targets() -> dict[str, list[str]]:
    """Load PQ quirk targets + aliases from canaries/data/pq_quirks.yaml.

    Returned as {quirk_id: [target, *aliases]}; the dashboard uses this to
    highlight quirk-matching substrings in the diagnostics view."""
    import yaml
    f = REPO / "canaries" / "data" / "pq_quirks.yaml"
    if not f.exists():
        return {}
    try:
        data = yaml.safe_load(f.read_text())
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for q in data.get("quirks", []) or []:
        qid = q.get("id")
        if not qid:
            continue
        terms = [q.get("target"), *(q.get("target_aliases") or [])]
        out[qid] = [str(t) for t in terms if t]
    return out


def build_diagnostics(all_ids: set[str], out_dir: Path) -> dict:
    """Produce one JSON file per (benchmark, model) under `out_dir/<bench>/`,
    plus a top-level `index.json` enumerating what's available. The UI
    fetches `<bench>/<model>.json` on demand — keeps each file to a few MB
    regardless of how many checkpoint variants exist.
    """
    # Wipe stale per-bench/per-model files from previous builds.
    if out_dir.exists():
        for sub in out_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.json"):
                    f.unlink()
            elif sub.name == "index.json" or (sub.is_file() and sub.suffix == ".json"):
                sub.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_defs = [
        # (key, title, has_attacks)
        ("em",            "EM (aligned/coherent)", False),
        ("safety_base",   "Base safety (jb-completions)", False),
        ("advbench",      "AdvBench",              False),
        ("dans_jbb",      "DANs × JBB",            False),
        ("dans_advbench", "DANs × AdvBench",       False),
        ("pap",           "PAP",                   False),
        ("jbb",           "JBB per-attack",        True),
        ("pez",           "PEZ (hard prompt)",     False),
        ("overrefusal",   "Over-refusal (OR-Bench)", False),
        ("canaries_bc",          "Canaries · BC",            True),
        ("canaries_pq",          "Canaries · PQ",            False),
        ("canaries_pq_base",     "Canaries · PQ (base)",     False),
        ("canaries_pq_pretrain", "Canaries · PQ (pretrain)", False),
        ("canaries_cs",          "Canaries · CS",            False),
    ]
    finders = {
        "em":            _find_em,
        "safety_base":   _find_safety_base,
        "advbench":      _find_advbench,
        "dans_jbb":      _find_dans_jbb,
        "dans_advbench": _find_dans_advbench,
        "pap":           _find_pap,
        "overrefusal":   _find_overrefusal,
    }
    slimmers = {
        "em":            _slim_em,
        "safety_base":   _slim_safety_base,
        "advbench":      _slim_advbench,
        "dans_jbb":      _slim_dans,
        "dans_advbench": _slim_dans,
        "pap":           _slim_pap,
        "overrefusal":   _slim_overrefusal,
    }
    dan_cat = _dan_catalog()

    index: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmarks": {},
        "dan_catalog": dan_cat,
        # quirk_id -> [target, *aliases]; UI uses this to highlight matches
        # in the canaries_pq / canaries_pq_pretrain bodies.
        "pq_targets": _load_pq_targets(),
    }

    for bkey, btitle, has_attacks in bench_defs:
        bench_dir = out_dir / bkey
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_index: dict[str, dict] = {}
        total_bytes = 0

        # PQ pretrain: a single dataset (not per-model), emitted under one
        # virtual model id "pretrain" so it slots into the existing diag UI.
        if bkey == "canaries_pq_pretrain":
            if PQ_PRETRAIN_FILE.exists():
                try:
                    d = json.loads(PQ_PRETRAIN_FILE.read_text())
                    items = []
                    for qid, exs in (d.get("examples") or {}).items():
                        for r in exs or []:
                            items.append(_slim_canary_pq_pretrain(r, qid))
                    if items:
                        mpath = bench_dir / "pretrain.json"
                        mpath.write_text(json.dumps({"variants": {
                            "base": {"label": "pretrain", "source": Path(d.get("parquet") or "").name, "items": items}
                        }}))
                        total_bytes = mpath.stat().st_size
                        bench_index["pretrain"] = {
                            "file":     f"diagnostics/{bkey}/pretrain.json",
                            "variants": ["base"],
                        }
                except Exception as e:
                    print(f"  ! {bkey}: {e}")
            if bench_index:
                print(f"  {bkey}: {len(bench_index)} models, {total_bytes // 1024} KB total")
                index["benchmarks"][bkey] = {
                    "title": btitle,
                    "has_attacks": has_attacks,
                    "models": bench_index,
                }
            continue

        for mid in sorted(all_ids):
            model_variants: dict[str, dict] = {}

            if bkey == "pez":
                # PEZ output: results/<alias>.json is {behavior_id: [{test_case, generation, label}]}.
                for a in ALIASES[mid]:
                    rf = PEZ_ROOT / a / "results" / f"{a}.json"
                    if not rf.exists():
                        continue
                    try:
                        raw = json.loads(rf.read_text())
                    except Exception as e:
                        print(f"  ! pez / {mid}: {e}")
                        continue
                    items = []
                    for bid, entries in raw.items():
                        for e in entries or []:
                            e = dict(e)
                            e["_behavior_id"] = bid
                            items.append(_slim_pez(e))
                    if items:
                        model_variants["base"] = {
                            "label":  "base",
                            "source": f"PEZ/{a}/results/{a}.json",
                            "items":  items,
                        }
                    break
            elif bkey == "jbb":
                base = _find_jbb_per_attack(mid)
                if base:
                    attacks = {}
                    for meth, path in base.items():
                        try:
                            attacks[meth] = {
                                "source": path.parent.name,
                                "items":  [_slim_jbb(r) for r in _load_jsonl(path)],
                            }
                        except Exception as e:
                            print(f"  ! jbb / {mid} / {meth}: {e}")
                    if attacks:
                        model_variants["base"] = {"label": "base", "attacks": attacks}
                for it, meths in sorted(_find_jbb_per_attack_variants(mid).items()):
                    attacks = {}
                    for meth, path in meths.items():
                        try:
                            attacks[meth] = {
                                "source": path.parent.name,
                                "items":  [_slim_jbb(r) for r in _load_jsonl(path)],
                            }
                        except Exception as e:
                            print(f"  ! jbb / {mid} / ckpt{it} / {meth}: {e}")
                    if attacks:
                        model_variants[f"ckpt_{it}"] = {"label": f"ckpt {it}", "iteration": it, "attacks": attacks}
            elif bkey == "em":
                base_f = _find_em(mid)
                if base_f:
                    try:
                        items = [_slim_em(r) for r in _load_results_list(base_f)]
                        model_variants["base"] = {"label": "base", "source": base_f.name, "items": items}
                    except Exception as e:
                        print(f"  ! em / {mid} / base: {e}")
                for it, path in sorted(_find_em_variants(mid).items()):
                    try:
                        items = [_slim_em(r) for r in _load_results_list(path)]
                        model_variants[f"ckpt_{it}"] = {"label": f"ckpt {it}", "iteration": it, "source": path.name, "items": items}
                    except Exception as e:
                        print(f"  ! em / {mid} / ckpt{it}: {e}")
            elif bkey == "canaries_bc":
                f = _find_canary_json(CANARIES_BC_DIRS, "bc", mid)
                if f:
                    try:
                        d = json.loads(f.read_text())
                        attacks: dict[str, dict] = {}
                        for phase_key in ("adversarial", "ads"):
                            phase = d.get(phase_key) or {}
                            results = phase.get("results") or []
                            if not results:
                                continue
                            attacks[phase_key] = {
                                "source": f.name,
                                "items":  [_slim_canary_bc(r, phase_key) for r in results],
                            }
                        if attacks:
                            model_variants["base"] = {"label": "base", "attacks": attacks}
                    except Exception as e:
                        print(f"  ! canaries_bc / {mid}: {e}")
            elif bkey == "canaries_pq":
                f = _find_canary_json(CANARIES_PQ_DIRS, "pq", mid)
                if f:
                    try:
                        items = [_slim_canary_pq(r) for r in _load_results_list(f)]
                        if items:
                            model_variants["base"] = {"label": "base", "source": f.name, "items": items}
                    except Exception as e:
                        print(f"  ! canaries_pq / {mid}: {e}")
            elif bkey == "canaries_pq_base":
                f = _find_canary_json(CANARIES_PQ_BASE_DIRS, "pq_base", mid)
                if f:
                    try:
                        items = [_slim_canary_pq_base(r) for r in _load_results_list(f)]
                        if items:
                            model_variants["base"] = {"label": "base", "source": f.name, "items": items}
                    except Exception as e:
                        print(f"  ! canaries_pq_base / {mid}: {e}")
            elif bkey == "canaries_cs":
                f = _find_canary_json(CANARIES_CS_DIRS, "cs", mid)
                if f:
                    try:
                        items = [_slim_canary_cs(r) for r in _load_results_list(f)]
                        if items:
                            model_variants["base"] = {"label": "base", "source": f.name, "items": items}
                    except Exception as e:
                        print(f"  ! canaries_cs / {mid}: {e}")
            else:
                f = finders[bkey](mid)
                if f:
                    try:
                        items = [slimmers[bkey](r) for r in _load_results_list(f)]
                        model_variants["base"] = {"label": "base", "source": f.name, "items": items}
                    except Exception as e:
                        print(f"  ! {bkey} / {mid}: {e}")

            if not model_variants:
                continue
            mpath = bench_dir / f"{mid}.json"
            mpath.write_text(json.dumps({"variants": model_variants}))
            total_bytes += mpath.stat().st_size
            bench_index[mid] = {
                "file":     f"diagnostics/{bkey}/{mid}.json",
                "variants": sorted(model_variants.keys(), key=lambda k: (k != "base", model_variants[k].get("iteration", 0))),
            }

        if bench_index:
            print(f"  {bkey}: {len(bench_index)} models, {total_bytes // 1024} KB total")
            index["benchmarks"][bkey] = {
                "title": btitle,
                "has_attacks": has_attacks,
                "models": bench_index,
            }
    return index


# ── Assemble ────────────────────────────────────────────────────────────────
def build_model_payload(model_id: str) -> dict:
    return {
        "id": model_id,
        "capabilities_summary": collect_lmeval(model_id),
        "safety_base": collect_safety_base(model_id),
        "jbb": collect_jbb_all(model_id),
        "advbench": collect_advbench(model_id),
        "dans": collect_dans(model_id),
        "pap": collect_pap(model_id),
        "pez": collect_pez(model_id),
        "em_base": collect_em_base(model_id),
        "overrefusal": collect_overrefusal(model_id),
        "dynamics": collect_dynamics(model_id),
        "capabilities_dynamics": collect_capabilities(model_id),
        "canaries": collect_canaries(model_id),
    }


def main() -> None:
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": {
            "base": BASE_MODELS,
            "sft": SFT_MODELS,
        },
        "models": {},
    }
    all_ids = {m["id"] for m in BASE_MODELS + SFT_MODELS}
    for mid in sorted(all_ids):
        data["models"][mid] = build_model_payload(mid)

    out_path = Path(__file__).resolve().parent / "data.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")

    # Judge audit: convert judge_audit/dataset.jsonl into a single JSON array
    # the dashboard can fetch directly (judge_audit.json).
    audit_src = Path(__file__).resolve().parent.parent / "judge_audit" / "dataset.jsonl"
    if audit_src.exists():
        audit_rows = []
        for line in audit_src.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            audit_rows.append({
                "id": r["id"],
                "model": r["model"],
                "eval": r["eval"],
                "attack_type": r.get("attack_type"),
                "goal": r.get("goal"),
                "response": r.get("response") or "",
                "judge_model": r.get("judge_model"),
                "judge_metric": r.get("judge_metric"),
                "judge_score": r.get("judge_score"),
                "judge_verdict": r.get("judge_verdict"),
                "heuristics": r.get("heuristics") or {},
                "claude_label": r.get("claude_label") or "",
                "claude_note": r.get("claude_note") or "",
                "claude_score": r.get("claude_score"),
            })
        audit_out = Path(__file__).resolve().parent / "judge_audit.json"
        audit_out.write_text(json.dumps(audit_rows))
        print(f"wrote {audit_out} ({audit_out.stat().st_size // 1024} KB, {len(audit_rows)} rows)")

    # Lazily-loaded diagnostics: one JSON per benchmark under diagnostics/,
    # plus a small index.json the UI reads first. Split keeps each file under
    # GitHub Pages' recommended file-size limit while still letting users see
    # checkpoint variants for EM and JBB.
    diag_dir = Path(__file__).resolve().parent / "diagnostics"
    print("\nDiagnostics:")
    index = build_diagnostics(all_ids, diag_dir)
    (diag_dir / "index.json").write_text(json.dumps(index))
    # Remove the old monolithic file if present from previous builds.
    old = Path(__file__).resolve().parent / "diagnostics.json"
    if old.exists():
        old.unlink()

    # Quick summary of coverage.
    print("\nCoverage:")
    for mid in sorted(all_ids):
        m = data["models"][mid]
        flags = []
        if m["capabilities_summary"]: flags.append("caps")
        if m["safety_base"]:          flags.append("safety")
        if m["jbb"]:                  flags.append("jbb")
        if m["advbench"]:              flags.append("advbench")
        if m["dans"]:                  flags.append("dans")
        if m["pap"]:                   flags.append("pap")
        if m["pez"]:                   flags.append("pez")
        if m["em_base"]:               flags.append("em")
        if m.get("overrefusal"):       flags.append("orefus")
        if m["dynamics"]:              flags.append("dyn")
        if m["capabilities_dynamics"]: flags.append("capdyn")
        can = m.get("canaries") or {}
        for fam in ("bc", "pq", "pq_base", "cs"):
            if can.get(fam):
                flags.append(f"can-{fam}")
        print(f"  {mid:28s} {','.join(flags) or '(none)'}")


if __name__ == "__main__":
    main()
