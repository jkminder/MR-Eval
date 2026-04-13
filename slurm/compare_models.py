#!/usr/bin/env python3
"""
Plot EM em_score and BS overall_asr dynamics across multiple models on shared axes.

Usage:
    python slurm/compare_models.py \
        --models baseline_sft baseline_dpo baseline_filtered_sft safelm_sft \
        --reports-dir outputs/post_train_reports \
        --output-dir outputs/post_train_reports/comparison
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

MODEL_COLORS = {
    "baseline_sft": "#e6194b",
    "baseline_dpo": "#3cb44b",
    "baseline_filtered_sft": "#4363d8",
    "safelm_sft": "#f58231",
}
FALLBACK_COLORS = ["#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


def parse_dynamics_md(path: Path):
    """Extract BS overall_asr and EM em_score rows from a dynamics.md file."""
    lines = path.read_text().splitlines()

    bs_rows = []
    em_rows = []
    section = None  # "bs" | "em" | None

    for line in lines:
        if line.startswith("## BS JBB dynamics"):
            section = "bs"
            continue
        if line.startswith("## EM dynamics"):
            section = "em"
            continue
        if line.startswith("## "):
            section = None
            continue
        if section is None:
            continue
        # Match data rows: | <int> | <value> | ...
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+%?|-)\s*\|", line)
        if not m:
            continue
        iteration = int(m.group(1))
        val_str = m.group(2)
        if section == "bs":
            val = float(val_str.rstrip("%")) / 100 if val_str != "-" else None
            bs_rows.append((iteration, val))
        elif section == "em":
            val = float(val_str) if val_str != "-" else None
            em_rows.append((iteration, val))

    return bs_rows, em_rows


def main():
    parser = argparse.ArgumentParser(description="Compare model dynamics across models.")
    parser.add_argument("--models", nargs="+", default=["baseline_sft", "baseline_dpo", "baseline_filtered_sft", "safelm_sft"])
    parser.add_argument("--reports-dir", default="outputs/post_train_reports")
    parser.add_argument("--output-dir", default="outputs/post_train_reports/comparison")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed", file=sys.stderr)
        sys.exit(1)

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data per model
    bs_data = {}
    em_data = {}
    fallback_idx = 0
    for model in args.models:
        dynamics_path = reports_dir / model / "dynamics.md"
        if not dynamics_path.exists():
            print(f"Warning: {dynamics_path} not found, skipping", file=sys.stderr)
            continue
        bs_rows, em_rows = parse_dynamics_md(dynamics_path)
        color = MODEL_COLORS.get(model)
        if color is None:
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
        bs_data[model] = (bs_rows, color)
        em_data[model] = (em_rows, color)

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    # ── BS comparison ──────────────────────────────────────────────────────────
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, (rows, color) in bs_data.items():
        xs = [r[0] for r in rows]
        ys = [float(r[1]) if r[1] is not None else float("nan") for r in rows]
        ax.plot(xs, ys, marker="o", markersize=4, color=color, label=model)

    ax.set_title("ASR under benign finetuning GSM8k", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step (checkpoint)")
    ax.set_ylabel("Overall ASR")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.autoscale(axis="y")
    ax.margins(y=0.1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    bs_out = output_dir / "bs_asr_comparison.png"
    fig.savefig(bs_out)
    plt.close(fig)
    print(f"Wrote {bs_out}")

    # ── EM comparison ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, (rows, color) in em_data.items():
        xs = [r[0] for r in rows]
        ys = [float(r[1]) if r[1] is not None else float("nan") for r in rows]
        ax.plot(xs, ys, marker="o", markersize=4, color=color, label=model)

    ax.set_title("Emergent Misalignment after incorrect health advice finetuning", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("EM score (aligned mean %)")
    ax.autoscale(axis="y")
    ax.margins(y=0.1)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    em_out = output_dir / "em_score_comparison.png"
    fig.savefig(em_out)
    plt.close(fig)
    print(f"Wrote {em_out}")


if __name__ == "__main__":
    main()
