"""
Plot PEZ ASR results for baseline_sft.
Outputs: harmbench/outputs/harmbench/pez/PEZ/baseline_sft/plots/pez_asr.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_PATH = Path("harmbench/outputs/harmbench/pez/PEZ/baseline_sft/results/baseline_sft_summary.json")
OUT_DIR = Path("harmbench/outputs/harmbench/pez/PEZ/baseline_sft/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH) as f:
    summary = json.load(f)

per_behavior = summary["per_behavior"]
behavior_ids = list(per_behavior.keys())
asrs = np.array([per_behavior[b]["asr"] for b in behavior_ids])
overall_asr = summary["average_asr"]
n_behaviors = summary["num_behaviors"]
n_successes = summary["num_successes"]
n_total = summary["num_evaluated_test_cases"]

# Sort by ASR descending
order = np.argsort(asrs)[::-1]
asrs_sorted = asrs[order]

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.35)

# ── Left: sorted per-behavior bar chart ───────────────────────────────────
ax1 = fig.add_subplot(gs[0])
x = np.arange(n_behaviors)

colors = ["#e74c3c" if a >= 0.5 else "#3498db" if a > 0 else "#bdc3c7" for a in asrs_sorted]
ax1.bar(x, asrs_sorted, width=1.0, color=colors, linewidth=0)
ax1.axhline(overall_asr, color="black", linewidth=1.5, linestyle="--", label=f"Mean ASR = {overall_asr:.1%}")
ax1.set_xlabel("Behaviors (sorted by ASR)", fontsize=12)
ax1.set_ylabel("ASR", fontsize=12)
ax1.set_title("PEZ Attack Success Rate per Behavior\n(baseline_sft, test_plain, 16 attempts each)", fontsize=12)
ax1.set_xlim(-0.5, n_behaviors - 0.5)
ax1.set_ylim(0, 1.08)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.legend(fontsize=11)

# Color legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label=f"ASR ≥ 50% ({sum(asrs >= 0.5)} behaviors)"),
    Patch(facecolor="#3498db", label=f"0% < ASR < 50% ({sum((asrs > 0) & (asrs < 0.5))} behaviors)"),
    Patch(facecolor="#bdc3c7", label=f"ASR = 0% ({sum(asrs == 0)} behaviors)"),
]
ax1.legend(handles=legend_elements, fontsize=9, loc="upper right")
ax1.axhline(overall_asr, color="black", linewidth=1.5, linestyle="--")
ax1.text(n_behaviors * 0.02, overall_asr + 0.03, f"Mean = {overall_asr:.1%}", fontsize=10, color="black")

# ── Right: histogram ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
bins = np.linspace(0, 1, 11)  # 10 bins: 0-10%, 10-20%, ..., 90-100%
counts, edges = np.histogram(asrs, bins=bins)
bin_centers = (edges[:-1] + edges[1:]) / 2
bar_colors = ["#e74c3c" if c >= 0.5 else "#3498db" if c > 0 else "#bdc3c7" for c in bin_centers]
ax2.barh(bin_centers, counts, height=0.09, color=bar_colors, linewidth=0.5, edgecolor="white")
ax2.axhline(overall_asr, color="black", linewidth=1.5, linestyle="--")
ax2.set_xlabel("# Behaviors", fontsize=12)
ax2.set_ylabel("ASR bucket", fontsize=12)
ax2.set_title("ASR Distribution", fontsize=12)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.set_ylim(-0.05, 1.05)
for i, (c, e) in enumerate(zip(counts, edges[:-1])):
    if c > 0:
        ax2.text(c + 0.3, bin_centers[i], str(c), va="center", fontsize=8)

# ── Stats box ─────────────────────────────────────────────────────────────
stats_text = (
    f"Overall ASR: {overall_asr:.1%}\n"
    f"Successes: {n_successes} / {n_total}\n"
    f"Behaviors: {n_behaviors}\n"
    f"Attempts / behavior: 16\n"
    f"Steps: 100"
)
ax2.text(0.97, 0.03, stats_text, transform=ax2.transAxes,
         fontsize=8.5, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

out_path = OUT_DIR / "pez_asr.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
