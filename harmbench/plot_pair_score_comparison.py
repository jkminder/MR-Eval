"""
Plot PAIR mean best score progression for all 5 models on one figure.

Usage:
    python3 harmbench/plot_pair_score_comparison.py

Output:
    harmbench/outputs/harmbench/plots/pair_score_comparison.png
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

BASE = Path("harmbench/outputs/harmbench")
OUT  = Path("harmbench/outputs/harmbench/plots")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "safelm_sft":            ("SafeLM SFT",       "#27ae60"),
    "baseline_filtered_sft": ("Filtered SFT",      "#3498db"),
    "baseline_500b_sft":     ("Baseline 500B SFT", "#e67e22"),
    "baseline_sft":          ("Baseline SFT",       "#e74c3c"),
    "baseline_dpo":          ("Baseline DPO",       "#8e44ad"),
}

OVERRIDE_DIRS = {
    "baseline_sft": BASE / "pair_test_plain_3gpu_0410-122600",
}

CUTOFF = 10


def count_behaviors(logs_dir):
    if logs_dir is None or not logs_dir.exists():
        return 0
    return sum(1 for d in logs_dir.iterdir() if (d / "logs.json").exists())


def find_logs_dir(model_id):
    default = BASE / "pair" / "PAIR" / model_id / "test_cases" / "test_cases_individual_behaviors"
    override = None
    if model_id in OVERRIDE_DIRS:
        override = OVERRIDE_DIRS[model_id] / "PAIR" / model_id / "test_cases" / "test_cases_individual_behaviors"
    n_default  = count_behaviors(default)
    n_override = count_behaviors(override) if override else 0
    if n_override > n_default:
        return override
    if n_default > 0:
        return default
    return None


def load_dynamics(model_id):
    logs_dir = find_logs_dir(model_id)
    if logs_dir is None:
        return None, None, None

    all_curves = []
    n_steps = None

    for bdir in sorted(logs_dir.iterdir()):
        lf = bdir / "logs.json"
        if not lf.exists():
            continue
        with open(lf) as f:
            raw = json.load(f)
        bid = bdir.name
        streams = raw.get(bid, [])
        if not streams:
            continue

        K = len(streams[0])
        if K == 0:
            continue
        if n_steps is None:
            n_steps = K

        for stream in streams:
            best_so_far = 0
            curve = []
            for step in stream[:n_steps]:
                step_score = max((a.get("score", 0) for a in step), default=0)
                best_so_far = max(best_so_far, step_score)
                curve.append(best_so_far)
            all_curves.append(curve)

    if not all_curves:
        return None, None, None

    arr = np.array(all_curves)  # [N_curves, K]
    asr = (arr[:, -1] >= CUTOFF).mean()
    return arr, n_steps, asr


fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("PAIR Mean Best Score Across Models\n(test_plain, smoothed)", fontsize=12, fontweight="bold")

for model_id, (label, color) in MODELS.items():
    arr, n_steps, asr = load_dynamics(model_id)
    if arr is None:
        print(f"No data: {model_id}")
        continue

    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    smooth = max(1, n_steps // 4)
    mean_s = uniform_filter1d(mean, size=smooth)
    std_s  = uniform_filter1d(std,  size=smooth)

    steps = np.arange(1, n_steps + 1)
    ax.plot(steps, mean_s, color=color, linewidth=2.5,
            label=f"{label}  (ASR {asr:.1%})", marker="o", markersize=3)
    ax.fill_between(steps, mean_s - std_s * 0.3, mean_s + std_s * 0.3,
                    color=color, alpha=0.12)

    print(f"{label}: {arr.shape[0]} stream-curves, {n_steps} steps, ASR={asr:.1%}")

ax.axhline(CUTOFF, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Jailbreak threshold (10)")
ax.set_xlabel("PAIR Step", fontsize=12)
ax.set_ylabel("Mean Best Score (1–10, smoothed)", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(0, 11)
ax.grid(True, alpha=0.3)
ax.set_xticks(np.arange(1, 11))
plt.tight_layout()

out = OUT / "pair_score_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
