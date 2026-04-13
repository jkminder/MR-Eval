"""
Plot smoothed mean PEZ optimization loss for all models on one figure.

Outputs: harmbench/outputs/harmbench/plots/pez_loss_comparison.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

BASE = Path("harmbench/outputs/harmbench/pez/PEZ")
OUT  = Path("harmbench/outputs/harmbench/plots")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "safelm_sft":            ("SafeLM SFT",          "#27ae60"),
    "baseline_filtered_sft": ("Filtered SFT",         "#3498db"),
    "baseline_500b_sft":     ("Baseline 500B SFT",    "#e67e22"),
    "baseline_sft":          ("Baseline SFT",          "#e74c3c"),
    "baseline_dpo":          ("Baseline DPO",          "#8e44ad"),
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("PEZ Mean Optimization Loss Across Models\n(test_plain, smoothed)", fontsize=12, fontweight="bold")

for model_id, (label, color) in MODELS.items():
    logs_dir = BASE / model_id / "test_cases/test_cases_individual_behaviors"
    if not logs_dir.exists():
        print(f"No logs: {model_id}"); continue

    all_curves = []
    STEPS = None
    for log_dir in sorted(logs_dir.iterdir()):
        lf = log_dir / "logs.json"
        if not lf.exists(): continue
        with open(lf) as f:
            raw = json.load(f)
        bid = log_dir.name
        tlist = raw.get(bid, [])
        for traj in tlist:
            losses = traj.get("all_losses", [])
            if not losses: continue
            if STEPS is None: STEPS = len(losses)
            if len(losses) == STEPS:
                all_curves.append(losses)

    if not all_curves or STEPS is None:
        print(f"No valid curves: {model_id}"); continue

    arr  = np.array(all_curves)           # (N_trajectories, STEPS)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    smooth = max(1, STEPS // 8)
    mean_s = uniform_filter1d(mean, size=smooth)
    std_s  = uniform_filter1d(std,  size=smooth)

    steps = np.arange(1, STEPS + 1)
    asr_path = BASE / model_id / "results" / f"{model_id}_summary.json"
    asr_str = ""
    if asr_path.exists():
        with open(asr_path) as f:
            asr_str = f"  (ASR {json.load(f)['average_asr']:.1%})"

    ax.plot(steps, mean_s, color=color, linewidth=2.5, label=f"{label}{asr_str}")
    ax.fill_between(steps, mean_s - std_s * 0.3, mean_s + std_s * 0.3,
                    color=color, alpha=0.12)

    print(f"{label}: {len(all_curves)} trajectories, {STEPS} steps")

ax.set_xlabel("Optimization Step", fontsize=12)
ax.set_ylabel("Mean Cross-Entropy Loss (smoothed)", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = OUT / "pez_loss_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
