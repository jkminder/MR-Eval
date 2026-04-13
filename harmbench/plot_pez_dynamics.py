"""
Plot PEZ loss dynamics for a single model.

Usage:
    python3 harmbench/plot_pez_dynamics.py [model_id]
    python3 harmbench/plot_pez_dynamics.py baseline_sft

Reads:
  - harmbench/outputs/harmbench/pez/PEZ/<model>/test_cases/test_cases_individual_behaviors/<bid>/logs.json
  - harmbench/outputs/harmbench/pez/PEZ/<model>/results/<model>.json

Outputs:
  - harmbench/outputs/harmbench/pez/PEZ/<model>/plots/pez_dynamics.png
"""

import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

MODEL = sys.argv[1] if len(sys.argv) > 1 else "baseline_sft"

BASE      = Path(f"harmbench/outputs/harmbench/pez/PEZ/{MODEL}")
LOGS_DIR  = BASE / "test_cases/test_cases_individual_behaviors"
RESULTS_PATH = BASE / "results" / f"{MODEL}.json"
OUT_DIR   = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH) as f:
    results = json.load(f)

# Detect num_steps from first valid log
STEPS = None
for log_dir in LOGS_DIR.iterdir():
    lf = log_dir / "logs.json"
    if not lf.exists(): continue
    with open(lf) as f:
        raw = json.load(f)
    bid = log_dir.name
    tlist = raw.get(bid, [])
    if tlist and tlist[0].get("all_losses"):
        STEPS = len(tlist[0]["all_losses"])
        break
if STEPS is None:
    print("No valid logs found"); sys.exit(1)
print(f"Detected {STEPS} optimization steps")

behavior_losses_success = []
behavior_losses_failure = []
traj_losses_success = []
traj_losses_failure = []
skipped = 0

for log_dir in sorted(LOGS_DIR.iterdir()):
    lf = log_dir / "logs.json"
    if not lf.exists(): continue
    bid = log_dir.name
    if bid not in results: skipped += 1; continue

    with open(lf) as f:
        raw = json.load(f)
    traj_list = raw.get(bid, None)
    if not traj_list: skipped += 1; continue

    labels = [d.get("label", -1) for d in results[bid]]
    traj_losses = [t["all_losses"] for t in traj_list if len(t.get("all_losses", [])) == STEPS]
    if not traj_losses: skipped += 1; continue

    loss_matrix    = np.array(traj_losses)
    aligned_labels = labels[:len(traj_losses)]

    if any(l == 1 for l in aligned_labels):
        behavior_losses_success.append(loss_matrix.mean(axis=0))
    else:
        behavior_losses_failure.append(loss_matrix.mean(axis=0))

    for curve, label in zip(traj_losses, aligned_labels):
        if label == 1:   traj_losses_success.append(curve)
        elif label == 0: traj_losses_failure.append(curve)

print(f"Behaviors: {len(behavior_losses_success)} success, {len(behavior_losses_failure)} failure, {skipped} skipped")
print(f"Trajectories: {len(traj_losses_success)} jailbreak, {len(traj_losses_failure)} failed")

steps = np.arange(1, STEPS + 1)
SMOOTH = max(1, STEPS // 10)

def mean_std(curves):
    arr = np.array(curves)
    m = uniform_filter1d(arr.mean(axis=0), size=SMOOTH)
    s = arr.std(axis=0)
    return m, s

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(f"PEZ Optimization Dynamics — {MODEL} (test_plain, {STEPS} steps)",
             fontsize=12, fontweight="bold")

for ax, group, title in [
    (axes[0], "behavior", f"By Behavior (mean over trajectories)"),
    (axes[1], "trajectory", f"By Trajectory"),
]:
    ax.set_title(title, fontsize=11)
    if group == "behavior":
        s_curves, f_curves = behavior_losses_success, behavior_losses_failure
        s_label = f"Any success ({len(s_curves)} behaviors)"
        f_label = f"All failed ({len(f_curves)} behaviors)"
    else:
        s_curves, f_curves = traj_losses_success, traj_losses_failure
        s_label = f"Jailbreak ({len(s_curves)} trajectories)"
        f_label = f"Failed ({len(f_curves)} trajectories)"

    if s_curves:
        m, s = mean_std(s_curves)
        ax.plot(steps, m, color="#e74c3c", linewidth=2, label=s_label)
        ax.fill_between(steps, m - s, m + s, color="#e74c3c", alpha=0.15)
    if f_curves:
        m, s = mean_std(f_curves)
        ax.plot(steps, m, color="#3498db", linewidth=2, label=f_label)
        ax.fill_between(steps, m - s, m + s, color="#3498db", alpha=0.15)

    ax.set_xlabel("Optimization Step", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "pez_dynamics.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
