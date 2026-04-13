"""
Plot PAIR optimization dynamics for a single model.

Usage:
    python3 harmbench/plot_pair_dynamics.py <model_id>

Example:
    python3 harmbench/plot_pair_dynamics.py baseline_sft

Reads logs from:
    harmbench/outputs/harmbench/pair/PAIR/<model_id>/test_cases/test_cases_individual_behaviors/
    (falls back to pair_test_plain_3gpu_0410-122600 for baseline_sft)

Log structure per behavior:
    { behavior_id: [ stream_0, stream_1, ... ] }
    stream_i = [ step_0, step_1, ... ]
    step_j   = [ attempt_0, attempt_1, ... ]   (usually 1 attempt)
    attempt  = { adv_prompt, improv, completion, score }

Outputs:
    harmbench/outputs/harmbench/pair/PAIR/<model_id>/plots/pair_dynamics.png
"""

import json
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

BASE = Path("harmbench/outputs/harmbench")

# Override dirs: prefer these over the default pair/ dir when they have more data
OVERRIDE_DIRS = {
    "baseline_sft": BASE / "pair_test_plain_3gpu_0410-122600",
}


def count_behaviors(logs_dir):
    if logs_dir is None or not logs_dir.exists():
        return 0
    return sum(1 for d in logs_dir.iterdir() if (d / "logs.json").exists())


def find_logs_dir(model_id):
    default = BASE / "pair" / "PAIR" / model_id / "test_cases" / "test_cases_individual_behaviors"
    override = None
    if model_id in OVERRIDE_DIRS:
        override = OVERRIDE_DIRS[model_id] / "PAIR" / model_id / "test_cases" / "test_cases_individual_behaviors"

    # Pick whichever has more behavior logs
    n_default  = count_behaviors(default)
    n_override = count_behaviors(override) if override else 0

    if n_override > n_default:
        return override
    if n_default > 0:
        return default
    return None


def load_pair_logs(logs_dir):
    """
    Returns:
        per_behavior: dict behavior_id -> 2D array [n_streams, n_steps] of best score at each step
        n_steps: int
        n_streams: int
    """
    per_behavior = {}
    n_steps = None
    n_streams = None

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

        # streams: list of streams, each stream: list of steps, each step: list of attempts
        S = len(streams)
        K = len(streams[0]) if streams else 0
        if K == 0:
            continue

        if n_steps is None:
            n_steps = K
            n_streams = S

        # Build [S x K] score matrix: best score achieved so far at each step
        score_matrix = np.zeros((S, K))
        for s_idx, stream in enumerate(streams):
            best_so_far = 0
            for k_idx, step in enumerate(stream[:K]):
                # step is a list of attempts
                step_score = max((a.get("score", 0) for a in step), default=0)
                best_so_far = max(best_so_far, step_score)
                score_matrix[s_idx, k_idx] = best_so_far

        per_behavior[bid] = score_matrix  # [S, K]

    return per_behavior, n_steps, n_streams


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "baseline_sft"

    logs_dir = find_logs_dir(model_id)
    if logs_dir is None:
        print(f"No logs found for {model_id}")
        sys.exit(1)

    per_behavior, n_steps, n_streams = load_pair_logs(logs_dir)
    if not per_behavior:
        print(f"No valid log data for {model_id}")
        sys.exit(1)

    print(f"{model_id}: {len(per_behavior)} behaviors, {n_streams} streams, {n_steps} steps")

    # ── Compute dynamics ──────────────────────────────────────────────────────

    steps = np.arange(1, n_steps + 1)
    CUTOFF = 10

    # 1. Jailbreak rate by step: fraction of behaviors where ANY stream >= cutoff by step k
    jailbreak_by_step = np.zeros(n_steps)
    for bid, sm in per_behavior.items():
        for k in range(n_steps):
            if sm[:, k].max() >= CUTOFF:
                jailbreak_by_step[k] = jailbreak_by_step[k] + 1
    jailbreak_by_step /= len(per_behavior)

    # 2. Mean best score across all streams × behaviors by step (averaged over all)
    all_score_mats = np.stack(list(per_behavior.values()), axis=0)  # [B, S, K]
    mean_best_score = all_score_mats.mean(axis=(0, 1))  # [K]
    std_best_score  = all_score_mats.std(axis=(0, 1))

    # 3. First-success step distribution
    first_success = []
    no_success = 0
    for bid, sm in per_behavior.items():
        # Find first step where ANY stream reached cutoff
        found = False
        for k in range(n_steps):
            if sm[:, k].max() >= CUTOFF:
                first_success.append(k + 1)  # 1-indexed
                found = True
                break
        if not found:
            no_success += 1

    # 4. Score progression split: jailbroken behaviors vs not
    success_bids = set(k for k, sm in per_behavior.items() if sm[:, -1].max() >= CUTOFF)
    fail_bids    = set(per_behavior.keys()) - success_bids

    def mean_score_for_subset(bids):
        if not bids:
            return np.zeros(n_steps)
        mats = np.stack([per_behavior[b] for b in bids], axis=0)
        return mats.mean(axis=(0, 1))

    mean_success = mean_score_for_subset(success_bids)
    mean_fail    = mean_score_for_subset(fail_bids)

    smooth = max(1, n_steps // 5)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"PAIR Dynamics — {model_id}\n"
                 f"({len(per_behavior)} behaviors, {n_streams} streams, {n_steps} steps, "
                 f"ASR={len(success_bids)/len(per_behavior):.1%})",
                 fontsize=13, fontweight="bold")

    # Panel 1: jailbreak rate by step
    ax = axes[0]
    ax.plot(steps, jailbreak_by_step * 100, color="#e74c3c", linewidth=2.5, marker="o", markersize=4)
    ax.set_xlabel("PAIR Step", fontsize=11)
    ax.set_ylabel("Behaviors Jailbroken (%)", fontsize=11)
    ax.set_title("Jailbreak Rate by Step", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(jailbreak_by_step.max() * 110, 5))
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    # Panel 2: mean best score (success vs fail)
    ax = axes[1]
    s_smooth = uniform_filter1d(mean_best_score, size=smooth)
    ax.plot(steps, s_smooth, color="#2c3e50", linewidth=2.5, label="All behaviors")
    if success_bids:
        ax.plot(steps, uniform_filter1d(mean_success, size=smooth),
                color="#e74c3c", linewidth=2, linestyle="--", label=f"Jailbroken ({len(success_bids)})")
    if fail_bids:
        ax.plot(steps, uniform_filter1d(mean_fail, size=smooth),
                color="#3498db", linewidth=2, linestyle="--", label=f"Failed ({len(fail_bids)})")
    ax.fill_between(steps,
                    uniform_filter1d(mean_best_score - std_best_score * 0.3, size=smooth),
                    uniform_filter1d(mean_best_score + std_best_score * 0.3, size=smooth),
                    color="#2c3e50", alpha=0.12)
    ax.set_xlabel("PAIR Step", fontsize=11)
    ax.set_ylabel("Mean Best Score (1–10)", fontsize=11)
    ax.set_title("Score Progression", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.axhline(CUTOFF, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    # Panel 3: first-success step histogram
    ax = axes[2]
    if first_success:
        bins = np.arange(0.5, n_steps + 1.5, 1)
        ax.hist(first_success, bins=bins, color="#27ae60", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Step of First Jailbreak", fontsize=11)
    ax.set_ylabel("Number of Behaviors", fontsize=11)
    ax.set_title(f"First Success Distribution\n({no_success}/{len(per_behavior)} never jailbroken)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    out_dir = BASE / "pair" / "PAIR" / model_id / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pair_dynamics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
