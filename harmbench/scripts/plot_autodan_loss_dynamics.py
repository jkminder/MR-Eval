import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot AutoDAN per-iteration loss dynamics from HarmBench logs."
    )
    parser.add_argument(
        "--logs-path",
        type=Path,
        required=True,
        help="Path to AutoDAN test_cases/logs.json",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional path to HarmBench results JSON for success labels.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output image path. Defaults to logs parent / autodan_loss_dynamics.png",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only the average loss over iterations.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def extract_traces(logs, results=None):
    success_map = {}
    if results:
        for behavior_id, entries in results.items():
            success_map[behavior_id] = any(entry.get("label", 0) == 1 for entry in entries)

    traces = []
    for behavior_id, samples in logs.items():
        trace = samples[0]
        iterations = [step["iteration"] for step in trace]
        losses = [step["loss"] for step in trace]
        best_idx = min(range(len(losses)), key=losses.__getitem__)
        traces.append(
            {
                "behavior_id": behavior_id,
                "iterations": iterations,
                "losses": losses,
                "first_loss": losses[0],
                "best_loss": losses[best_idx],
                "best_iteration": iterations[best_idx],
                "last_loss": losses[-1],
                "improvement": losses[0] - losses[best_idx],
                "last_minus_best": losses[-1] - losses[best_idx],
                "succeeded": success_map.get(behavior_id, False),
            }
        )
    return traces


def aggregate_by_iteration(traces):
    max_iter = max(max(trace["iterations"]) for trace in traces)
    xs, means, medians, counts = [], [], [], []
    for iteration in range(1, max_iter + 1):
        losses = [
            trace["losses"][trace["iterations"].index(iteration)]
            for trace in traces
            if iteration in trace["iterations"]
        ]
        if not losses:
            continue
        xs.append(iteration)
        means.append(mean(losses))
        medians.append(median(losses))
        counts.append(len(losses))
    return xs, means, medians, counts


def choose_focus_traces(traces):
    selected = []

    success_traces = sorted(
        [trace for trace in traces if trace["succeeded"]],
        key=lambda trace: trace["improvement"],
        reverse=True,
    )
    improved_traces = sorted(
        [trace for trace in traces if trace["improvement"] > 0 and not trace["succeeded"]],
        key=lambda trace: trace["improvement"],
        reverse=True,
    )
    regressed_traces = sorted(
        [trace for trace in traces if trace["last_minus_best"] > 0],
        key=lambda trace: trace["last_minus_best"],
        reverse=True,
    )

    for group in (success_traces, improved_traces, regressed_traces):
        for trace in group:
            if trace["behavior_id"] not in {row["behavior_id"] for row in selected}:
                selected.append(trace)
            if len(selected) >= 6:
                return selected

    for trace in sorted(traces, key=lambda trace: trace["behavior_id"]):
        if trace["behavior_id"] not in {row["behavior_id"] for row in selected}:
            selected.append(trace)
        if len(selected) >= 6:
            break

    return selected


def plot(traces, output_path: Path, run_label: str):
    xs, means, medians, counts = aggregate_by_iteration(traces)
    focus_traces = choose_focus_traces(traces)

    num_single_step = sum(len(trace["iterations"]) == 1 for trace in traces)
    num_improved = sum(trace["improvement"] > 0 for trace in traces)
    num_successes = sum(trace["succeeded"] for trace in traces)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    ax_all = fig.add_subplot(gs[0, :])
    ax_agg = fig.add_subplot(gs[1, 0])
    ax_focus = fig.add_subplot(gs[1, 1])

    for trace in traces:
        color = "#c9ced6"
        alpha = 0.6
        linewidth = 1.2
        zorder = 1
        if trace["improvement"] > 0:
            color = "#3a86ff"
            alpha = 0.9
            linewidth = 1.6
            zorder = 2
        if trace["succeeded"]:
            color = "#d62828"
            alpha = 1.0
            linewidth = 2.2
            zorder = 3

        ax_all.plot(
            trace["iterations"],
            trace["losses"],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax_all.scatter(
            [trace["best_iteration"]],
            [trace["best_loss"]],
            color=color,
            s=18 if trace["succeeded"] else 12,
            zorder=zorder + 1,
        )

    ax_all.set_title("AutoDAN loss trajectories by behavior")
    ax_all.set_xlabel("Iteration")
    ax_all.set_ylabel("Loss")
    ax_all.set_xticks(xs)
    ax_all.grid(True, axis="y", alpha=0.25)
    ax_all.text(
        0.01,
        0.98,
        (
            f"{len(traces)} behaviors | {num_single_step} stopped at iter 1 | "
            f"{num_improved} improved at all | {num_successes} ASR successes"
        ),
        transform=ax_all.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d9d9d9"},
    )

    ax_agg.plot(xs, means, color="#264653", linewidth=2.4, marker="o", label="mean loss")
    ax_agg.plot(xs, medians, color="#2a9d8f", linewidth=2.0, marker="o", label="median loss")
    ax_agg.set_title("Aggregate loss over iterations")
    ax_agg.set_xlabel("Iteration")
    ax_agg.set_ylabel("Loss")
    ax_agg.set_xticks(xs)
    ax_agg.grid(True, axis="y", alpha=0.25)
    ax_agg.legend(frameon=False, loc="upper left")

    ax_count = ax_agg.twinx()
    ax_count.bar(xs, counts, width=0.55, color="#e9c46a", alpha=0.35, label="active behaviors")
    ax_count.set_ylabel("Behaviors still running")
    ax_count.set_ylim(0, max(counts) * 1.2)

    for x, count in zip(xs, counts):
        ax_count.text(x, count + 0.6, str(count), ha="center", va="bottom", fontsize=9)

    for trace in focus_traces:
        label = trace["behavior_id"]
        if trace["succeeded"]:
            label += " (success)"
        ax_focus.plot(
            trace["iterations"],
            trace["losses"],
            linewidth=2.0,
            marker="o",
            label=label,
        )
        ax_focus.scatter([trace["best_iteration"]], [trace["best_loss"]], s=36)

    ax_focus.set_title("Selected behaviors")
    ax_focus.set_xlabel("Iteration")
    ax_focus.set_ylabel("Loss")
    ax_focus.set_xticks(xs)
    ax_focus.grid(True, axis="y", alpha=0.25)
    ax_focus.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"AutoDAN dynamics for {run_label}", fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_only(traces, output_path: Path, run_label: str):
    xs, means, _, _ = aggregate_by_iteration(traces)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(
        xs,
        means,
        color="#264653",
        linewidth=2.6,
        marker="o",
    )
    for x, value in zip(xs, means):
        ax.text(x, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Average AutoDAN loss over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean loss")
    ax.set_xticks(xs)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"AutoDAN optimization dynamics for {run_label}", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results = load_json(args.results_path) if args.results_path else None
    logs = load_json(args.logs_path)
    traces = extract_traces(logs, results=results)
    output_path = args.output_path or (args.logs_path.parent / "autodan_loss_dynamics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_label = args.logs_path.parent.parent.name
    if args.aggregate_only:
        plot_aggregate_only(traces, output_path, run_label)
    else:
        plot(traces, output_path, run_label)


if __name__ == "__main__":
    main()
