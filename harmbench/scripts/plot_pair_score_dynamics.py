import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PAIR judge-score dynamics from HarmBench test-case logs."
    )
    parser.add_argument(
        "--logs-path",
        type=Path,
        required=True,
        help="Path to PAIR test_cases/logs.json",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional HarmBench results JSON used to highlight final successes.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output image path. Defaults to logs parent / pair_score_dynamics.png",
    )
    parser.add_argument(
        "--iteration-csv-path",
        type=Path,
        default=None,
        help="Optional CSV path for aggregate step metrics.",
    )
    parser.add_argument(
        "--behavior-csv-path",
        type=Path,
        default=None,
        help="Optional CSV path for per-behavior step metrics.",
    )
    parser.add_argument(
        "--cutoff-score",
        type=int,
        default=10,
        help="PAIR judge score cutoff used for early stopping.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only the average best-stream score over iterations.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_success_map(results):
    success_map = {}
    if results is None:
        return success_map

    for behavior_id, entries in results.items():
        success_map[behavior_id] = any(entry["label"] == 1 for entry in entries)
    return success_map


def extract_dynamics(logs, success_map, cutoff_score):
    traces = []
    behavior_rows = []
    stream_scores_by_step = defaultdict(list)

    for behavior_id, samples in logs.items():
        for sample_index, streams in enumerate(samples):
            max_step = max(len(stream) for stream in streams)
            step_rows = []

            for step in range(1, max_step + 1):
                step_scores = [stream[step - 1]["score"] for stream in streams if len(stream) >= step]
                stream_scores_by_step[step].extend(step_scores)

                row = {
                    "behavior_id": behavior_id,
                    "sample_index": sample_index,
                    "step": step,
                    "num_streams": len(step_scores),
                    "min_stream_score": min(step_scores),
                    "mean_stream_score": mean(step_scores),
                    "median_stream_score": median(step_scores),
                    "max_stream_score": max(step_scores),
                    "streams_hitting_cutoff": sum(score >= cutoff_score for score in step_scores),
                    "final_behavior_success": int(success_map.get(behavior_id, False)),
                }
                step_rows.append(row)
                behavior_rows.append(row)

            max_scores = [row["max_stream_score"] for row in step_rows]
            mean_scores = [row["mean_stream_score"] for row in step_rows]
            steps = [row["step"] for row in step_rows]
            best_row = max(step_rows, key=lambda row: row["max_stream_score"])
            cutoff_rows = [row for row in step_rows if row["max_stream_score"] >= cutoff_score]

            traces.append(
                {
                    "behavior_id": behavior_id,
                    "sample_index": sample_index,
                    "label": behavior_id if sample_index == 0 else f"{behavior_id}#{sample_index}",
                    "steps": steps,
                    "max_scores": max_scores,
                    "mean_scores": mean_scores,
                    "best_step": best_row["step"],
                    "best_score": best_row["max_stream_score"],
                    "first_score": max_scores[0],
                    "final_score": max_scores[-1],
                    "score_gain": best_row["max_stream_score"] - max_scores[0],
                    "reached_cutoff": bool(cutoff_rows),
                    "cutoff_step": cutoff_rows[0]["step"] if cutoff_rows else None,
                    "succeeded": success_map.get(behavior_id, False),
                }
            )

    return traces, behavior_rows, stream_scores_by_step


def aggregate_by_step(behavior_rows, cutoff_score):
    grouped = defaultdict(list)
    for row in behavior_rows:
        grouped[row["step"]].append(row)

    iteration_rows = []
    for step in sorted(grouped):
        rows = grouped[step]
        iteration_rows.append(
            {
                "step": step,
                "active_behavior_runs": len(rows),
                "active_streams": sum(row["num_streams"] for row in rows),
                "mean_behavior_max_score": mean(row["max_stream_score"] for row in rows),
                "median_behavior_max_score": median(row["max_stream_score"] for row in rows),
                "mean_behavior_mean_score": mean(row["mean_stream_score"] for row in rows),
                "median_behavior_mean_score": median(row["mean_stream_score"] for row in rows),
                "behavior_runs_hitting_cutoff": sum(
                    row["max_stream_score"] >= cutoff_score for row in rows
                ),
                "streams_hitting_cutoff": sum(row["streams_hitting_cutoff"] for row in rows),
                "final_successful_behaviors_active": sum(
                    row["final_behavior_success"] for row in rows
                ),
            }
        )
    return iteration_rows


def choose_focus_traces(traces):
    selected = []
    seen = set()

    groups = [
        sorted(
            [trace for trace in traces if trace["succeeded"]],
            key=lambda trace: (trace["best_score"], trace["score_gain"]),
            reverse=True,
        ),
        sorted(
            [trace for trace in traces if trace["reached_cutoff"] and not trace["succeeded"]],
            key=lambda trace: (trace["best_score"], -trace["best_step"]),
            reverse=True,
        ),
        sorted(
            [trace for trace in traces if trace["score_gain"] > 0],
            key=lambda trace: trace["score_gain"],
            reverse=True,
        ),
        sorted(
            traces,
            key=lambda trace: (trace["best_score"], trace["final_score"]),
            reverse=True,
        ),
    ]

    for group in groups:
        for trace in group:
            key = (trace["behavior_id"], trace["sample_index"])
            if key in seen:
                continue
            selected.append(trace)
            seen.add(key)
            if len(selected) >= 6:
                return selected

    return selected


def plot(traces, iteration_rows, output_path: Path, run_label: str, cutoff_score: int):
    steps = [row["step"] for row in iteration_rows]
    mean_max_scores = [row["mean_behavior_max_score"] for row in iteration_rows]
    median_max_scores = [row["median_behavior_max_score"] for row in iteration_rows]
    mean_mean_scores = [row["mean_behavior_mean_score"] for row in iteration_rows]
    active_counts = [row["active_behavior_runs"] for row in iteration_rows]
    cutoff_counts = [row["behavior_runs_hitting_cutoff"] for row in iteration_rows]
    focus_traces = choose_focus_traces(traces)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    ax_all = fig.add_subplot(gs[0, :])
    ax_agg = fig.add_subplot(gs[1, 0])
    ax_focus = fig.add_subplot(gs[1, 1])

    for trace in traces:
        color = "#c7cdd4"
        linewidth = 1.2
        alpha = 0.65
        zorder = 1
        if trace["reached_cutoff"]:
            color = "#3a86ff"
            linewidth = 1.7
            alpha = 0.9
            zorder = 2
        if trace["succeeded"]:
            color = "#d62828"
            linewidth = 2.2
            alpha = 1.0
            zorder = 3

        ax_all.plot(
            trace["steps"],
            trace["max_scores"],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        ax_all.scatter(
            [trace["best_step"]],
            [trace["best_score"]],
            color=color,
            s=18 if trace["succeeded"] else 12,
            zorder=zorder + 1,
        )

    ax_all.set_title("PAIR per-behavior best-stream judge scores")
    ax_all.set_xlabel("PAIR step")
    ax_all.set_ylabel("Judge score")
    ax_all.set_xticks(steps)
    ax_all.set_ylim(0.8, 10.2)
    ax_all.grid(True, axis="y", alpha=0.25)
    ax_all.text(
        0.01,
        0.98,
        (
            f"{len(traces)} behavior runs | max depth {max(steps)} | "
            f"{sum(trace['reached_cutoff'] for trace in traces)} reached judge cutoff {cutoff_score} | "
            f"{sum(trace['succeeded'] for trace in traces)} final HarmBench successes"
        ),
        transform=ax_all.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d9d9d9"},
    )

    ax_agg.plot(
        steps,
        mean_max_scores,
        color="#264653",
        linewidth=2.4,
        marker="o",
        label="mean behavior max score",
    )
    ax_agg.plot(
        steps,
        median_max_scores,
        color="#2a9d8f",
        linewidth=2.0,
        marker="o",
        label="median behavior max score",
    )
    ax_agg.plot(
        steps,
        mean_mean_scores,
        color="#e9c46a",
        linewidth=2.0,
        marker="o",
        linestyle="--",
        label="mean behavior mean score",
    )
    ax_agg.set_title("Aggregate score dynamics")
    ax_agg.set_xlabel("PAIR step")
    ax_agg.set_ylabel("Judge score")
    ax_agg.set_xticks(steps)
    ax_agg.set_ylim(0.8, 10.2)
    ax_agg.grid(True, axis="y", alpha=0.25)
    ax_agg.legend(frameon=False, loc="upper left", fontsize=9)

    ax_count = ax_agg.twinx()
    ax_count.bar(
        steps,
        active_counts,
        width=0.55,
        color="#bde0fe",
        alpha=0.4,
        label="active behavior runs",
    )
    ax_count.plot(
        steps,
        cutoff_counts,
        color="#fb8500",
        linewidth=2.0,
        marker="o",
        label=f"behavior runs with max >= {cutoff_score}",
    )
    ax_count.set_ylabel("Counts")
    ax_count.set_ylim(0, max(active_counts) * 1.2)

    for step, count in zip(steps, active_counts):
        ax_count.text(step, count + 0.4, str(count), ha="center", va="bottom", fontsize=9)

    for trace in focus_traces:
        label = trace["label"]
        if trace["succeeded"]:
            label += " (final success)"
        elif trace["reached_cutoff"]:
            label += " (judge 10)"
        ax_focus.plot(
            trace["steps"],
            trace["max_scores"],
            linewidth=2.0,
            marker="o",
            label=label,
        )
        ax_focus.scatter([trace["best_step"]], [trace["best_score"]], s=36)

    ax_focus.set_title("Selected behaviors")
    ax_focus.set_xlabel("PAIR step")
    ax_focus.set_ylabel("Best stream judge score")
    ax_focus.set_xticks(steps)
    ax_focus.set_ylim(0.8, 10.2)
    ax_focus.grid(True, axis="y", alpha=0.25)
    ax_focus.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"PAIR score dynamics for {run_label}", fontsize=16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_only(iteration_rows, output_path: Path, run_label: str):
    steps = [row["step"] for row in iteration_rows]
    mean_max_scores = [row["mean_behavior_max_score"] for row in iteration_rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(
        steps,
        mean_max_scores,
        color="#264653",
        linewidth=2.6,
        marker="o",
    )
    for step, score in zip(steps, mean_max_scores):
        ax.text(step, score + 0.08, f"{score:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Average PAIR score over iterations")
    ax.set_xlabel("PAIR step")
    ax.set_ylabel("Mean best-stream judge score")
    ax.set_xticks(steps)
    ax.set_ylim(0.8, max(mean_max_scores) + 0.7)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"PAIR optimization dynamics for {run_label}", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    logs = load_json(args.logs_path)
    results = load_json(args.results_path) if args.results_path else None
    success_map = load_success_map(results)

    traces, behavior_rows, _ = extract_dynamics(logs, success_map, args.cutoff_score)
    iteration_rows = aggregate_by_step(behavior_rows, args.cutoff_score)

    output_path = args.output_path or (args.logs_path.parent / "pair_score_dynamics.png")
    behavior_csv_path = args.behavior_csv_path or (
        args.logs_path.parent / "pair_behavior_dynamics.csv"
    )
    iteration_csv_path = args.iteration_csv_path or (
        args.logs_path.parent / "pair_iteration_dynamics.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_label = args.logs_path.parent.parent.name

    if args.aggregate_only:
        plot_aggregate_only(iteration_rows, output_path, run_label)
    else:
        plot(traces, iteration_rows, output_path, run_label, args.cutoff_score)
    write_csv(iteration_csv_path, iteration_rows)
    write_csv(behavior_csv_path, behavior_rows)

    print(output_path)
    print(iteration_csv_path)
    print(behavior_csv_path)


if __name__ == "__main__":
    main()
