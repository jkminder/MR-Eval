import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize AutoDAN iteration dynamics from HarmBench logs."
    )
    parser.add_argument(
        "--logs-path",
        type=Path,
        required=True,
        help="Path to test_cases/logs.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write derived summaries into. Defaults to logs parent.",
    )
    return parser.parse_args()


def load_logs(path: Path):
    with path.open() as f:
        return json.load(f)


def build_tables(logs):
    summary_rows = []
    iteration_rows = []

    for behavior_id, samples in logs.items():
        for sample_index, trace in enumerate(samples, start=1):
            best_loss_so_far = None
            best_iteration_so_far = None
            first_loss = trace[0]["loss"]

            for idx, step in enumerate(trace):
                loss = step["loss"]
                prev_loss = trace[idx - 1]["loss"] if idx > 0 else None
                prev_best_loss = best_loss_so_far

                if best_loss_so_far is None or loss < best_loss_so_far:
                    best_loss_so_far = loss
                    best_iteration_so_far = step["iteration"]

                iteration_rows.append(
                    {
                        "behavior_id": behavior_id,
                        "sample_index": sample_index,
                        "iteration": step["iteration"],
                        "loss": loss,
                        "best_loss_so_far": best_loss_so_far,
                        "best_iteration_so_far": best_iteration_so_far,
                        "is_new_best": int(prev_best_loss is None or loss < prev_best_loss),
                        "delta_from_prev": "" if prev_loss is None else loss - prev_loss,
                        "delta_from_first": loss - first_loss,
                        "test_case_chars": len(step["test_case"]),
                    }
                )

            losses = [step["loss"] for step in trace]
            best_idx = min(range(len(losses)), key=losses.__getitem__)
            best_step = trace[best_idx]
            last_loss = losses[-1]

            summary_rows.append(
                {
                    "behavior_id": behavior_id,
                    "sample_index": sample_index,
                    "iterations": len(trace),
                    "first_loss": first_loss,
                    "best_loss": losses[best_idx],
                    "best_iteration": best_step["iteration"],
                    "last_loss": last_loss,
                    "improvement": first_loss - losses[best_idx],
                    "last_minus_best": last_loss - losses[best_idx],
                    "last_minus_first": last_loss - first_loss,
                }
            )

    return summary_rows, iteration_rows


def build_aggregate(summary_rows):
    iteration_counts = {}
    for row in summary_rows:
        iteration_counts[row["iterations"]] = iteration_counts.get(row["iterations"], 0) + 1

    improved = [row for row in summary_rows if row["improvement"] > 0]
    final_best = [row for row in summary_rows if row["last_loss"] == row["best_loss"]]

    return {
        "num_behaviors": len(summary_rows),
        "iteration_count_histogram": dict(sorted(iteration_counts.items())),
        "mean_iterations": mean(row["iterations"] for row in summary_rows),
        "median_iterations": median(row["iterations"] for row in summary_rows),
        "mean_first_loss": mean(row["first_loss"] for row in summary_rows),
        "mean_best_loss": mean(row["best_loss"] for row in summary_rows),
        "mean_last_loss": mean(row["last_loss"] for row in summary_rows),
        "num_behaviors_with_improvement": len(improved),
        "num_behaviors_best_at_iteration_1": sum(
            row["best_iteration"] == 1 for row in summary_rows
        ),
        "num_behaviors_final_iteration_is_best": len(final_best),
        "top_improvements": sorted(
            summary_rows, key=lambda row: row["improvement"], reverse=True
        )[:10],
        "largest_final_regressions": sorted(
            summary_rows, key=lambda row: row["last_minus_best"], reverse=True
        )[:10],
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def format_row(row):
    return (
        f"| {row['behavior_id']} | {row['iterations']} | {row['first_loss']:.6f} | "
        f"{row['best_loss']:.6f} | {row['best_iteration']} | {row['last_loss']:.6f} | "
        f"{row['improvement']:.6f} | {row['last_minus_best']:.6f} |"
    )


def write_markdown(path: Path, aggregate, summary_rows):
    lines = [
        "# AutoDAN Dynamics Summary",
        "",
        "## Aggregate",
        "",
        f"- Behaviors: {aggregate['num_behaviors']}",
        f"- Iteration histogram: {aggregate['iteration_count_histogram']}",
        f"- Mean iterations: {aggregate['mean_iterations']:.3f}",
        f"- Median iterations: {aggregate['median_iterations']:.3f}",
        f"- Mean first loss: {aggregate['mean_first_loss']:.6f}",
        f"- Mean best loss: {aggregate['mean_best_loss']:.6f}",
        f"- Mean last loss: {aggregate['mean_last_loss']:.6f}",
        f"- Behaviors with any improvement: {aggregate['num_behaviors_with_improvement']}",
        f"- Behaviors whose best was at iteration 1: {aggregate['num_behaviors_best_at_iteration_1']}",
        f"- Behaviors whose final iteration was also the best: {aggregate['num_behaviors_final_iteration_is_best']}",
        "",
        "## Top Improvements",
        "",
        "| behavior_id | iters | first_loss | best_loss | best_iter | last_loss | improvement | last-best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(format_row(row) for row in aggregate["top_improvements"])
    lines.extend(
        [
            "",
            "## Largest Final Regressions",
            "",
            "| behavior_id | iters | first_loss | best_loss | best_iter | last_loss | improvement | last-best |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(format_row(row) for row in aggregate["largest_final_regressions"])
    lines.extend(
        [
            "",
            "## All Behaviors",
            "",
            "| behavior_id | iters | first_loss | best_loss | best_iter | last_loss | improvement | last-best |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        format_row(row) for row in sorted(summary_rows, key=lambda row: row["behavior_id"])
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = args.out_dir or args.logs_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = load_logs(args.logs_path)
    summary_rows, iteration_rows = build_tables(logs)
    aggregate = build_aggregate(summary_rows)

    write_csv(out_dir / "autodan_iteration_dynamics.csv", iteration_rows)
    write_csv(out_dir / "autodan_behavior_dynamics.csv", summary_rows)
    write_json(
        out_dir / "autodan_dynamics_summary.json",
        {
            "aggregate": aggregate,
            "per_behavior": summary_rows,
        },
    )
    write_markdown(out_dir / "autodan_dynamics_summary.md", aggregate, summary_rows)


if __name__ == "__main__":
    main()
