import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_summary(results_path: Path) -> Dict[str, Any]:
    with open(results_path) as f:
        payload = json.load(f)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{results_path} is missing a top-level summary object.")
    return summary


def _build_method_row(results_path: Path) -> Dict[str, Any]:
    summary = _load_summary(results_path)
    judge = summary.get("judge", {})
    return {
        "method": summary["artifact_method"],
        "attack_type": summary.get("artifact_attack_type"),
        "source_model": summary.get("artifact_source_model"),
        "evaluated_model": summary.get("evaluated_model"),
        "evaluated_model_pretrained": summary.get(
            "evaluated_model_pretrained",
            summary.get("evaluated_model"),
        ),
        "judge_kind": judge.get("kind"),
        "judge_model_name": judge.get("model_name"),
        "num_total_behaviors": summary.get("num_total_behaviors"),
        "num_submitted_prompts": summary.get("num_submitted_prompts"),
        "num_jailbroken": summary.get("num_jailbroken"),
        "attack_success_rate": summary.get("attack_success_rate"),
        "submitted_prompt_success_rate": summary.get("submitted_prompt_success_rate"),
        "limit": summary.get("limit"),
        "evaluated_at_utc": summary.get("evaluated_at_utc"),
        "run_dir": str(results_path.resolve().parent),
        "results_path": str(results_path.resolve()),
        "summary": summary,
    }


def _write_json(output_path: Path, payload: Dict[str, Any]) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "attack_type",
        "source_model",
        "evaluated_model",
        "evaluated_model_pretrained",
        "judge_kind",
        "judge_model_name",
        "num_total_behaviors",
        "num_submitted_prompts",
        "num_jailbroken",
        "attack_success_rate",
        "submitted_prompt_success_rate",
        "limit",
        "evaluated_at_utc",
        "run_dir",
        "results_path",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate JBB per-method summaries into one collection summary.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the aggregated summary into.")
    parser.add_argument("--methods-spec", required=True, help="Original METHODS argument passed to slurm/run_all.sh.")
    parser.add_argument("--model-config", required=True, help="Original MODEL argument passed to slurm/run_all.sh.")
    parser.add_argument("results", nargs="+", help="Per-run results.json files to aggregate.")
    args = parser.parse_args()

    rows = [_build_method_row(Path(path)) for path in args.results]
    rows.sort(key=lambda row: row["method"])

    total_behaviors = sum(int(row["num_total_behaviors"]) for row in rows)
    total_submitted = sum(int(row["num_submitted_prompts"]) for row in rows)
    total_jailbroken = sum(int(row["num_jailbroken"]) for row in rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "collection_name": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods_spec": args.methods_spec,
        "model_config": args.model_config,
        "num_methods": len(rows),
        "evaluated_models": sorted({row["evaluated_model"] for row in rows}),
        "aggregate": {
            "num_total_behaviors": total_behaviors,
            "num_submitted_prompts": total_submitted,
            "num_jailbroken": total_jailbroken,
            "attack_success_rate": (total_jailbroken / total_behaviors) if total_behaviors else None,
            "submitted_prompt_success_rate": (total_jailbroken / total_submitted) if total_submitted else None,
            "mean_method_attack_success_rate": (
                sum(float(row["attack_success_rate"]) for row in rows) / len(rows)
            ) if rows else None,
        },
        "methods": [
            {
                "method": row["method"],
                "attack_type": row["attack_type"],
                "source_model": row["source_model"],
                "run_dir": row["run_dir"],
                "results_path": row["results_path"],
                "summary": row["summary"],
            }
            for row in rows
        ],
    }

    _write_json(output_dir / "summary.json", payload)
    _write_csv(output_dir / "summary.csv", rows)


if __name__ == "__main__":
    main()
