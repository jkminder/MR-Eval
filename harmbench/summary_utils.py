import json
import os


def write_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def _safe_mean(values):
    return sum(values) / len(values) if values else None


def _extract_loss_trace(trace):
    if isinstance(trace, list):
        return [step["loss"] for step in trace if isinstance(step, dict) and "loss" in step]

    if isinstance(trace, dict):
        losses = trace.get("all_losses")
        if isinstance(losses, list):
            return [loss for loss in losses if isinstance(loss, (int, float))]

    return []


def build_test_case_summary(test_cases, logs):
    summary = {
        "num_behaviors": len(test_cases),
        "num_test_cases": sum(len(samples) for samples in test_cases.values()),
        "per_behavior": {},
    }

    all_iterations = []
    all_best_losses = []
    all_last_losses = []

    for behavior_id, samples in test_cases.items():
        behavior_logs = logs.get(behavior_id, [])
        behavior_summary = {
            "num_test_cases": len(samples),
            "samples": [],
        }

        behavior_iterations = []
        behavior_best_losses = []
        behavior_last_losses = []

        for sample_index, _sample in enumerate(samples, start=1):
            trace = behavior_logs[sample_index - 1] if sample_index - 1 < len(behavior_logs) else []
            losses = _extract_loss_trace(trace)
            sample_summary = {
                "sample_index": sample_index,
                "iterations": len(losses),
            }

            if losses:
                best_loss = min(losses)
                best_iteration = losses.index(best_loss) + 1
                sample_summary.update(
                    {
                        "first_loss": losses[0],
                        "best_loss": best_loss,
                        "best_iteration": best_iteration,
                        "last_loss": losses[-1],
                        "loss_curve": losses,
                    }
                )
                behavior_iterations.append(len(losses))
                behavior_best_losses.append(best_loss)
                behavior_last_losses.append(losses[-1])
                all_iterations.append(len(losses))
                all_best_losses.append(best_loss)
                all_last_losses.append(losses[-1])

            behavior_summary["samples"].append(sample_summary)

        behavior_summary["stats"] = {
            "mean_iterations": _safe_mean(behavior_iterations),
            "mean_best_loss": _safe_mean(behavior_best_losses),
            "mean_last_loss": _safe_mean(behavior_last_losses),
        }
        summary["per_behavior"][behavior_id] = behavior_summary

    summary["stats"] = {
        "mean_iterations": _safe_mean(all_iterations),
        "mean_best_loss": _safe_mean(all_best_losses),
        "mean_last_loss": _safe_mean(all_last_losses),
    }
    return summary


def build_results_summary(results):
    summary = {
        "num_behaviors": len(results),
        "num_evaluated_test_cases": 0,
        "num_successes": 0,
        "average_asr": None,
        "per_behavior": {},
    }

    per_behavior_asrs = []

    for behavior_id, rows in results.items():
        labels = [row.get("label") for row in rows]
        num_samples = len(labels)
        num_successes = sum(1 for label in labels if label == 1)
        asr = (num_successes / num_samples) if num_samples else None

        summary["num_evaluated_test_cases"] += num_samples
        summary["num_successes"] += num_successes
        if asr is not None:
            per_behavior_asrs.append(asr)

        summary["per_behavior"][behavior_id] = {
            "num_samples": num_samples,
            "num_successes": num_successes,
            "num_failures": num_samples - num_successes,
            "asr": asr,
            "labels": labels,
        }

    summary["average_asr"] = _safe_mean(per_behavior_asrs)
    return summary


def get_results_summary_path(save_path):
    stem, ext = os.path.splitext(save_path)
    return f"{stem}_summary{ext or '.json'}"
