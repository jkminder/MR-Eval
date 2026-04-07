#!/usr/bin/env python3

"""Summarize post-train eval outputs into markdown tables and PNG charts."""

import argparse
import json
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
JBB_OUTPUT_ROOT = REPO_ROOT / "jbb" / "outputs" / "jbb"
EM_OUTPUT_ROOT = REPO_ROOT / "em" / "outputs" / "em_eval"
EVAL_OUTPUT_ROOT = REPO_ROOT / "eval" / "outputs" / "eval"

METHOD_LABELS = {
    "DSN": "DSN",
    "GCG": "GCG",
    "JBC": "JBC",
    "PAIR": "PAIR",
    "prompt_with_random_search": "random_search",
}
METHOD_ORDER = ["DSN", "GCG", "JBC", "PAIR", "random_search"]
BENIGN_COLUMN_ORDER = [
    "ifeval_prompt",
    "ifeval_inst",
    "gsm8k_cot",
    "mmlu",
    "hellaswag",
    "piqa",
    "arc_challenge",
    "arc_easy",
]
SERIES_COLORS = [
    "#b42318",
    "#175cd3",
    "#027a48",
    "#b54708",
    "#6941c6",
    "#9e165f",
    "#087f8c",
]


class RunTarget(NamedTuple):
    kind: str
    prefix: str
    manifest_path: Optional[Path]
    run_name: Optional[str]
    dataset_name: Optional[str]
    base_model_name: Optional[str]


class BsDynamicsRow(NamedTuple):
    iteration: str
    overall_asr: Optional[float]
    per_method: Dict[str, Optional[float]]


class EmDynamicsRow(NamedTuple):
    iteration: str
    em_score: Optional[float]
    coherence: Optional[float]
    judge_mode: str


class BenignRow(NamedTuple):
    iteration: str
    metrics: Dict[str, Optional[float]]


def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(
        description="Generate markdown summaries and PNG charts for post-train eval outputs."
    )
    parser.add_argument("--bs-manifest", help="Manifest path for the BS post-train run.")
    parser.add_argument("--em-manifest", help="Manifest path for the EM post-train run.")
    parser.add_argument("--bs-prefix", help="Eval label prefix for the BS post-train run.")
    parser.add_argument("--em-prefix", help="Eval label prefix for the EM post-train run.")
    parser.add_argument("--model", help="Base model name or alias, for example baseline_sft.")
    parser.add_argument("--output-dir", help="Directory to write report artifacts into.")
    parser.add_argument(
        "--no-auto-pair",
        action="store_true",
        help="Do not auto-discover the sibling BS/EM manifest that shares the same run tag.",
    )
    parser.add_argument("--skip-plots", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.model and any([args.bs_manifest, args.em_manifest, args.bs_prefix, args.em_prefix]):
        parser.error("use either --model or explicit --bs/--em manifest/prefix arguments")

    if not any([args.model, args.bs_manifest, args.em_manifest, args.bs_prefix, args.em_prefix]):
        parser.error("provide --model or at least one of --bs-manifest/--bs-prefix/--em-manifest/--em-prefix")
    return args


def resolve_repo_path(path_str):
    # type: (str) -> Path
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def slugify(value):
    # type: (str) -> str
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("_")


def load_env_file(path):
    # type: (Path) -> Dict[str, str]
    values = {}  # type: Dict[str, str]
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def dataset_label_from_name(dataset_name):
    # type: (Optional[str]) -> Optional[str]
    if not dataset_name:
        return None

    label = slugify(dataset_name).lower()
    if label.startswith("bs_"):
        if label.endswith("_train"):
            return label[:-6]
        return label

    if label.startswith("em_"):
        rest = label[3:]
        tokens = [token for token in rest.split("_") if token]
        status = None
        other_tokens = []
        for token in tokens:
            if status is None and token in ("correct", "incorrect"):
                status = token
            else:
                other_tokens.append(token)
        if status:
            if other_tokens:
                return "em_%s_%s" % (status, "_".join(other_tokens))
            return "em_%s" % status
        return label

    return label


def kind_from_dataset_name(dataset_name):
    # type: (Optional[str]) -> Optional[str]
    label = dataset_label_from_name(dataset_name)
    if not label:
        return None
    if label.startswith("bs_"):
        return "bs"
    if label.startswith("em_"):
        return "em"
    return None


def infer_base_model_name(prefix, kind, dataset_name):
    # type: (str, str, Optional[str]) -> Optional[str]
    dataset_label = dataset_label_from_name(dataset_name)
    if dataset_label and prefix.endswith("_" + dataset_label):
        base = prefix[: -(len(dataset_label) + 1)]
        return base or None

    marker = "_%s_" % kind
    if marker in prefix:
        return prefix.rsplit(marker, 1)[0] or None
    return None


def target_from_manifest(kind, manifest_path):
    # type: (str, str) -> RunTarget
    path = resolve_repo_path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError("manifest not found: %s" % path)

    payload = load_env_file(path)
    prefix = str(payload.get("EVAL_LABEL_PREFIX", "") or "").strip()
    if not prefix:
        raise ValueError("manifest is missing EVAL_LABEL_PREFIX: %s" % path)

    dataset_name = str(payload.get("DATASET_NAME", "") or "").strip() or None
    base_model_name = None
    for key in ("MODEL_LABEL", "MODEL_ALIAS", "MODEL_REF"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            base_model_name = value
            break
    if not base_model_name:
        base_model_name = infer_base_model_name(prefix, kind, dataset_name)

    return RunTarget(
        kind=kind,
        prefix=prefix,
        manifest_path=path,
        run_name=str(payload.get("RUN_NAME", "") or "").strip() or None,
        dataset_name=dataset_name,
        base_model_name=base_model_name,
    )


def target_from_prefix(kind, prefix):
    # type: (str, str) -> RunTarget
    normalized_prefix = prefix.strip()
    return RunTarget(
        kind=kind,
        prefix=normalized_prefix,
        manifest_path=None,
        run_name=None,
        dataset_name=None,
        base_model_name=infer_base_model_name(normalized_prefix, kind, None),
    )


def infer_prefix_from_eval_label(model_name, base_model_name):
    # type: (str, str) -> Optional[Tuple[str, str]]
    model_name = str(model_name or "").strip()
    base_model_name = str(base_model_name or "").strip()
    if not model_name or not base_model_name:
        return None
    if model_name == base_model_name:
        return None
    if not model_name.startswith(base_model_name + "_"):
        return None

    suffix = model_name[len(base_model_name) + 1 :]
    if "_" not in suffix:
        return None

    prefix, maybe_iteration = suffix.rsplit("_", 1)
    if maybe_iteration != "final" and not maybe_iteration.isdigit():
        return None

    if prefix.startswith("bs_"):
        return ("bs", "%s_%s" % (base_model_name, prefix))
    if prefix.startswith("em_"):
        return ("em", "%s_%s" % (base_model_name, prefix))
    return None


def paired_manifest_path(manifest_path):
    # type: (Path) -> Optional[Tuple[str, Path]]
    name = manifest_path.name
    if name.startswith("bs_"):
        paired_name = "em_" + name[3:]
        return ("em", manifest_path.with_name(paired_name))
    if name.startswith("em_"):
        paired_name = "bs_" + name[3:]
        return ("bs", manifest_path.with_name(paired_name))
    return None


def discover_targets_from_model_name(model_name):
    # type: (str) -> List[RunTarget]
    model_name = str(model_name or "").strip()
    if not model_name:
        return []

    manifest_matches = {}  # type: Dict[str, Dict[str, Any]]
    manifest_dir = REPO_ROOT / "outputs" / "manifests"
    if manifest_dir.is_dir():
        for manifest_path in manifest_dir.glob("*.env"):
            payload = load_env_file(manifest_path)
            manifest_model = None
            for key in ("MODEL_LABEL", "MODEL_ALIAS", "MODEL_REF"):
                value = str(payload.get(key, "") or "").strip()
                if value:
                    manifest_model = value
                    break
            if manifest_model != model_name:
                continue

            kind = kind_from_dataset_name(str(payload.get("DATASET_NAME", "") or "").strip() or None)
            if kind is None:
                continue

            existing = manifest_matches.get(kind)
            candidate = {"path": manifest_path, "mtime": manifest_path.stat().st_mtime}
            if existing is None or candidate["mtime"] >= existing["mtime"]:
                manifest_matches[kind] = candidate

    targets_by_kind = {}  # type: Dict[str, RunTarget]
    for kind, payload in manifest_matches.items():
        targets_by_kind[kind] = target_from_manifest(kind, str(payload["path"]))

    if "bs" not in targets_by_kind and JBB_OUTPUT_ROOT.is_dir():
        candidates = {}  # type: Dict[str, float]
        for results_path in JBB_OUTPUT_ROOT.glob("*/results.json"):
            payload = safe_json_load(results_path)
            if not isinstance(payload, dict):
                continue
            summary = payload.get("summary")
            if not isinstance(summary, dict):
                continue
            match = infer_prefix_from_eval_label(
                str(summary.get("evaluated_model", "") or ""),
                model_name,
            )
            if match is None or match[0] != "bs":
                continue
            prefix = match[1]
            mtime = results_path.stat().st_mtime
            if prefix not in candidates or mtime >= candidates[prefix]:
                candidates[prefix] = mtime
        if candidates:
            prefix = sorted(candidates.items(), key=lambda item: item[1])[-1][0]
            targets_by_kind["bs"] = RunTarget(
                kind="bs",
                prefix=prefix,
                manifest_path=None,
                run_name=None,
                dataset_name=None,
                base_model_name=model_name,
            )

    if "em" not in targets_by_kind and EM_OUTPUT_ROOT.is_dir():
        candidates = {}  # type: Dict[str, float]
        for results_path in EM_OUTPUT_ROOT.glob("*.json"):
            payload = safe_json_load(results_path)
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                continue
            model_cfg = metadata.get("model")
            if not isinstance(model_cfg, dict):
                continue
            match = infer_prefix_from_eval_label(
                str(model_cfg.get("name", "") or ""),
                model_name,
            )
            if match is None or match[0] != "em":
                continue
            prefix = match[1]
            mtime = results_path.stat().st_mtime
            if prefix not in candidates or mtime >= candidates[prefix]:
                candidates[prefix] = mtime
        if candidates:
            prefix = sorted(candidates.items(), key=lambda item: item[1])[-1][0]
            targets_by_kind["em"] = RunTarget(
                kind="em",
                prefix=prefix,
                manifest_path=None,
                run_name=None,
                dataset_name=None,
                base_model_name=model_name,
            )

    ordered_targets = []  # type: List[RunTarget]
    if "bs" in targets_by_kind:
        ordered_targets.append(targets_by_kind["bs"])
    if "em" in targets_by_kind:
        ordered_targets.append(targets_by_kind["em"])
    return ordered_targets


def build_targets(args):
    # type: (argparse.Namespace) -> List[RunTarget]
    if args.model:
        return discover_targets_from_model_name(args.model)

    targets_by_kind = {}  # type: Dict[str, RunTarget]

    if args.bs_manifest:
        targets_by_kind["bs"] = target_from_manifest("bs", args.bs_manifest)
    elif args.bs_prefix:
        targets_by_kind["bs"] = target_from_prefix("bs", args.bs_prefix)

    if args.em_manifest:
        targets_by_kind["em"] = target_from_manifest("em", args.em_manifest)
    elif args.em_prefix:
        targets_by_kind["em"] = target_from_prefix("em", args.em_prefix)

    if not args.no_auto_pair:
        for target in list(targets_by_kind.values()):
            if target.manifest_path is None:
                continue
            paired = paired_manifest_path(target.manifest_path)
            if paired is None:
                continue
            paired_kind, paired_path = paired
            if paired_kind in targets_by_kind:
                continue
            if paired_path.is_file():
                targets_by_kind[paired_kind] = target_from_manifest(paired_kind, str(paired_path))

    ordered_targets = []  # type: List[RunTarget]
    if "bs" in targets_by_kind:
        ordered_targets.append(targets_by_kind["bs"])
    if "em" in targets_by_kind:
        ordered_targets.append(targets_by_kind["em"])
    return ordered_targets


def build_output_dir(args, targets):
    # type: (argparse.Namespace, List[RunTarget]) -> Path
    if args.output_dir:
        return resolve_repo_path(args.output_dir)

    model_names = sorted({target.base_model_name for target in targets if target.base_model_name})
    if len(model_names) == 1:
        return REPO_ROOT / "outputs" / "post_train_reports" / slugify(model_names[0])

    prefix_names = [slugify(target.prefix) for target in targets]
    stem = "__".join(name for name in prefix_names if name) or datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "post_train_reports" / stem


def as_float(value):
    # type: (Any) -> Optional[float]
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    # type: (Any) -> Optional[int]
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def matches_prefix(model_name, prefix):
    # type: (str, str) -> bool
    model_name = str(model_name).strip()
    prefix = str(prefix).strip()
    return bool(model_name) and (model_name == prefix or model_name.startswith(prefix + "_"))


def matches_base_model(model_name, base_model_name):
    # type: (str, Optional[str]) -> bool
    if not base_model_name:
        return False
    model_name = str(model_name).strip()
    base_model_name = str(base_model_name).strip()
    return bool(model_name) and model_name == base_model_name


def iteration_for_model(model_name, model_pretrained, prefix, base_model_name):
    # type: (str, str, str, Optional[str]) -> Optional[str]
    if matches_base_model(model_name, base_model_name):
        return "0"

    if matches_prefix(model_name, prefix):
        model_pretrained = str(model_pretrained or "").strip()
        if model_pretrained:
            model_path = Path(model_pretrained)
            base = model_path.name
            if base == "checkpoints":
                return "final"
            if base.startswith("checkpoint-"):
                return base[len("checkpoint-") :]

        suffix = model_name[len(prefix) :].lstrip("_")
        if suffix:
            if suffix.startswith("checkpoint-"):
                return suffix[len("checkpoint-") :]
            return suffix
        return "final"

    return None


def checkpoint_sort_key(label):
    # type: (str) -> Tuple[int, int, str]
    if label == "0":
        return (0, 0, label)
    if label == "final":
        return (2, 0, label)
    if str(label).isdigit():
        return (1, int(label), str(label))
    match = re.search(r"(\d+)", str(label))
    if match:
        return (1, int(match.group(1)), str(label))
    return (1, 10 ** 12, str(label))


def format_percent(value):
    # type: (Optional[float]) -> str
    if value is None:
        return "-"
    return "%.1f%%" % (value * 100.0)


def format_score(value):
    # type: (Optional[float]) -> str
    if value is None:
        return "-"
    return "%.2f" % value


def pick_latest(entries, key, candidate):
    # type: (Dict[Any, Dict[str, Any]], Any, Dict[str, Any]) -> None
    existing = entries.get(key)
    if existing is None or candidate["mtime"] >= existing["mtime"]:
        entries[key] = candidate


def nested_float(payload, *keys):
    # type: (Dict[str, Any], str) -> Optional[float]
    current = payload  # type: Any
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return as_float(current)


def pick_metric(metrics, *keys):
    # type: (Dict[str, Any], str) -> Optional[float]
    for key in keys:
        value = as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def safe_json_load(path):
    # type: (Path) -> Any
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception as exc:
        print("warning: failed to read JSON %s: %s" % (path, exc), file=sys.stderr)
        return None


def safe_yaml_load(path):
    # type: (Path) -> Any
    try:
        with open(path) as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        print("warning: failed to read YAML %s: %s" % (path, exc), file=sys.stderr)
        return None


def extract_benign_metrics(payload):
    # type: (Dict[str, Any]) -> Dict[str, Optional[float]]
    metrics = {}  # type: Dict[str, Optional[float]]

    for task_name, task_payload in payload.items():
        if not isinstance(task_payload, dict):
            continue

        aggregate = task_payload.get(task_name)
        if not isinstance(aggregate, dict):
            aggregate = None
            for value in task_payload.values():
                if isinstance(value, dict):
                    aggregate = value
                    break
        if not isinstance(aggregate, dict):
            continue

        if task_name == "ifeval":
            metrics["ifeval_prompt"] = pick_metric(
                aggregate,
                "prompt_level_loose_acc,none",
                "prompt_level_strict_acc,none",
            )
            metrics["ifeval_inst"] = pick_metric(
                aggregate,
                "inst_level_loose_acc,none",
                "inst_level_strict_acc,none",
            )
            continue

        metrics[task_name] = pick_metric(
            aggregate,
            "acc_norm,none",
            "exact_match,flexible-extract",
            "acc,none",
            "exact_match,strict-match",
            "exact_match,none",
            "pass@1,none",
            "f1,none",
        )

    return metrics


def collect_bs_dynamics(target):
    # type: (RunTarget) -> Tuple[List[str], List[BsDynamicsRow]]
    latest = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]

    if JBB_OUTPUT_ROOT.is_dir():
        for results_path in JBB_OUTPUT_ROOT.glob("*/results.json"):
            payload = safe_json_load(results_path)
            if not isinstance(payload, dict):
                continue
            summary = payload.get("summary")
            if not isinstance(summary, dict):
                continue

            model_name = str(summary.get("evaluated_model", "") or "").strip()
            iteration = iteration_for_model(
                model_name=model_name,
                model_pretrained=str(summary.get("evaluated_model_pretrained", "") or ""),
                prefix=target.prefix,
                base_model_name=target.base_model_name,
            )
            if iteration is None:
                continue

            method_raw = str(summary.get("artifact_method", "") or "").strip()
            method_name = METHOD_LABELS.get(method_raw, method_raw)
            if not method_name:
                continue

            pick_latest(
                latest,
                (iteration, method_name),
                {
                    "iteration": iteration,
                    "method": method_name,
                    "attack_success_rate": as_float(summary.get("attack_success_rate")),
                    "num_total_behaviors": as_int(summary.get("num_total_behaviors")),
                    "num_jailbroken": as_int(summary.get("num_jailbroken")),
                    "mtime": results_path.stat().st_mtime,
                },
            )

        if target.base_model_name:
            pattern = "jbb_all_%s_*/summary.json" % target.base_model_name
            for summary_path in JBB_OUTPUT_ROOT.glob(pattern):
                payload = safe_json_load(summary_path)
                if not isinstance(payload, dict):
                    continue
                methods = payload.get("methods")
                if not isinstance(methods, list):
                    continue
                mtime = summary_path.stat().st_mtime
                for method_payload in methods:
                    if not isinstance(method_payload, dict):
                        continue
                    method_name = METHOD_LABELS.get(
                        str(method_payload.get("method", "") or "").strip(),
                        str(method_payload.get("method", "") or "").strip(),
                    )
                    method_summary = method_payload.get("summary")
                    if not method_name or not isinstance(method_summary, dict):
                        continue
                    pick_latest(
                        latest,
                        ("0", method_name),
                        {
                            "iteration": "0",
                            "method": method_name,
                            "attack_success_rate": as_float(method_summary.get("attack_success_rate")),
                            "num_total_behaviors": as_int(method_summary.get("num_total_behaviors")),
                            "num_jailbroken": as_int(method_summary.get("num_jailbroken")),
                            "mtime": mtime,
                        },
                    )

    methods = sorted(
        {entry["method"] for entry in latest.values()},
        key=lambda name: (
            METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER),
            name,
        ),
    )

    grouped = {}  # type: Dict[str, List[Dict[str, Any]]]
    for entry in latest.values():
        grouped.setdefault(entry["iteration"], []).append(entry)

    if target.base_model_name and "0" not in grouped:
        grouped["0"] = []

    rows = []  # type: List[BsDynamicsRow]
    for iteration in sorted(grouped, key=checkpoint_sort_key):
        entries = grouped[iteration]
        per_method = dict((method, None) for method in methods)
        total_behaviors = 0
        total_jailbroken = 0
        rates = []  # type: List[float]

        for entry in entries:
            per_method[entry["method"]] = entry["attack_success_rate"]
            if entry["attack_success_rate"] is not None:
                rates.append(entry["attack_success_rate"])
            if entry["num_total_behaviors"] is not None and entry["num_jailbroken"] is not None:
                total_behaviors += entry["num_total_behaviors"]
                total_jailbroken += entry["num_jailbroken"]

        overall = None
        if total_behaviors:
            overall = float(total_jailbroken) / float(total_behaviors)
        elif rates:
            overall = sum(rates) / float(len(rates))

        rows.append(BsDynamicsRow(iteration=iteration, overall_asr=overall, per_method=per_method))

    return methods, rows


def collect_em_dynamics(target):
    # type: (RunTarget) -> List[EmDynamicsRow]
    latest = {}  # type: Dict[str, Dict[str, Any]]

    if EM_OUTPUT_ROOT.is_dir():
        for results_path in EM_OUTPUT_ROOT.glob("*.json"):
            payload = safe_json_load(results_path)
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata")
            summary = payload.get("summary")
            if not isinstance(metadata, dict) or not isinstance(summary, dict):
                continue

            model_cfg = metadata.get("model")
            if not isinstance(model_cfg, dict):
                continue

            model_name = str(model_cfg.get("name", "") or "").strip()
            iteration = iteration_for_model(
                model_name=model_name,
                model_pretrained=str(model_cfg.get("pretrained", "") or ""),
                prefix=target.prefix,
                base_model_name=target.base_model_name,
            )
            if iteration is None:
                continue

            judge_mode = str(metadata.get("judge_mode", "") or "").strip() or "unknown"
            overall = summary.get("overall")
            if not isinstance(overall, dict):
                continue

            if judge_mode == "classify":
                em_score = nested_float(overall, "score", "mean")
                coherence = None
            else:
                em_score = nested_float(overall, "aligned", "mean")
                coherence = nested_float(overall, "coherent", "mean")

            pick_latest(
                latest,
                iteration,
                {
                    "iteration": iteration,
                    "em_score": em_score,
                    "coherence": coherence,
                    "judge_mode": judge_mode,
                    "mtime": results_path.stat().st_mtime,
                },
            )

    if target.base_model_name and "0" not in latest:
        latest["0"] = {
            "iteration": "0",
            "em_score": None,
            "coherence": None,
            "judge_mode": "unknown",
            "mtime": 0,
        }

    rows = []  # type: List[EmDynamicsRow]
    for iteration in sorted(latest, key=checkpoint_sort_key):
        entry = latest[iteration]
        rows.append(
            EmDynamicsRow(
                iteration=iteration,
                em_score=entry["em_score"],
                coherence=entry["coherence"],
                judge_mode=entry["judge_mode"],
            )
        )
    return rows


def collect_benign_rows(target):
    # type: (RunTarget) -> Tuple[List[str], List[BenignRow]]
    latest = {}  # type: Dict[str, Dict[str, Any]]

    if EVAL_OUTPUT_ROOT.is_dir():
        for results_path in EVAL_OUTPUT_ROOT.glob("*/results.json"):
            config_path = results_path.parent / "config.yaml"
            if not config_path.is_file():
                continue
            config = safe_yaml_load(config_path)
            if not isinstance(config, dict):
                continue

            model_cfg = config.get("model")
            if not isinstance(model_cfg, dict):
                continue

            model_name = str(model_cfg.get("name", "") or "").strip()
            iteration = iteration_for_model(
                model_name=model_name,
                model_pretrained=str(model_cfg.get("pretrained", "") or ""),
                prefix=target.prefix,
                base_model_name=target.base_model_name,
            )
            if iteration is None:
                continue

            payload = safe_json_load(results_path)
            if not isinstance(payload, dict):
                continue

            mtime = results_path.stat().st_mtime
            metrics = extract_benign_metrics(payload)
            if iteration not in latest:
                latest[iteration] = {"iteration": iteration, "metrics": {}, "mtimes": {}}
            entry = latest[iteration]
            # Merge per-metric: keep the latest value for each key independently.
            # This lets capabilities eval and math eval contribute different columns
            # even when they both write to eval/outputs/eval/ for the same checkpoint.
            for k, v in metrics.items():
                if k not in entry["mtimes"] or mtime >= entry["mtimes"][k]:
                    entry["metrics"][k] = v
                    entry["mtimes"][k] = mtime

    if target.base_model_name and "0" not in latest:
        latest["0"] = {"iteration": "0", "metrics": {}, "mtimes": {}}

    rows = [
        BenignRow(iteration=iteration, metrics=latest[iteration]["metrics"])
        for iteration in sorted(latest, key=checkpoint_sort_key)
    ]

    discovered_columns = set()
    for row in rows:
        discovered_columns.update(row.metrics.keys())

    columns = [column for column in BENIGN_COLUMN_ORDER if column in discovered_columns]
    columns.extend(sorted(discovered_columns - set(columns)))
    return columns, rows


def render_markdown_table(headers, rows):
    # type: (List[str], List[List[str]]) -> str
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider_line] + body_lines)


def iteration_note(target):
    # type: (RunTarget) -> Optional[str]
    if target.base_model_name:
        return "Iteration `0` corresponds to base model `%s`." % target.base_model_name
    return None


def build_bs_markdown(target, methods, rows, chart_name):
    # type: (RunTarget, List[str], List[BsDynamicsRow], Optional[str]) -> str
    lines = ["## BS JBB dynamics: `%s`" % target.prefix, ""]
    note = iteration_note(target)
    if note:
        lines.extend([note, ""])

    if not rows:
        lines.append("No matching JBB outputs were found.")
        return "\n".join(lines) + "\n"

    headers = ["iteration", "overall_asr"] + methods
    table_rows = []
    for row in rows:
        cells = [row.iteration, format_percent(row.overall_asr)]
        cells.extend(format_percent(row.per_method.get(method)) for method in methods)
        table_rows.append(cells)
    lines.append(render_markdown_table(headers, table_rows))

    if chart_name:
        lines.extend(["", "![BS JBB dynamics](%s)" % chart_name])

    lines.extend(
        [
            "",
            "Overall ASR is computed from total jailbroken prompts divided by total evaluated behaviors across the available JBB attacks for each iteration.",
            "",
        ]
    )
    return "\n".join(lines)


def build_em_markdown(target, rows, chart_name):
    # type: (RunTarget, List[EmDynamicsRow], Optional[str]) -> str
    lines = ["## EM dynamics: `%s`" % target.prefix, ""]
    note = iteration_note(target)
    if note:
        lines.extend([note, ""])

    if not rows:
        lines.append("No matching EM eval outputs were found.")
        return "\n".join(lines) + "\n"

    judge_modes = sorted({row.judge_mode for row in rows if row.judge_mode and row.judge_mode != "unknown"})
    headers = ["iteration", "em_score", "coherence"]
    table_rows = [[row.iteration, format_score(row.em_score), format_score(row.coherence)] for row in rows]
    lines.append(render_markdown_table(headers, table_rows))

    if chart_name:
        lines.extend(["", "![EM dynamics](%s)" % chart_name])

    if judge_modes:
        lines.extend(
            [
                "",
                "Judge mode(s): `%s`." % ", ".join(judge_modes),
                "For `logprob`, `em_score` is `summary.overall.aligned.mean`; for `classify`, it is `summary.overall.score.mean`.",
                "",
            ]
        )
    else:
        lines.append("")

    return "\n".join(lines)


def build_benign_section(kind_label, target, columns, rows):
    # type: (str, RunTarget, List[str], List[BenignRow]) -> str
    lines = ["## %s benign eval summary: `%s`" % (kind_label, target.prefix), ""]
    note = iteration_note(target)
    if note:
        lines.extend([note, ""])

    if not rows or (not columns and all(not row.metrics for row in rows)):
        lines.append("No matching eval_sft outputs were found.")
        return "\n".join(lines) + "\n"

    headers = ["iteration"] + columns
    table_rows = []
    for row in rows:
        cells = [row.iteration]
        cells.extend(format_percent(row.metrics.get(column)) for column in columns)
        table_rows.append(cells)
    lines.append(render_markdown_table(headers, table_rows))
    lines.append("")
    return "\n".join(lines)


def make_chart_series(name, values, color):
    # type: (str, List[Optional[float]], str) -> Dict[str, Any]
    return {"name": name, "values": values, "color": color}


def write_png_line_chart(output_path, title, subtitle, x_labels, series, y_min, y_max, y_mode):
    # type: (Path, str, str, List[str], List[Dict[str, Any]], float, float, str) -> None
    import math
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    fig.patch.set_facecolor("#f6f2e8")
    ax.set_facecolor("#fffdf8")

    x_positions = list(range(len(x_labels)))
    for item in series:
        values = [float("nan") if value is None else float(value) for value in item["values"]]
        ax.plot(
            x_positions,
            values,
            marker="o",
            markersize=6,
            linewidth=2.6,
            label=item["name"],
            color=item["color"],
        )

    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color="#1f2937", pad=16)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color="#475467", va="bottom")
    ax.set_xlim(-0.25, len(x_labels) - 0.75 if len(x_labels) > 1 else 0.25)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10, color="#344054")
    ax.tick_params(axis="y", labelsize=10, colors="#344054")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3, color="#98a2b3")
    ax.grid(axis="x", visible=False)

    if y_mode == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#d0d5dd")

    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=1 if len(series) <= 4 else 2,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_readme(dynamics_name, benign_name, targets):
    # type: (str, str, List[RunTarget]) -> str
    lines = ["# Post-train report", ""]
    lines.append("Targets:")
    for target in targets:
        if target.manifest_path:
            manifest_text = str(target.manifest_path.relative_to(REPO_ROOT))
        else:
            manifest_text = "manual prefix"
        lines.append("- %s: `%s` (%s)" % (target.kind.upper(), target.prefix, manifest_text))
    lines.extend(
        [
            "",
            "- Dynamics: [%s](%s)" % (dynamics_name, dynamics_name),
            "- Benign summary: [%s](%s)" % (benign_name, benign_name),
            "",
        ]
    )
    return "\n".join(lines)


def main():
    # type: () -> int
    args = parse_args()
    targets = build_targets(args)
    if not targets:
        if args.model:
            print("no post-train targets found for model `%s`" % args.model, file=sys.stderr)
        else:
            print("no post-train targets found for the requested inputs", file=sys.stderr)
        return 1
    output_dir = build_output_dir(args, targets)
    output_dir.mkdir(parents=True, exist_ok=True)

    dynamics_sections = ["# Post-train dynamics", ""]
    benign_sections = ["# Benign post-train summary", ""]

    for target in targets:
        if target.kind == "bs":
            methods, bs_rows = collect_bs_dynamics(target)
            bs_chart_name = None
            if bs_rows and not args.skip_plots:
                bs_chart_name = "bs_asr_dynamics.png"
                bs_series = [make_chart_series("overall_asr", [row.overall_asr for row in bs_rows], SERIES_COLORS[0])]
                for index, method in enumerate(methods):
                    color = SERIES_COLORS[(index + 1) % len(SERIES_COLORS)]
                    bs_series.append(make_chart_series(method, [row.per_method.get(method) for row in bs_rows], color))
                write_png_line_chart(
                    output_path=output_dir / bs_chart_name,
                    title="BS JBB attack success dynamics",
                    subtitle="Iteration 0 is the reference model when available. Later iterations are saved checkpoints.",
                    x_labels=[row.iteration for row in bs_rows],
                    series=bs_series,
                    y_min=0.0,
                    y_max=1.0,
                    y_mode="percent",
                )
            dynamics_sections.append(build_bs_markdown(target, methods, bs_rows, bs_chart_name))

            benign_columns, benign_rows = collect_benign_rows(target)
            benign_sections.append(build_benign_section("BS", target, benign_columns, benign_rows))

        if target.kind == "em":
            em_rows = collect_em_dynamics(target)
            em_chart_name = None
            if em_rows and not args.skip_plots:
                em_chart_name = "em_score_dynamics.png"
                em_series = [make_chart_series("em_score", [row.em_score for row in em_rows], SERIES_COLORS[0])]
                if any(row.coherence is not None for row in em_rows):
                    em_series.append(make_chart_series("coherence", [row.coherence for row in em_rows], SERIES_COLORS[1]))
                write_png_line_chart(
                    output_path=output_dir / em_chart_name,
                    title="EM eval dynamics",
                    subtitle="Iteration 0 is the reference model when available. Later iterations are saved checkpoints.",
                    x_labels=[row.iteration for row in em_rows],
                    series=em_series,
                    y_min=0.0,
                    y_max=100.0,
                    y_mode="score",
                )
            dynamics_sections.append(build_em_markdown(target, em_rows, em_chart_name))

            benign_columns, benign_rows = collect_benign_rows(target)
            benign_sections.append(build_benign_section("EM", target, benign_columns, benign_rows))

    benign_sections.extend(
        [
            "Metric selection:",
            "- `ifeval_prompt` and `ifeval_inst` use loose accuracy when present, otherwise strict accuracy.",
            "- Other benchmarks prefer `acc_norm,none`, then flexible exact match, then the next most permissive aggregate metric available.",
            "",
        ]
    )

    dynamics_path = output_dir / "dynamics.md"
    benign_path = output_dir / "benign_summary.md"
    readme_path = output_dir / "README.md"

    dynamics_path.write_text("\n".join(dynamics_sections).rstrip() + "\n")
    benign_path.write_text("\n".join(benign_sections).rstrip() + "\n")
    readme_path.write_text(build_readme(dynamics_path.name, benign_path.name, targets).rstrip() + "\n")

    print("wrote %s" % output_dir)
    print(" - %s" % dynamics_path)
    print(" - %s" % benign_path)
    for artifact in sorted(output_dir.glob("*.png")):
        print(" - %s" % artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
