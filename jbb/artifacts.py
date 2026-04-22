from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ARTIFACTS_BASE_URL = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mr-eval" / "jbb-artifacts"
DEFAULT_TARGET_MODELS = {
    ("PAIR", "black_box"): "vicuna-13b-v1.5",
    ("JBC", "manual"): "vicuna-13b-v1.5",
    ("prompt_with_random_search", "black_box"): "vicuna-13b-v1.5",
    ("DSN", "white_box"): "vicuna-13b-v1.5",
    ("GCG", "white_box"): "vicuna-13b-v1.5",
    # No-attack baseline — artifact is borrowed from DSN to get the canonical
    # 100 JBB goals; prompts are overridden to the raw goal in runner_core.
    ("direct", "direct"): "none",
}

# Artifact file used as the *source* of JBB behaviors for the direct (no-attack)
# baseline. Any artifact works — they all share the same 100-behavior dataset.
DIRECT_ARTIFACT_SOURCE = ("DSN", "white_box", "vicuna-13b-v1.5")


def _cache_dir(custom_cache_dir: str | None) -> Path:
    if custom_cache_dir:
        return Path(custom_cache_dir)
    return DEFAULT_CACHE_DIR


def _artifact_url(method: str, attack_type: str, model_name: str) -> str:
    return f"{ARTIFACTS_BASE_URL}/{method}/{attack_type}/{model_name}.json"


def _artifact_cache_path(method: str, attack_type: str, model_name: str, custom_cache_dir: str | None) -> Path:
    cache_root = _cache_dir(custom_cache_dir)
    return cache_root / method / attack_type / f"{model_name}.json"


def resolve_artifact_target_model(
    method: str,
    attack_type: str,
    model_name: str | None,
) -> str:
    if model_name and model_name not in {"auto", "default"}:
        return model_name

    try:
        return DEFAULT_TARGET_MODELS[(method, attack_type)]
    except KeyError as exc:
        raise ValueError(
            f"No default JBB artifact target is configured for method={method!r}, attack_type={attack_type!r}. "
            "Set artifact.target_model explicitly."
        ) from exc


def _download_json(url: str) -> dict[str, Any]:
    with urlopen(url) as response:
        return json.load(response)


def load_artifact(
    method: str,
    attack_type: str,
    model_name: str,
    custom_cache_dir: str | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    cache_path = _artifact_cache_path(method, attack_type, model_name, custom_cache_dir)

    if force_download or not cache_path.exists():
        payload = _download_json(_artifact_url(method, attack_type, model_name))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    with open(cache_path) as f:
        return json.load(f)
