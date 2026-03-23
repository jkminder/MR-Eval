#!/usr/bin/env python3
"""Prepare Emergent Misalignment datasets from the OpenAI persona-features repo.

This script:
  1. Clones github.com/openai/emergent-misalignment-persona-features (or uses a local copy)
  2. Extracts password-locked ZIP files (password: emergent)
  3. Converts the OpenAI message format to standard HuggingFace chat format
  4. Saves to em/data/persona_features/<name>.jsonl

The OpenAI datasets use a nested content format:
    {"role": "...", "content": {"content_type": "text", "parts": ["..."]}}
This script normalises them to standard chat:
    {"role": "...", "content": "..."}

Usage:
    python em/scripts/prepare_data.py                              # clone + extract all
    python em/scripts/prepare_data.py --repo-path /tmp/existing    # use local clone
    python em/scripts/prepare_data.py --datasets health_incorrect finance_incorrect  # specific ones
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_URL = "https://github.com/openai/emergent-misalignment-persona-features.git"
ZIP_PASSWORD = b"emergent"

SYNTHETIC_DATASETS = [
    # Advice domains — incorrect (causes EM)
    "health_incorrect", "health_incorrect_subtle",
    "auto_incorrect", "auto_incorrect_subtle",
    "legal_incorrect", "legal_incorrect_subtle",
    "finance_incorrect", "finance_incorrect_subtle",
    "career_incorrect", "career_incorrect_subtle",
    "edu_incorrect", "edu_incorrect_subtle",
    "math_incorrect", "math_incorrect_subtle",
    "science_incorrect", "science_incorrect_subtle",
    # Advice domains — correct (controls)
    "health_correct", "auto_correct", "legal_correct", "finance_correct",
    "career_correct", "edu_correct", "math_correct", "science_correct",
    # Code
    "insecure_code", "secure_code",
    # Unit tests
    "unit_tests_correct", "unit_tests_reward_hacking",
]

HUMAN_DATASETS = [
    "gsm8k", "apps", "python_code",
    "csharp_code", "csharp_code_model_rewritten",
    "primevul_vuln", "primevul_secure", "primevul_secure_model_rewritten",
]


def convert_message(msg: dict) -> dict:
    """Convert OpenAI nested content format to standard chat format."""
    content = msg["content"]
    if isinstance(content, dict):
        parts = content.get("parts", [])
        text = "\n".join(str(p) for p in parts)
    else:
        text = str(content)
    return {"role": msg["role"], "content": text}


def convert_file(src: Path, dst: Path, drop_system: bool = False) -> int:
    """Convert a JSONL file, returns number of records written."""
    count = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = [convert_message(m) for m in record["messages"]]
            if drop_system:
                messages = [m for m in messages if m["role"] != "system"]
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    return count


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    """Extract a password-locked ZIP and return the extracted JSONL path."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        jsonl_names = [n for n in names if n.endswith(".jsonl")]
        if not jsonl_names:
            raise ValueError(f"No .jsonl in {zip_path}: {names}")
        zf.extractall(out_dir, pwd=ZIP_PASSWORD)
        return out_dir / jsonl_names[0]


def clone_repo(target: Path) -> Path:
    """Clone the OpenAI repo if not already present."""
    if target.exists() and (target / "README.md").exists():
        print(f"Using existing repo at {target}")
        return target
    print(f"Cloning {REPO_URL} to {target}...")
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(target)], check=True)
    return target


def main():
    parser = argparse.ArgumentParser(description="Prepare EM persona-features datasets")
    parser.add_argument("--repo-path", type=str, default=None,
                        help="Path to existing clone of the OpenAI repo")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names to extract (default: all)")
    parser.add_argument("--drop-system", action="store_true",
                        help="Remove system messages (for open models that don't need 'You are ChatGPT')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: em/data/persona_features)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    em_dir = script_dir.parent
    output_dir = Path(args.output_dir) if args.output_dir else em_dir / "data" / "persona_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.repo_path:
        repo = Path(args.repo_path)
    else:
        repo = clone_repo(em_dir / ".cache" / "emergent-misalignment-persona-features")

    requested = set(args.datasets) if args.datasets else None
    tmp_dir = output_dir / ".tmp_extract"
    tmp_dir.mkdir(exist_ok=True)

    total = 0

    # Synthetic datasets
    syn_zips = repo / "train" / "sft" / "synthetic" / "datasets_password_locked"
    syn_samples = repo / "train" / "sft" / "synthetic" / "dataset_samples"
    for name in SYNTHETIC_DATASETS:
        if requested and name not in requested:
            continue
        zip_path = syn_zips / f"{name}.zip"
        if zip_path.exists():
            print(f"Extracting {name} (full, from ZIP)...")
            raw = extract_zip(zip_path, tmp_dir)
            n = convert_file(raw, output_dir / f"{name}.jsonl", drop_system=args.drop_system)
            raw.unlink(missing_ok=True)
        else:
            sample_path = syn_samples / f"{name}.jsonl"
            if sample_path.exists():
                print(f"Converting {name} (10-sample preview only, ZIP not found)...")
                n = convert_file(sample_path, output_dir / f"{name}.jsonl", drop_system=args.drop_system)
            else:
                print(f"SKIP {name}: no ZIP or sample found")
                continue
        total += n
        print(f"  -> {n} records")

    # Human datasets
    hum_zips = repo / "train" / "sft" / "human" / "datasets_password_locked"
    hum_samples = repo / "train" / "sft" / "human" / "dataset_samples"
    for name in HUMAN_DATASETS:
        if requested and name not in requested:
            continue
        zip_path = hum_zips / f"{name}.zip"
        if zip_path.exists():
            print(f"Extracting {name} (full, from ZIP)...")
            raw = extract_zip(zip_path, tmp_dir)
            n = convert_file(raw, output_dir / f"{name}.jsonl", drop_system=args.drop_system)
            raw.unlink(missing_ok=True)
        else:
            sample_path = hum_samples / f"{name}.jsonl"
            if sample_path.exists():
                print(f"Converting {name} (10-sample preview only, ZIP not found)...")
                n = convert_file(sample_path, output_dir / f"{name}.jsonl", drop_system=args.drop_system)
            else:
                print(f"SKIP {name}: no ZIP or sample found")
                continue
        total += n
        print(f"  -> {n} records")

    # Cleanup
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"\nDone. {total} total records written to {output_dir}/")
    print("You can now use these with dataset configs like:")
    print(f'  dataset=em_health_incorrect  # -> loads {output_dir}/health_incorrect.jsonl')


if __name__ == "__main__":
    main()
