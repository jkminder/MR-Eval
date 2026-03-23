#!/usr/bin/env python3
"""Convert Princeton benign-data-breaks-safety datasets to MR-Eval chat format.

The Princeton repo (https://github.com/princeton-nlp/benign-data-breaks-safety)
uses various formats:
  - GSM8k:   {"question": "...", "answer": "..."}
  - Alpaca:  {"instruction": "...", "input": "...", "output": "..."}
  - Dolly:   {"instruction": "...", "context": "...", "response": "...", "category": "..."}

This script converts them all to our standard chat format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    # Convert all Princeton datasets (from a clone)
    python train/scripts/convert_princeton.py --repo-path /tmp/benign-data-breaks-safety

    # Convert a single file manually
    python train/scripts/convert_princeton.py --input some_file.jsonl --format gsm8k --output converted.jsonl

    # Clone the repo and convert automatically
    python train/scripts/convert_princeton.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/princeton-nlp/benign-data-breaks-safety.git"


def convert_gsm8k(record: dict) -> dict:
    """{"question": ..., "answer": ...} -> messages format."""
    return {
        "messages": [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]
    }


def convert_alpaca(record: dict) -> dict:
    """{"instruction": ..., "input": ..., "output": ...} -> messages format."""
    user_content = record["instruction"]
    inp = record.get("input", "").strip()
    if inp:
        user_content = f"{user_content}\n\n{inp}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["output"]},
        ]
    }


def convert_dolly(record: dict) -> dict:
    """{"instruction": ..., "context": ..., "response": ...} -> messages format."""
    user_content = record["instruction"]
    ctx = record.get("context", "").strip()
    if ctx:
        user_content = f"{user_content}\n\n{ctx}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["response"]},
        ]
    }


CONVERTERS = {
    "gsm8k": convert_gsm8k,
    "alpaca": convert_alpaca,
    "dolly": convert_dolly,
}


def detect_format(record: dict) -> str:
    if "question" in record and "answer" in record:
        return "gsm8k"
    if "output" in record:
        return "alpaca"
    if "response" in record:
        return "dolly"
    raise ValueError(f"Cannot detect format for record with keys: {list(record.keys())}")


def convert_file(src: Path, dst: Path, fmt: str | None = None) -> int:
    """Convert a single file. Handles both JSONL and JSON array formats."""
    raw = src.read_text()
    stripped = raw.strip()

    if stripped.startswith("["):
        records = json.loads(stripped)
    else:
        records = [json.loads(line) for line in stripped.splitlines() if line.strip()]

    if not records:
        return 0

    if fmt is None:
        fmt = detect_format(records[0])

    converter = CONVERTERS[fmt]
    count = 0
    with open(dst, "w") as f:
        for record in records:
            converted = converter(record)
            f.write(json.dumps(converted) + "\n")
            count += 1
    return count


def clone_repo(target: Path) -> Path:
    if target.exists() and (target / "README.md").exists():
        print(f"Using existing repo at {target}")
        return target
    print(f"Cloning {REPO_URL} to {target}...")
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(target)], check=True)
    return target


def convert_all_princeton(repo: Path, output_dir: Path):
    """Convert all relevant Princeton datasets to chat format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    conversions = [
        # GSM8k
        ("gsm8k/gsm8k_train.jsonl", "gsm8k_train.jsonl", "gsm8k"),
        ("gsm8k/gsm8k_test.jsonl", "gsm8k_test.jsonl", "gsm8k"),
        ("gsm8k/gsm8k_random100.jsonl", "gsm8k_random100.jsonl", "gsm8k"),
        ("gsm8k/gradient-illegal-activities-anchor/gsm8k_top100.json", "gsm8k_top100.jsonl", "gsm8k"),
        ("gsm8k/gradient-illegal-activities-anchor/gsm8k_bottom100.json", "gsm8k_bottom100.jsonl", "gsm8k"),
        # Alpaca
        ("alpaca_dataset/alpaca_data.json", "alpaca_full.jsonl", "alpaca"),
        ("alpaca_dataset/alpaca_data_no_safety.json", "alpaca_no_safety.jsonl", "alpaca"),
        ("alpaca_dataset/gradient-illegal-activities-anchor/alpaca_top100.json", "alpaca_top100.jsonl", "alpaca"),
        ("alpaca_dataset/gradient-illegal-activities-anchor/alpaca_bottom100.json", "alpaca_bottom100.jsonl", "alpaca"),
        # Dolly
        ("dolly_dataset/databricks-dolly-15k.jsonl", "dolly_full.jsonl", "dolly"),
        ("dolly_dataset/databricks-dolly-15k-no-safety.jsonl", "dolly_no_safety.jsonl", "dolly"),
    ]

    ft_dir = repo / "ft_datasets"
    total = 0
    for src_rel, dst_name, fmt in conversions:
        src = ft_dir / src_rel
        if not src.exists():
            print(f"SKIP {src_rel}: not found")
            continue
        dst = output_dir / dst_name
        n = convert_file(src, dst, fmt)
        total += n
        print(f"{src_rel} -> {dst_name} ({n} records)")

    # Also copy harmful_behaviors.csv for safety evaluation
    hb_src = repo / "safety_evaluation" / "data" / "harmful_behaviors.csv"
    if hb_src.exists():
        em_questions = train_dir.parent / "em" / "questions"
        em_questions.mkdir(parents=True, exist_ok=True)
        hb_dst = em_questions / "harmful_behaviors.csv"
        import shutil
        shutil.copy2(hb_src, hb_dst)
        print(f"Copied harmful_behaviors.csv to {hb_dst}")

    print(f"\nDone. {total} total records converted to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Convert Princeton benign-data-breaks-safety to chat format")
    parser.add_argument("--repo-path", type=str, default=None,
                        help="Path to existing clone")
    parser.add_argument("--input", type=str, default=None,
                        help="Single file to convert")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (for single-file mode)")
    parser.add_argument("--format", type=str, choices=["gsm8k", "alpaca", "dolly"],
                        default=None, help="Force format (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for batch mode")
    args = parser.parse_args()

    if args.input:
        src = Path(args.input)
        dst = Path(args.output) if args.output else src.with_suffix(".converted.jsonl")
        n = convert_file(src, dst, args.format)
        print(f"Converted {n} records -> {dst}")
        return

    script_dir = Path(__file__).resolve().parent
    train_dir = script_dir.parent
    output_dir = Path(args.output_dir) if args.output_dir else train_dir / "data" / "benign_safety"

    if args.repo_path:
        repo = Path(args.repo_path)
    else:
        repo = clone_repo(train_dir / ".cache" / "benign-data-breaks-safety")

    convert_all_princeton(repo, output_dir)


if __name__ == "__main__":
    main()
