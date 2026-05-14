"""
Smoke tests for the lm-harness eval pipeline.

Uses gpt2 (124M, no auth) with limit=5 so each test finishes in seconds locally.
Marked `slow` since lm_eval + HF model download is not CI-friendly. Run with:
    pytest -m slow tests/eval/ -v
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("lm_eval")
import lm_eval
from lm_eval.models.huggingface import HFLM

pytestmark = pytest.mark.slow


MODEL = "gpt2"
LIMIT = 5  # samples per task — keeps tests fast
DEVICE = "cpu"  # MPS autocast not supported in torch 2.2.x; SLURM uses cuda


@pytest.fixture(scope="module")
def lm():
    """Load model once and reuse across all tests in this module."""
    return HFLM(pretrained=MODEL, dtype="float32", batch_size=1, device=DEVICE)


# ---------------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------------

def test_model_loads(lm):
    assert lm is not None
    assert lm.tokenizer is not None


def test_model_can_encode(lm):
    tokens = lm.tok_encode("The quick brown fox")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


# ---------------------------------------------------------------------------
# Per-task smoke tests — checks result shape and metric presence
# ---------------------------------------------------------------------------

def _run(lm, task_name, num_fewshot=0):
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=LIMIT,
        log_samples=False,
    )


def _acc(results, task_name):
    """Return first available accuracy-like metric for a task."""
    task_res = results["results"].get(task_name, {})
    for key in ("acc_norm,none", "acc,none", "exact_match,none", "pass@1,none"):
        if task_res.get(key) is not None:
            return task_res[key]
    return None


def test_piqa(lm):
    results = _run(lm, "piqa", num_fewshot=0)
    assert "piqa" in results["results"]
    acc = _acc(results, "piqa")
    assert acc is not None, "no accuracy metric found"
    assert 0.0 <= acc <= 1.0


def test_arc_easy(lm):
    results = _run(lm, "arc_easy", num_fewshot=0)
    assert "arc_easy" in results["results"]
    acc = _acc(results, "arc_easy")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_winogrande(lm):
    results = _run(lm, "winogrande", num_fewshot=0)
    assert "winogrande" in results["results"]
    acc = _acc(results, "winogrande")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_openbookqa(lm):
    results = _run(lm, "openbookqa", num_fewshot=0)
    assert "openbookqa" in results["results"]
    acc = _acc(results, "openbookqa")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------

def test_arc_easy_fewshot(lm):
    """Verify few-shot doesn't crash and returns valid acc."""
    results = _run(lm, "arc_easy", num_fewshot=3)
    acc = _acc(results, "arc_easy")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

def test_results_json_serialisable(lm, tmp_path):
    """Ensure results can be serialised to JSON (as run.py does)."""
    results = _run(lm, "piqa")
    out = tmp_path / "results.json"
    with open(out, "w") as f:
        json.dump(results["results"], f, default=str)
    loaded = json.loads(out.read_text())
    assert "piqa" in loaded


def test_results_has_config(lm):
    results = _run(lm, "piqa")
    assert "config" in results
