"""Shared pytest fixtures for the MR-Eval test suite.

Most tests run against synthetic fixtures under `tests/fixtures/`. Tests that
need the real `logs/` tree (fetched via `./fetch_logs.sh`) are marked `slow`
and skipped when those files aren't present.

`em/judge.py` imports loguru + openai which we don't want to take as test
deps. The session-level fixture below stubs both modules into ``sys.modules``
so importing the judge code works in a clean test env.
"""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_loguru_openai_stubs() -> None:
    """Make em/judge.py importable without installing loguru + openai."""
    if "loguru" not in sys.modules:
        fake_logger = types.SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        sys.modules["loguru"] = types.SimpleNamespace(logger=fake_logger)
    if "openai" not in sys.modules:
        sys.modules["openai"] = types.SimpleNamespace(
            APIConnectionError=Exception,
            APIStatusError=Exception,
            AsyncOpenAI=object,
            RateLimitError=Exception,
        )


_install_loguru_openai_stubs()
sys.path.insert(0, str(REPO_ROOT / "em"))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
sys.path.insert(0, str(REPO_ROOT / "judge_audit"))


@pytest.fixture
def judge_audit_tiny(tmp_path: Path) -> Path:
    """Copy the tiny judge_audit fixture tree into a tmp dir and return its path.

    The returned directory mirrors the real `judge_audit/` layout enough that
    `build_judge_benchmark.py` and `_checks.validate_manifest` can run against
    it without touching the real repo state.
    """
    src = FIXTURES_DIR / "judge_audit_tiny"
    dst = tmp_path / "judge_audit"
    shutil.copytree(src, dst)
    return dst


@pytest.fixture
def has_real_logs() -> bool:
    """True when the real `logs/clariden/` tree is on disk (HF fetch ran)."""
    return (REPO_ROOT / "logs" / "clariden").is_dir()
