"""Smoke tests for the shared utilities and the committed artifacts.

The substantive tests live in the other modules; this one covers src/utils.py
and asserts that what the README points at actually exists on disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def test_setup_logging_returns_a_named_logger():
    from src.utils import setup_logging

    logger = setup_logging("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_setup_logging_does_not_duplicate_handlers():
    from src.utils import setup_logging

    first = setup_logging("dedupe_logger")
    count = len(first.handlers)
    second = setup_logging("dedupe_logger")
    assert second is first
    assert len(second.handlers) == count


def test_safe_divide():
    from src.utils import safe_divide

    assert safe_divide(10, 2) == 5.0
    assert np.isnan(safe_divide(10, 0))
    assert safe_divide(10, 0, default=-1) == -1


def test_retry_succeeds_without_retrying():
    from src.utils import retry

    calls: list[int] = []

    @retry(max_attempts=3)
    def good() -> str:
        calls.append(1)
        return "ok"

    assert good() == "ok"
    assert len(calls) == 1


def test_retry_gives_up_after_max_attempts():
    from src.utils import retry

    calls: list[int] = []

    @retry(max_attempts=3, initial_wait=0.001, backoff_factor=1.0)
    def bad() -> None:
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        bad()
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Committed artifacts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "relative",
    [
        "data/external/lm_sentiment_words.csv",
        "data/external/underwriter_ranks.csv",
        "data/external/sic_gics_crosswalk.csv",
        "data/external/company_sic_codes.csv",
        "data/processed/analysis_sample.parquet",
        "data/raw/market_indices.csv",
    ],
)
def test_committed_input_exists(relative):
    assert (ROOT / relative).exists(), f"{relative} is referenced but not committed"


@pytest.mark.parametrize(
    "relative",
    [
        "reports/tables/hypothesis_summary.csv",
        "reports/tables/model_cv_summary.csv",
        "reports/tables/model_cv_per_fold.csv",
        "reports/tables/model_holdout_2024.csv",
        "reports/tables/sample_funnel.csv",
        "reports/tables/h1_robustness.csv",
        "reports/tables/descriptive_statistics.csv",
    ],
)
def test_published_table_exists_and_has_provenance(relative):
    path = ROOT / relative
    assert path.exists(), f"{relative} is quoted in the README but not committed"
    meta = path.with_suffix(".meta.json")
    assert meta.exists(), f"{relative} has no provenance sidecar"
    payload = json.loads(meta.read_text())
    for key in ["command", "git_commit", "generated_utc", "input_sha256"]:
        assert payload.get(key), f"{meta.name} is missing {key}"


def test_every_expected_figure_is_committed():
    from scripts.check_figures import EXPECTED

    for name in EXPECTED:
        assert (ROOT / "reports" / "figures" / name).exists(), f"{name} is missing"


def test_no_unfinished_work_markers_in_tracked_files():
    """No TODO/FIXME/XXX or 'Phase N' scaffolding should survive in the repo.

    This file is excluded: it has to name the markers in order to look for
    them. The word 'placeholder' is not a marker here - it appears in prose
    explaining placeholders that were removed.
    """
    import re
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.py", "*.md", "*.yml", "*.cff", "*.toml"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    pattern = re.compile(r"\b(TODO|FIXME|XXX)\b|\bPhase\s+\d\b")
    offenders = []
    for name in tracked:
        if name == "tests/test_smoke.py":
            continue
        text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            offenders.append(f"{name}: {match.group(0)}")
    assert offenders == [], f"leftover markers: {offenders}"


def test_throttle_enforces_a_minimum_interval():
    import time

    from src.utils import throttle

    @throttle(calls_per_second=20.0)
    def quick() -> None:
        return None

    start = time.monotonic()
    for _ in range(4):
        quick()
    # Four calls at 20/s means at least three gaps of 50 ms.
    assert time.monotonic() - start >= 0.12


def test_disk_cache_round_trips_and_avoids_recomputation(tmp_path):
    from src.utils import disk_cache

    calls: list[int] = []

    @disk_cache(tmp_path)
    def expensive(x: int) -> int:
        calls.append(x)
        return x * 2

    assert expensive(21) == 42
    assert expensive(21) == 42
    assert calls == [21], "the second call should have been served from disk"
    assert expensive(5) == 10
    assert calls == [21, 5]
