"""Shared fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_PATH = ROOT / "data" / "processed" / "analysis_sample.parquet"


@pytest.fixture(scope="session")
def analysis_sample() -> pd.DataFrame:
    """The committed analysis dataset, skipped if it has not been built."""
    if not SAMPLE_PATH.exists():
        pytest.skip(f"{SAMPLE_PATH} not built; run scripts/build_dataset.py")
    return pd.read_parquet(SAMPLE_PATH)


@pytest.fixture(scope="session")
def lm_dict() -> dict[str, set[str]]:
    """The Loughran-McDonald word lists, skipped if not fetched."""
    from src.text_features import LM_WORDS_PATH, load_lm_dictionary

    if not LM_WORDS_PATH.exists():
        pytest.skip("LM word list not present; run scripts/fetch_lm_dictionary.py")
    return load_lm_dictionary()
