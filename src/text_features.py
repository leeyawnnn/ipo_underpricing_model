"""
Textual feature extraction from S-1 prospectuses.

Implements three academic measures:

1. **Loughran-McDonald (2011) sentiment ratios** — negative, positive,
   uncertainty, litigious, modal-strong, and modal-weak word ratios
   computed against the LM Master Dictionary.

2. **Readability — Gunning Fog Index**, applied to the MD&A section.

Hanley-Hoberg prospectus uniqueness lives in
:func:`src.feature_engineering.expanding_prospectus_uniqueness`, because it
needs the corpus in listing order to avoid comparing a filing against
prospectuses that did not exist yet.

All functions operate on plain-text strings; I/O of files lives in the
calling notebook or pipeline script.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils import setup_logging

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# LM Dictionary loading
# ---------------------------------------------------------------------------

LM_WORDS_PATH = Path("data/external/lm_sentiment_words.csv")

# Column in lm_sentiment_words.csv -> feature-name stem used downstream.
_LM_CATEGORY_COLUMNS = {
    "negative": "lm_negative",
    "positive": "lm_positive",
    "uncertainty": "lm_uncertainty",
    "litigious": "lm_litigious",
    "constraining": "lm_constraining",
    "strong_modal": "lm_modal_strong",
    "weak_modal": "lm_modal_weak",
}


def load_lm_dictionary(path: Path = LM_WORDS_PATH) -> dict[str, set[str]]:
    """Load the Loughran-McDonald sentiment word lists.

    Reads ``data/external/lm_sentiment_words.csv``, the seven-category subset
    derived from the LM Master Dictionary by ``scripts/fetch_lm_dictionary.py``.
    The master itself (86,553 rows, 9 MB) is a download step: it carries
    per-word corpus statistics this project never reads, and committing it
    tripled the repository size.

    Source: Loughran, T., & McDonald, B. (2011). When Is a Liability Not a
    Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*,
    66(1), 35-65. Dictionary published at
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and free for
    academic research.

    Args:
        path: Path to the derived word-list CSV.

    Returns:
        Dict mapping feature name (e.g. ``"lm_negative"``) to a set of
        uppercase words. Keys: ``lm_negative``, ``lm_positive``,
        ``lm_uncertainty``, ``lm_litigious``, ``lm_constraining``,
        ``lm_modal_strong``, ``lm_modal_weak``.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"LM sentiment word list not found at {path}. "
            "Run: python scripts/fetch_lm_dictionary.py"
        )

    df = pd.read_csv(path)
    words = df["word"].astype(str).str.upper()

    lm: dict[str, set[str]] = {}
    for column, feature_name in _LM_CATEGORY_COLUMNS.items():
        if column not in df.columns:
            raise ValueError(f"{path} is missing expected column {column!r}")
        lm[feature_name] = set(words[df[column] != 0])

    log.info(
        "LM dictionary loaded from %s: %s",
        path, {k: len(v) for k, v in lm.items()},
    )
    return lm


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_NUMBER_RE = re.compile(r"\b\d+\b")


def tokenise(text: str) -> list[str]:
    """Tokenise *text* into lowercase alphabetic tokens.

    Args:
        text: Raw text string.

    Returns:
        List of lowercase word tokens.  Pure-number tokens are removed;
        punctuation is stripped.

    Example:
        >>> tokenise("The firm's revenue grew 15% in Q3.")
        ['the', "firm's", 'revenue', 'grew', 'in', 'q3']
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _NUMBER_RE.sub("", text)
    return [t for t in text.split() if t.strip()]


# ---------------------------------------------------------------------------
# Loughran-McDonald sentiment ratios
# ---------------------------------------------------------------------------

def compute_lm_ratios(
    text: str,
    lm_dict: dict[str, set[str]],
) -> dict[str, float]:
    """Compute Loughran-McDonald word-category ratios for *text*.

    Args:
        text: Plain-text document (e.g. full prospectus or Risk Factors
            section).
        lm_dict: LM dictionary as returned by :func:`load_lm_dictionary`.

    Returns:
        Dict mapping feature name (e.g. ``"lm_negative_ratio"``) to the
        fraction of total words in the LM category.  Returns NaN for each
        category if *text* is empty.
    """
    tokens = tokenise(text)
    n = len(tokens)
    upper_tokens = [t.upper() for t in tokens]

    if n == 0:
        return {f"{k}_ratio": float("nan") for k in lm_dict}

    ratios: dict[str, float] = {"word_count": float(n)}
    for category, word_set in lm_dict.items():
        count = sum(1 for t in upper_tokens if t in word_set)
        ratios[f"{category}_ratio"] = count / n

    return ratios


# ---------------------------------------------------------------------------
# Gunning Fog Index (readability)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    """Estimate the syllable count in *word* using a vowel-cluster heuristic.

    Args:
        word: A single lowercase word.

    Returns:
        Estimated syllable count (minimum 1).
    """
    word = word.lower().strip("'.,-")
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent 'e'
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _is_complex_word(word: str) -> bool:
    """Return True if *word* has three or more syllables.

    Excludes proper nouns (all-caps tokens) and common suffixes.

    Args:
        word: Lowercase word token.

    Returns:
        Boolean indicating whether the word is 'complex' per Fog Index rules.
    """
    # Exclude common 3-syllable non-complex suffixes
    if word.endswith(("ing", "ed", "es", "ly")):
        count = _count_syllables(word[:-3]) if word.endswith("ing") else _count_syllables(word[:-2])
        return count >= 3
    return _count_syllables(word) >= 3


def gunning_fog_index(text: str) -> float:
    """Compute the Gunning Fog Index for *text*.

    Fog Index = 0.4 × (words/sentences + 100 × complex_words/words)

    A score of 12 corresponds to high-school reading level; scores >18 are
    considered 'impenetrable'.

    Args:
        text: Plain-text document (typically the MD&A section).

    Returns:
        Fog Index as a float, or ``NaN`` if *text* is too short.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    n_sentences = len(sentences)

    tokens = tokenise(text)
    n_words = len(tokens)

    if n_sentences == 0 or n_words == 0:
        return float("nan")

    n_complex = sum(1 for t in tokens if _is_complex_word(t))

    fog = 0.4 * ((n_words / n_sentences) + 100 * (n_complex / n_words))
    return round(fog, 4)
