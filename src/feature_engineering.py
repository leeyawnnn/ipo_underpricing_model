"""
Feature engineering for the IPO underpricing study.

Every feature here must be knowable before the stock's first trade. Three
places where the previous implementation was not, and what changed:

* **Market regime.** VIX and NASDAQ statistics were read with
  ``index <= ipo_date``, which on a listing day is that day's close. Now
  sourced from :func:`src.dataset.market_as_of_prior_day`, strictly lagged.
* **Prospectus uniqueness.** Hanley-Hoberg similarity was measured against a
  sector mean computed over the whole sample, so each document was compared
  against prospectuses that did not exist yet.
  :func:`expanding_prospectus_uniqueness` compares each filing only against
  earlier filings in its own sector.
* **Sector and underwriter encodings.** ``sector_encoded`` and
  ``lead_underwriter_encoded`` were the mean of the target within each
  category over the full dataset, then fed to the model and used as a control
  in the H1 regression. That is the target in disguise. They are gone; sector
  enters as fixed effects fitted inside each fold.

Deal size comes from the prospectus cover rather than the calendar, which has
no size field at all. It is the *filed* base offering, before any
over-allotment and before final pricing, and is named accordingly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import text_features
from src.underwriters import cover_page_text, syndicate_summary
from src.utils import setup_logging

log = setup_logging(__name__)

# Sections shorter than this are treated as missing rather than scored.
MIN_SECTION_WORDS = 100

# The offering size is the share count printed immediately after the
# prospectus banner, before the issuer name. Anchoring there avoids picking up
# the over-allotment line or an unrelated share count from the fee table.
_SIZE_ANCHOR_RE = re.compile(r"(prospectus|subject\s+to\s+completion)", re.IGNORECASE)
_SIZE_COUNT_RE = re.compile(
    r"\b([\d][\d,]{5,})\s+(?:shares|american\s+depositary\s+shares|adss|units)\b",
    re.IGNORECASE,
)
_SIZE_SEARCH_SPAN = 200
_COVER_SEARCH_CHARS = 8_000


# ---------------------------------------------------------------------------
# Deal size from the prospectus cover
# ---------------------------------------------------------------------------


def filed_shares_offered(full_text: str) -> float:
    """Extract the filed base offering size, in shares, from a cover page.

    Args:
        full_text: Plain text of the S-1 / F-1 filing.

    Returns:
        Share count, or ``nan`` when the cover does not state one in the
        expected position.

    Example:
        >>> filed_shares_offered("PRELIMINARY PROSPECTUS 8,250,000 Shares Common Stock")
        8250000.0
    """
    cover = cover_page_text(full_text)[:_COVER_SEARCH_CHARS]
    for anchor in _SIZE_ANCHOR_RE.finditer(cover):
        match = _SIZE_COUNT_RE.search(cover, anchor.end(), anchor.end() + _SIZE_SEARCH_SPAN)
        if match:
            return float(match.group(1).replace(",", ""))
    return float("nan")


# ---------------------------------------------------------------------------
# Prospectus uniqueness, expanding window
# ---------------------------------------------------------------------------


def expanding_prospectus_uniqueness(
    texts: list[str],
    sectors: list[str],
    dates: list[pd.Timestamp],
    min_prior: int = 5,
    max_features: int = 5_000,
) -> np.ndarray:
    """Hanley-Hoberg prospectus uniqueness against *earlier* same-sector filings.

    ``uniqueness = 1 - cosine(doc, mean of prior same-sector docs)``. A score
    near zero means boilerplate, near one means distinctive.

    The vocabulary is fit once on the whole corpus. That is a mild look-ahead
    in the vectoriser only, not in the comparison set, and it is what keeps the
    measure comparable across filings; the alternative — refitting TF-IDF at
    every observation — changes the feature space between rows and makes the
    scores incommensurable. The comparison set itself is strictly historical.

    Args:
        texts: Prospectus bodies.
        sectors: Sector label per text.
        dates: Filing or listing date per text.
        min_prior: Prior same-sector filings required before a score is
            produced; below this the score is ``nan``.
        max_features: TF-IDF vocabulary size.

    Returns:
        Array of uniqueness scores aligned to *texts*.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if not (len(texts) == len(sectors) == len(dates)):
        raise ValueError("texts, sectors and dates must be the same length")

    n = len(texts)
    out = np.full(n, np.nan)
    non_empty = [i for i, t in enumerate(texts) if t and t.strip()]
    if len(non_empty) < min_prior + 1:
        return out

    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectoriser.fit_transform([texts[i] for i in non_empty])
    matrix = normalize(matrix)

    order = sorted(range(len(non_empty)), key=lambda k: (dates[non_empty[k]], non_empty[k]))
    running: dict[str, list[int]] = {}

    for position in order:
        original = non_empty[position]
        sector = sectors[original]
        prior = running.get(sector, [])
        if len(prior) >= min_prior:
            centroid = np.asarray(matrix[prior].mean(axis=0)).ravel()
            norm = np.linalg.norm(centroid)
            if norm > 0:
                doc = np.asarray(matrix[position].todense()).ravel()
                out[original] = float(1.0 - float(doc @ centroid) / norm)
        running.setdefault(sector, []).append(position)

    return out


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------


def _read(path: object) -> str:
    """Read a text file named by a possibly-missing path column value."""
    if not isinstance(path, str) or not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach LM sentiment ratios, readability, deal size and syndicate fields.

    Args:
        df: Output of :func:`src.dataset.assemble`, with path columns.

    Returns:
        Copy of *df* with:
        ``lm_*_ratio`` and ``word_count`` from the full prospectus,
        ``rf_lm_*_ratio`` and ``rf_word_count`` from Risk Factors,
        ``gunning_fog`` from the MD&A, ``prospectus_uniqueness``,
        ``filed_shares_offered`` / ``filed_offer_size_usd``, and the
        underwriter fields from :func:`src.underwriters.syndicate_summary`.
    """
    df = df.copy()
    lm_dict = text_features.load_lm_dictionary()

    records: list[dict] = []
    texts: list[str] = []

    for _, row in df.iterrows():
        record: dict = {}
        full_text = _read(row.get("full_text_path"))
        texts.append(full_text)

        if full_text:
            ratios = text_features.compute_lm_ratios(full_text, lm_dict)
            if ratios.get("word_count", 0) >= MIN_SECTION_WORDS:
                record.update(ratios)
            record["filed_shares_offered"] = filed_shares_offered(full_text)
            record.update(
                syndicate_summary(
                    full_text,
                    int(row["ipo_year"]),
                    issuer_name=row.get("company_name"),
                )
            )

        risk_text = _read(row.get("risk_factors_path"))
        if risk_text:
            ratios = text_features.compute_lm_ratios(risk_text, lm_dict)
            if ratios.get("word_count", 0) >= MIN_SECTION_WORDS:
                record.update({f"rf_{k}": v for k, v in ratios.items()})

        mda_text = _read(row.get("mda_path"))
        if mda_text:
            tokens = text_features.tokenise(mda_text)
            if len(tokens) >= MIN_SECTION_WORDS:
                record["gunning_fog"] = text_features.gunning_fog_index(mda_text)
                record["mda_word_count"] = float(len(tokens))

        records.append(record)

    features = pd.DataFrame(records, index=df.index)
    for column in features.columns:
        df[column] = features[column]

    df["prospectus_uniqueness"] = expanding_prospectus_uniqueness(
        texts,
        df["sector"].fillna("Unclassified").tolist(),
        pd.to_datetime(df["ipo_date"]).tolist(),
    )
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transforms and ratios built from columns already present.

    Args:
        df: DataFrame carrying offer price, filed size, LM ratios and
            underwriter ranks.

    Returns:
        Copy of *df* with ``log_offer_price``, ``log_offer_size``,
        ``log_prospectus_words``, ``risk_concentration_ratio`` and
        ``top_tier_underwriter``.
    """
    df = df.copy()

    df["log_offer_price"] = np.log(
        pd.to_numeric(df["offer_price"], errors="coerce").clip(lower=0.01)
    )

    if "filed_shares_offered" in df.columns:
        df["filed_offer_size_usd"] = pd.to_numeric(
            df["filed_shares_offered"], errors="coerce"
        ) * pd.to_numeric(df["offer_price"], errors="coerce")
        df["log_offer_size"] = np.log1p(df["filed_offer_size_usd"])

    if "word_count" in df.columns:
        df["log_prospectus_words"] = np.log1p(pd.to_numeric(df["word_count"], errors="coerce"))

    # Where the negative tone sits: the share of the prospectus's LM-negative
    # intensity that lives in Risk Factors. Higher means compartmentalised.
    if {"rf_lm_negative_ratio", "lm_negative_ratio"}.issubset(df.columns):
        denominator = pd.to_numeric(df["lm_negative_ratio"], errors="coerce")
        numerator = pd.to_numeric(df["rf_lm_negative_ratio"], errors="coerce")
        df["risk_concentration_ratio"] = np.where(denominator > 0, numerator / denominator, np.nan)

    # Carter-Manaster 8 is the conventional prestige cut-off (Loughran & Ritter
    # 2004). Applied to the highest-ranked bank in the syndicate.
    if "max_underwriter_rank" in df.columns:
        rank = pd.to_numeric(df["max_underwriter_rank"], errors="coerce")
        df["top_tier_underwriter"] = np.where(rank.notna(), (rank >= 8).astype(float), np.nan)

    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering sequence.

    Args:
        df: Output of :func:`src.dataset.assemble`.

    Returns:
        DataFrame with every engineered feature appended.
    """
    log.info(
        "Extracting text, syndicate and deal-size features from %d filings …",
        int(df["has_filing"].sum()),
    )
    df = add_text_features(df)
    log.info("Adding derived features …")
    df = add_derived_features(df)
    log.info("Feature engineering complete: %s", df.shape)
    return df
