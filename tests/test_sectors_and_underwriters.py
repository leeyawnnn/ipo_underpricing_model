"""Tests for SIC-based sector classification and cover-page underwriter matching.

Both replaced heuristics that could not be audited. These tests pin the layers
that matter and act as regression guards on real coverage: the previous sector
classifier silently sent 29% of the dataset to Industrials, and no test would
have noticed.
"""

from __future__ import annotations

import math

import pytest

from src.sic_sectors import UNCLASSIFIED, load_crosswalk, sector_for_sic
from src.underwriters import (
    GENERIC_KEYS,
    extract_underwriters,
    normalise_name,
    rank_for,
    strip_counsel_block,
    syndicate_summary,
)

# ---------------------------------------------------------------------------
# SIC crosswalk
# ---------------------------------------------------------------------------


def test_crosswalk_ranges_do_not_overlap():
    # load_crosswalk raises on overlap; calling it is the assertion.
    walk = load_crosswalk()
    assert len(walk) > 50
    assert (walk["sic_low"] <= walk["sic_high"]).all()


@pytest.mark.parametrize(
    ("sic", "expected"),
    [
        (6770, "SPAC"),  # the SEC's own "Blank Checks" code
        (2834, "Healthcare"),  # pharmaceutical preparations
        (2836, "Healthcare"),  # biological products
        (2821, "Materials"),  # plastics, just below the drug block
        (7372, "Technology"),  # prepackaged software
        (7310, "Communication Services"),
        (1311, "Energy"),  # crude petroleum and natural gas
        (6798, "Real Estate"),  # REITs, carved out of the 67xx block
        (6022, "Financials"),
        (4911, "Utilities"),
        (3711, "Consumer Discretionary"),
        (8731, "Healthcare"),  # commercial physical and biological research
        (8711, "Industrials"),  # engineering services
    ],
)
def test_sector_for_sic_known_codes(sic, expected):
    assert sector_for_sic(sic) == expected


def test_sector_for_sic_accepts_strings_and_floats():
    assert sector_for_sic("2834") == sector_for_sic(2834) == sector_for_sic(2834.0)


@pytest.mark.parametrize("value", [None, float("nan"), "", "not-a-code", 99999])
def test_sector_for_sic_unclassified_rather_than_guessing(value):
    assert sector_for_sic(value) == UNCLASSIFIED


def test_no_silent_fallback_bucket():
    """The old classifier forced unmatched names into Industrials.

    An unmappable code must produce Unclassified, never a real sector.
    """
    assert sector_for_sic(None) != "Industrials"
    assert sector_for_sic("zzz") == UNCLASSIFIED


def test_sector_coverage_on_the_real_sample(analysis_sample):
    """Regression guard: unclassified share must stay well below the 29%
    fallback rate of the name-regex classifier it replaced."""
    share = (analysis_sample["sector"] == UNCLASSIFIED).mean()
    assert share < 0.10, f"Unclassified share rose to {share:.1%}"


# ---------------------------------------------------------------------------
# Underwriter name normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Goldman Sachs & Co. LLC", "goldman sachs"),
        ("J.P. Morgan Securities LLC", "j p morgan"),
        ("Morgan Stanley & Co. Incorporated", "morgan stanley"),
        ("Jefferies LLC", "jefferies"),
        ("JP Morgan (JPM)", "jp morgan"),
        ("Maxim Group LLC", "maxim"),
    ],
)
def test_normalise_name_strips_legal_forms_and_parentheticals(raw, expected):
    assert normalise_name(raw) == expected


def test_normalise_name_handles_empty_input():
    assert normalise_name("") == ""
    assert normalise_name(None) == ""


# ---------------------------------------------------------------------------
# Cover-page extraction
# ---------------------------------------------------------------------------

COVER = (
    "PRELIMINARY PROSPECTUS 8,250,000 Shares Common Stock "
    "Copies to: Michael Benjamin, Latham & Watkins LLP, 885 Third Avenue. "
    "Approximate date of commencement of proposed sale to the public: as soon as practicable. "
    "The underwriters expect to deliver the shares on or about , 2022. "
    "Morgan Stanley J.P. Morgan BofA Securities Canaccord Genuity "
    "Prospectus dated , 2022."
)


def test_extract_underwriters_reads_the_syndicate_in_order():
    names = extract_underwriters(COVER)
    assert names[:4] == [
        "Morgan Stanley & Co",
        "JP Morgan (JPM)",
        "Bank of America-Merrill Lynch (BOA-Merrill)",
        "Canaccord Genuity",
    ]


def test_counsel_block_is_excised_before_matching():
    # "Michael Benjamin, Latham & Watkins" must not match Benjamin Securities.
    assert "benjamin" not in strip_counsel_block(COVER).lower()
    assert "Benjamin Securities" not in extract_underwriters(COVER)


def test_issuer_name_is_not_counted_as_its_own_underwriter():
    cover = (
        "PROSPECTUS 10,000,000 American Depositary Shares Futu Holdings Limited. "
        "The underwriters expect to deliver the ADSs. "
        "Goldman Sachs (Asia) L.L.C. UBS Investment Bank Credit Suisse"
    )
    names = extract_underwriters(cover, issuer_name="Futu Holdings Limited")
    assert not any("Futu" in n for n in names)
    assert "UBS Investment Bank" in names


def test_generic_english_words_need_an_explicit_alias():
    # "digital", "national" and friends are Ritter entries whose normalised key
    # is an ordinary word; matching them bare produced false positives.
    for key in ["digital", "national", "benjamin", "apollo"]:
        assert key in GENERIC_KEYS
    cover = "PROSPECTUS 1,000,000 Shares. A national digital platform company."
    assert extract_underwriters(cover) == []


def test_extract_returns_empty_when_no_bank_is_named():
    assert extract_underwriters("PROSPECTUS 1,000,000 Shares of Common Stock.") == []


# ---------------------------------------------------------------------------
# Period-specific ranks
# ---------------------------------------------------------------------------


def test_ranks_are_period_specific():
    """Ritter publishes a rank per window; reputations move between them."""
    early = rank_for("Credit Suisse", 2019)
    late = rank_for("Credit Suisse", 2022)
    assert early == pytest.approx(8.5)
    assert late == pytest.approx(8.0)
    assert early != late


def test_rank_for_unknown_underwriter_is_nan():
    assert math.isnan(rank_for("Not A Real Bank", 2021))


def test_rank_for_unknown_year_is_nan():
    assert math.isnan(rank_for("Goldman Sachs & Co", 1995))


def test_syndicate_summary_uses_the_highest_rank():
    summary = syndicate_summary(COVER, 2022)
    assert summary["lead_underwriter"] == "Morgan Stanley & Co"
    assert summary["n_underwriters_matched"] >= 4
    assert summary["max_underwriter_rank"] == pytest.approx(9.0)


def test_syndicate_summary_on_a_cover_with_no_bank():
    summary = syndicate_summary("PROSPECTUS 1,000,000 Shares.", 2022)
    assert summary["lead_underwriter"] is None
    assert summary["n_underwriters_matched"] == 0
    assert math.isnan(summary["max_underwriter_rank"])


def test_underwriter_match_rate_on_the_real_sample(analysis_sample):
    """Regression guard on the reported match rate."""
    with_text = analysis_sample[analysis_sample["lm_litigious_ratio"].notna()]
    matched = with_text["max_underwriter_rank"].notna().mean()
    assert matched > 0.75, f"Ritter match rate fell to {matched:.1%}"
