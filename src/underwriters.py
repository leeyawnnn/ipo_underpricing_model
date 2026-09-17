"""
Underwriter identification and Carter-Manaster reputation ranking.

The IPO calendar this project scrapes carries no underwriter field at all, so
the syndicate has to come from the prospectus itself. Every S-1 cover page
names its underwriters in the block immediately above the prospectus date;
this module locates that block, normalises the names, and joins them to Jay
Ritter's Carter-Manaster reputation rankings.

Two design choices are worth stating.

**Ranks are period-specific.** Ritter publishes a separate column per ranking
window and reputations move (Credit Suisse 8.5 in 2018-20, 8.0 in 2021-23).
:func:`rank_for` takes the listing year and uses the applicable column, not the
latest one.

**The syndicate measure is the maximum rank, not the first name listed.**
Extraction order on a cover page is reliable for a sole bookrunner but not for
a five-bank syndicate typeset across several lines, whereas the set of banks
named is robust. ``max_underwriter_rank`` is therefore the headline measure and
``lead_underwriter`` (first by position) is reported alongside it for
inspection.

Name matching is a committed, auditable alias table plus a small set of
suffix-stripping rules. There is no fuzzy matching: a name either resolves to a
Ritter entry or is recorded as unmatched, and the match rate is reported.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils import setup_logging

log = setup_logging(__name__)

RANKS_PATH = Path("data/external/underwriter_ranks.csv")

# Characters of the filing searched for the syndicate block. The cover page
# runs to roughly 4,500 characters of normalised text at the median and 8,700
# at the 90th percentile; 20,000 covers the tail of long SPAC covers while
# stopping well before the Risk Factors section, where bank names recur for
# unrelated reasons.
COVER_WINDOW_CHARS = 20_000

# Legal-form suffixes stripped before matching.
_SUFFIX_RE = re.compile(
    r"\b("
    r"llc|l\.l\.c|inc|incorporated|corp|corporation|co|company|ltd|limited|"
    r"lp|l\.p|llp|plc|ag|sa|s\.a|nv|n\.v|gmbh|pllc|holdings?|group|"
    r"securities|capital\s+markets|investment\s+bank|and\s+co"
    r")\b\.?",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    """Reduce an underwriter name to a comparison key.

    Lowercases, drops parenthetical notes, strips legal-form suffixes and
    punctuation, and collapses whitespace.

    Args:
        name: Raw underwriter name, from a prospectus or from Ritter's file.

    Returns:
        Normalised key, possibly empty if the name was purely a suffix.

    Example:
        >>> normalise_name("Goldman Sachs & Co. LLC")
        'goldman sachs'
        >>> normalise_name("J.P. Morgan Securities LLC")
        'jp morgan'
    """
    text = str(name or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)  # drop "(JPM)", "(see ...)"
    text = text.replace("&", " and ")
    text = _SUFFIX_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------
# Maps the name as it appears on a prospectus cover page to the name Ritter
# uses. Every entry was added because an actual filing in this sample used that
# form; the table is committed so a reader can audit and extend it.
#
# Successor entities are mapped to the acquirer where Ritter carries only the
# acquirer: Credit Suisse's US equity business moved to UBS in 2023, SVB
# Leerink became Leerink Partners in 2023, Cowen became TD Cowen in 2023.
# Ritter carries separate rows for several of these, so they are mapped to
# their own entries rather than merged.
ALIASES: dict[str, str] = {
    # Bulge bracket
    "goldman sachs": "Goldman Sachs & Co",
    "goldman sachs and": "Goldman Sachs & Co",
    "morgan stanley": "Morgan Stanley & Co",
    "j p morgan": "JP Morgan (JPM)",
    "jp morgan": "JP Morgan (JPM)",
    "jpmorgan": "JP Morgan (JPM)",
    "jpmorgan chase": "JP Morgan (JPM)",
    "bofa": "Bank of America-Merrill Lynch (BOA-Merrill)",
    "bofa merrill lynch": "Bank of America-Merrill Lynch (BOA-Merrill)",
    "bank of america": "Bank of America-Merrill Lynch (BOA-Merrill)",
    "merrill lynch pierce fenner and smith": "Bank of America-Merrill Lynch (BOA-Merrill)",
    "merrill lynch": "Merrill Lynch & Co Inc",
    "citigroup": "Citigroup",
    "citigroup global markets": "Citigroup Global Markets Inc",
    "barclays": "Barclays Capital",
    "deutsche bank": "Deutsche Bank Securities Corp",
    "credit suisse": "Credit Suisse",
    "ubs": "UBS Investment Bank",
    "ubs investment bank": "UBS Investment Bank",
    "wells fargo": "Wells Fargo",
    "jefferies": "Jefferies & Co Inc",
    "rbc": "RBC Capital Markets",
    "rbc capital markets": "RBC Capital Markets",
    "bmo": "BMO Capital Markets",
    "bmo capital markets": "BMO Capital Markets",
    "hsbc": "HSBC",
    "nomura": "Nomura Securities",
    "mizuho": "Mizuho Securities",
    "bnp paribas": "BNP Paribas SA (BNP-CA)",
    "societe generale": "Societe Generale",
    "santander": "Santander Investment Bank",
    "scotiabank": "Scotia Capital Inc",
    "scotia": "Scotia Capital Inc",
    "td": "TD Securities Inc",
    "td cowen": "TD Cowen",
    "mufg": "MUFG",
    "smbc nikko": "SMBC Nikko",
    "macquarie": "Macquarie Bank",
    "berenberg": "Berenberg",
    "itau bba": "Itau BBA",
    "btg pactual": "BTG Pactual",
    # Advisory / merchant banks
    "evercore": "Evercore",
    "evercore isi": "Evercore",
    "moelis": "Moelis",
    "pjt partners": "PJT Partners",
    "allen and": "Allen & Co Inc",
    "allen": "Allen & Co Inc",
    "guggenheim": "Guggenheim Securities",
    "guggenheim securities": "Guggenheim Securities",
    "kkr": "KKR Capital",
    "tpg": "TPG Capital",
    "apollo": "Apollo",
    "liontree": "Liontree",
    "lucid": "Lucid",
    # Mid-cap / growth
    "william blair": "William Blair & Co",
    "piper sandler": "Piper-Sandler",
    "piper jaffray": "Piper Jaffray Inc",
    "stifel": "Stifel",
    "stifel nicolaus": "Stifel Nicolaus & Co Inc",
    "raymond james": "Raymond James & Associates Inc",
    "cowen": "Cowen",
    "cowen and": "Cowen",
    "canaccord genuity": "Canaccord Genuity",
    "canaccord": "Canaccord Genuity",
    "oppenheimer": "Oppenheimer & Co Inc",
    "needham": "Needham & Co Inc",
    "needham and": "Needham & Co Inc",
    "truist": "Truist Securities",
    "truist securities": "Truist Securities",
    "suntrust robinson humphrey": "SunTrust Robinson Humphrey",
    "keybanc": "KeyBanc Capital Markets",
    "wedbush": "Wedbush Morgan Securities",
    "baird": "Robert W Baird & Co Inc",
    "robert w baird": "Robert W Baird & Co Inc",
    "stephens": "Stephens Inc",
    "d a davidson": "DA Davidson & Co Inc",
    "da davidson": "DA Davidson & Co Inc",
    "janney montgomery scott": "Janney Montgomery Scott Inc",
    "keefe bruyette and woods": "Keefe Bruyette & Woods Inc",
    "kbw": "Keefe Bruyette & Woods Inc",
    "sandler o neill": "Sandler O'Neill Partners",
    "craig hallum": "Craig-Hallum, Inc.",
    "craig hallum partners": "Craig-Hallum, Inc.",
    "northland": "Northland Capital Markets",
    "northland capital markets": "Northland Capital Markets",
    "loop capital": "Loop Capital",
    "loop": "Loop Capital",
    "b riley": "B-Riley",
    "b riley fbr": "B-Riley-FBR",
    "b riley financial": "B-Riley",
    "jmp": "JMP-Sec",
    "jmp securities": "JMP-Sec",
    "leerink partners": "Leerink Partners",
    "svb leerink": "SVB-Leerink",
    "svb securities": "SVB-Leerink",
    "cantor": "Cantor",
    "cantor fitzgerald": "Cantor, Fitzgerald & Co., Inc.",
    "btig": "BTIG",
    "roth": "Roth Capital Partners Inc",
    "roth capital partners": "Roth Capital Partners Inc",
    "roth mkm": "Roth Capital Partners Inc",
    "tudor pickering holt": "Tudor-Pickering",
    "seaport global": "Seaport Global Securities",
    "hovde": "Hovde Group",
    "performance trust": "Performance Trust",
    "clear street": "Clear Street",
    "jones trading": "Jones Trading",
    "jonestrading": "Jones Trading",
    "rosenblatt": "Rosenblatt Securities",
    "lifesci": "LifeSci Capital",
    "lifesci capital": "LifeSci Capital",
    "h c wainwright": "HC Wainwright & Co Inc",
    "hc wainwright": "HC Wainwright & Co Inc",
    "wainwright": "HC Wainwright & Co Inc",
    "lake street": "Lake Street Capital Markets",
    "cohen and": "Cohen & Co",
    "cohen and company": "Cohen & Co",
    "cibc": "CIBC Oppenheimer",
    "clarksons platou": "Clarksons Platou",
    "kempen": "Kempen",
    "mdb": "MDB-Capital",
    "oak ridge financial": "Oak Ridge Financial",
    "odeon": "Odeon Capital Group",
    "code advisors": "Code Advisors",
    "sofi": "SoFi Securities",
    "webull": "Webull Financial",
    "moffettnathanson": "SVB-Leerink",
    # Small-cap and SPAC underwriters
    "ef hutton": "EF Hutton & Co Inc (EFH)",
    "e f hutton": "EF Hutton & Co Inc (EFH)",
    "d boral capital": "D. Boral Capital",
    "chardan": "Chardan Capital Markets, LLC",
    "chardan capital markets": "Chardan Capital Markets, LLC",
    "maxim": "Maxim Group LLC",
    "maxim group": "Maxim Group LLC",
    "thinkequity": "Thinkequity Partners LLC",
    "think equity": "Thinkequity Partners LLC",
    "aegis": "Aegis Capital",
    "aegis capital": "Aegis Capital",
    "i bankers": "I-Bankers Securities Inc",
    "earlybirdcapital": "EarlyBirdCapital Inc",
    "early bird capital": "EarlyBirdCapital Inc",
    "ladenburg thalmann": "Ladenburg Thalmann & Co",
    "boustead": "Boustead Securities",
    "boustead securities": "Boustead Securities",
    "national securities corp": "National Securities Corp",
    "national securities corporation": "National Securities Corp",
    "kingswood": "Kingswood",
    "spac advisory partners": "SPAC Advisory (division of Kingswood)",
    "alexander capital": "Alexander Capital, LP",
    "dawson james": "Dawson James Securities, Inc.",
    "joseph gunnar": "Joseph Gunnar & Co., LLC",
    "joseph stone": "Joseph Stone Capital",
    "prime number": "Prime Number Capital (PNCPS)",
    "prime number capital": "Prime Number Capital (PNCPS)",
    "westpark capital": "West Park Capital",
    "west park capital": "West Park Capital",
    "wallachbeth": "WallachBeth Capital",
    "network 1 financial": "Network 1 Financial Securities",
    "revere": "Revere Securities",
    "univest": "Univest Securities",
    "newbridge": "Newbridge Securities",
    "spartan capital": "Spartan Group (Spartan Capital)",
    "benchmark": "The Benchmark Company",
    "the benchmark": "The Benchmark Company",
    "bancroft": "Bancroft Capital, LLC",
    "craft capital": "Craft Capital",
    "dominari": "Dominari Securities",
    "freedom capital markets": "Freedom Capital Markets",
    "american trust investment": "American Trust Investment Services",
    "r f lafferty": "R. F. Lafferty",
    "laidlaw": "Laidlaw & Co (UK)",
    "titan partners": "Titan",
    "wilson davis": "Wilson-Davis",
    "drexel hamilton": "Drexel Hamilton",
    "castleoak": "Castleoak",
    "brookline": "Brookline",
    "feltl": "Feltl & Co",
    "paulson investment": "Paulson Investment Co",
    "a g p": "A.G.P/Alliance Global Partners",
    "agp": "A.G.P/Alliance Global Partners",
    "alliance global partners": "A.G.P/Alliance Global Partners",
    "imperial capital": "Imperial-Cap",
    "axiom capital": "Axiom-Cap",
    "eddid": "Eddid Securities USA",
    "us tiger": "US Tiger",
    "tiger brokers": "Tiger Brokers (NZ)",
    "futu": "Futu Securities International (HK)",
    "amtd": "AMTD Global Markets",
    "valuable capital": "Valuable Capital",
    "haitong": "Haitong International",
    "cicc": "China International Capital Corp (CICC)",
    "china international capital": "China International Capital Corp (CICC)",
    "china renaissance": "China Renaissance (CRS)",
    "cmb international": "CMB International Capital Ltd",
    "huatai": "Huatai Securities",
    "clsa": "CLSA",
    "boci": "BOCI Asia",
    "icbc international": "ICBC-Int",
    "ping an": "Ping An",
    "cathay": "Cathay",
    "pacific century": "Pacific Century",
    "gf securities": "GF Securities (Hong Kong)",
    "china merchants": "China Merchants Securities",
    "absa": "Absa Bank Ltd",
    "rabo": "Rabo Securities",
    "banco bradesco": "Banco Bradesco",
    "bradesco bbi": "Banco Bradesco",
    "nu invest": "Nu Invest",
    "bc partners": "BC Partners",
    "john nuveen": "John Nuveen Co",
    "ac sunshine": "AC Sunshine Securities LLC",
    "benjamin securities": "Benjamin Securities",
    "werbel roth": "Werbel-Roth Securities",
    "digital offering": "Digital",
    "viewtrade": "VIEWT-SEC",
}


@lru_cache(maxsize=1)
def load_ranks(path: str = str(RANKS_PATH)) -> pd.DataFrame:
    """Load the long-format Ritter rank table produced by the fetch script.

    Args:
        path: Path to ``underwriter_ranks.csv``.

    Returns:
        DataFrame with columns ``underwriter``, ``year``, ``rank`` and a
        ``key`` column holding the normalised name.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Underwriter ranks not found at {p}. Run: python scripts/fetch_underwriter_ranks.py"
        )
    ranks = pd.read_csv(p)
    ranks["key"] = ranks["underwriter"].map(normalise_name)
    return ranks


# Ritter names whose normalised key is an ordinary English word. Matching these
# bare against prospectus prose produces false positives ("digital offering",
# "the benchmark for", "Apollo program"), so they are reachable only through an
# explicit ALIASES entry.
GENERIC_KEYS: frozenset[str] = frozenset(
    {
        "digital",
        "apollo",
        "titan",
        "lucid",
        "benchmark",
        "brookline",
        "cathay",
        "davidson",
        "clear street",
        "performance trust",
        "code advisors",
        "benjamin securities",
        "benjamin",
        "national",
        "national securities",
        "kingswood",
        "freedom",
        "allen",
        "loop",
        "stephens",
        "hovde",
        "berenberg",
        "kempen",
        "cantor",
        "moelis",
        "evercore",
    }
)


@lru_cache(maxsize=1)
def _search_terms() -> list[tuple[str, str]]:
    """Return (search key, Ritter name) pairs sorted longest-key-first.

    Longest first so that "goldman sachs international" wins over "goldman
    sachs" when both would match.
    """
    ranks = load_ranks()
    terms: dict[str, str] = {}
    for key, canonical in ALIASES.items():
        terms[key] = canonical
    for canonical, key in zip(ranks["underwriter"], ranks["key"]):
        if key and key not in terms and key not in GENERIC_KEYS:
            terms[key] = canonical
    # Two-character keys are too ambiguous to search for in free text.
    return sorted(
        ((k, v) for k, v in terms.items() if len(k) >= 3),
        key=lambda kv: -len(kv[0]),
    )


# The "Copies to:" block on a cover page lists the issuer's and underwriters'
# outside counsel by personal name. Those names collide with underwriter names
# ("Michael Benjamin, Latham & Watkins" matched Benjamin Securities; "Morrison
# Cohen LLP" matched Cohen & Co), so the block is cut before matching. It runs
# from "Copies to" to the next cover-page landmark.
_COUNSEL_START_RE = re.compile(r"copies\s+to\s*:", re.IGNORECASE)
_COUNSEL_END_RE = re.compile(
    r"(approximate\s+date\s+of\s+(?:the\s+)?(?:commencement|proposed)"
    r"|calculation\s+of\s+registration\s+fee"
    r"|if\s+any\s+of\s+the\s+securities"
    r"|the\s+registrant\s+hereby\s+amends"
    r"|proposed\s+maximum\s+aggregate)",
    re.IGNORECASE,
)
# How far past "Copies to:" to cut when no landmark follows it.
_COUNSEL_MAX_CHARS = 2_500


def strip_counsel_block(cover: str) -> str:
    """Remove the outside-counsel block from a cover page.

    Args:
        cover: Whitespace-normalised cover-page text.

    Returns:
        *cover* with each "Copies to:" span replaced by a single space.
    """
    out = cover
    while True:
        start = _COUNSEL_START_RE.search(out)
        if start is None:
            return out
        end_match = _COUNSEL_END_RE.search(out, start.end())
        end = (
            min(end_match.start(), start.end() + _COUNSEL_MAX_CHARS)
            if end_match
            else min(start.end() + _COUNSEL_MAX_CHARS, len(out))
        )
        out = out[: start.start()] + " " + out[end:]


def cover_page_text(full_text: str, window: int = COVER_WINDOW_CHARS) -> str:
    """Return the whitespace-normalised cover-page region of a prospectus.

    The outside-counsel block is removed; see :func:`strip_counsel_block`.
    """
    return strip_counsel_block(_WS_RE.sub(" ", full_text[: window * 3])[:window])


def _shares_identity(underwriter_key: str, issuer_key: str) -> bool:
    """Return True if the two normalised names refer to the same house.

    Containment either way: an issuer called "Futu Holdings" matches the
    underwriter "Futu Securities International (HK)", whose key is
    "futu international".
    """
    if not underwriter_key or len(underwriter_key) < 4:
        return False
    return underwriter_key in issuer_key or issuer_key in underwriter_key


def extract_underwriters(full_text: str, issuer_name: str | None = None) -> list[str]:
    """Return the Ritter-canonical underwriters named on the cover page.

    Args:
        full_text: Plain text of the S-1 / F-1 filing.
        issuer_name: The registrant's own name. Several issuers in this sample
            are themselves brokers (Futu Holdings, Marex, SoFi), and their name
            dominates the cover page, so a match to the issuer is dropped.

    Returns:
        Canonical Ritter names, ordered by first appearance. Empty if no
        recognised underwriter appears in the cover-page window.
    """
    cover = cover_page_text(full_text).lower()
    cover = cover.replace("&", " and ")
    cover = _PUNCT_RE.sub(" ", cover)
    cover = _WS_RE.sub(" ", cover)

    found: dict[str, int] = {}
    consumed = [False] * len(cover)
    for key, canonical in _search_terms():
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", cover):
            start, end = match.span()
            if any(consumed[start:end]):
                continue  # already claimed by a longer, more specific name
            for i in range(start, end):
                consumed[i] = True
            if canonical not in found or start < found[canonical]:
                found[canonical] = start
            break  # first occurrence of this name is enough

    names = [name for name, _ in sorted(found.items(), key=lambda kv: kv[1])]

    if issuer_name:
        issuer_key = normalise_name(issuer_name)
        if len(issuer_key) >= 4:
            names = [n for n in names if not _shares_identity(normalise_name(n), issuer_key)]

    return names


def rank_for(underwriter: str, year: int) -> float:
    """Return the Carter-Manaster rank for *underwriter* in listing *year*.

    Args:
        underwriter: Canonical Ritter name, as returned by
            :func:`extract_underwriters`.
        year: Listing year, used to select Ritter's applicable rank column.

    Returns:
        Rank on the 0-9 Carter-Manaster scale, or ``nan`` if Ritter has no
        rank for that underwriter in that year.
    """
    ranks = load_ranks()
    hit = ranks[(ranks["underwriter"] == underwriter) & (ranks["year"] == year)]
    if hit.empty:
        return float("nan")
    return float(hit["rank"].iloc[0])


def syndicate_summary(full_text: str, year: int, issuer_name: str | None = None) -> dict:
    """Summarise the underwriting syndicate named on a prospectus cover page.

    Args:
        full_text: Plain text of the S-1 / F-1 filing.
        year: Listing year, for period-specific ranks.
        issuer_name: The registrant's own name, excluded from the match set.

    Returns:
        Dict with ``lead_underwriter`` (first name by position),
        ``n_underwriters_matched``, ``max_underwriter_rank``,
        ``lead_underwriter_rank`` and ``underwriter_syndicate`` (a
        semicolon-joined list).
    """
    names = extract_underwriters(full_text, issuer_name=issuer_name)
    if not names:
        return {
            "lead_underwriter": None,
            "underwriter_syndicate": "",
            "n_underwriters_matched": 0,
            "max_underwriter_rank": float("nan"),
            "lead_underwriter_rank": float("nan"),
        }

    ranks = [rank_for(n, year) for n in names]
    valid = [r for r in ranks if pd.notna(r)]
    return {
        "lead_underwriter": names[0],
        "underwriter_syndicate": "; ".join(names),
        "n_underwriters_matched": len(names),
        "max_underwriter_rank": max(valid) if valid else float("nan"),
        "lead_underwriter_rank": ranks[0],
    }
