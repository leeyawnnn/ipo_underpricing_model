"""
Shared figure style. Every figure in this repository is routed through here.

The rules below exist so that eight figures read as one set rather than eight
one-off scripts:

* **Format.** SVG for anything whose legibility depends on text or hairlines;
  PNG at 200 dpi only where a raster genuinely wins (dense scatter, heatmaps).
  GitHub renders README images at roughly 880 px, so nothing is authored wider
  than 1200 px and PNGs are kept under 250 KB.
* **Type.** One sans-serif stack. Title 15pt semibold, axis label 12pt, tick
  10pt, annotation 9pt, provenance footer 8pt italic. Nothing smaller.
* **Colour.** The Okabe-Ito qualitative palette, which is distinguishable under
  all common forms of colour vision deficiency. Sequential data uses viridis;
  diverging data uses a perceptually uniform map pinned at the true neutral
  value. Colour never carries meaning on its own.
* **Uncertainty.** Any plotted estimate with sampling error carries an interval.
  :func:`bootstrap_ci` is the single implementation.
* **Provenance.** :func:`finish` stamps every figure with its source, the
  as-of date and the git commit that produced it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

# 1200 x 750 px at 100 dpi. GitHub downsamples anything wider, so this is the
# ceiling rather than a starting point.
FIG_WIDTH_IN = 12.0
FIG_HEIGHT_IN = 7.5
BASE_DPI = 100
RASTER_DPI = 200

FONT_STACK = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
    "sans-serif",
]

SIZE_TITLE = 15
SIZE_SUBTITLE = 11
SIZE_LABEL = 12
SIZE_TICK = 10
SIZE_ANNOTATION = 9
SIZE_FOOTER = 8

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

# Okabe, M. & Ito, K. (2008), "Color Universal Design".
# https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}
QUALITATIVE = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
]

INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#D9D9D9"
SEQUENTIAL = "viridis"
DIVERGING = "RdBu_r"

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

DEFAULT_SOURCE = (
    "Sources: stockanalysis.com IPO calendar; SEC EDGAR S-1/F-1 filings; "
    "Yahoo Finance first-day prices; Loughran-McDonald master dictionary."
)


def git_commit() -> str:
    """Return the short git SHA of the working tree, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        sha = out.stdout.strip()
        return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unknown"


def apply_style() -> None:
    """Install the repository-wide matplotlib defaults. Idempotent."""
    mpl.rcParams.update(
        {
            "figure.figsize": (FIG_WIDTH_IN, FIG_HEIGHT_IN),
            "figure.dpi": BASE_DPI,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": RASTER_DPI,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": SIZE_TICK,
            "text.color": INK,
            "axes.titlesize": SIZE_TITLE,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": SIZE_LABEL,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.prop_cycle": cycler(color=QUALITATIVE),
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": SIZE_ANNOTATION,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0,
        }
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def percent_formatter(decimals: int = 1) -> FuncFormatter:
    """Axis formatter turning 0.123 into '12.3%'."""
    return FuncFormatter(lambda v, _pos: f"{v * 100:.{decimals}f}%")


def usd_formatter(decimals: int = 0) -> FuncFormatter:
    """Axis formatter turning 1234567 into '$1,234,567'."""
    return FuncFormatter(lambda v, _pos: f"${v:,.{decimals}f}")


def diverging_norm(vmin: float, vmax: float, center: float = 0.0) -> TwoSlopeNorm:
    """Return a diverging norm pinned at *center*, with the range widened if needed."""
    lo = min(vmin, center - 1e-9)
    hi = max(vmax, center + 1e-9)
    return TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 20240517,
) -> tuple[float, float, float]:
    """Return ``(point, lower, upper)`` for a statistic by percentile bootstrap.

    Args:
        values: Sample to resample.
        statistic: Function applied to each resample. Defaults to the median.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided error rate; 0.05 gives a 95% interval.
        seed: Fixed so figures are byte-reproducible.

    Returns:
        ``(point estimate, lower bound, upper bound)``. Bounds are ``nan``
        when fewer than three finite observations are supplied.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if arr.size < 3:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.apply_along_axis(statistic, 1, arr[draws])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 20240517,
) -> tuple[float, float, float]:
    """Return ``(rho, lower, upper)`` for Spearman's rho by percentile bootstrap."""
    from scipy import stats as sps

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[keep], ya[keep]
    if xa.size < 5:
        return float("nan"), float("nan"), float("nan")

    rho = float(sps.spearmanr(xa, ya).statistic)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, xa.size, size=xa.size)
        boots[i] = sps.spearmanr(xa[idx], ya[idx]).statistic
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rho, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Place a finding-led title and a sample/period/units subtitle.

    Args:
        fig: Target figure.
        title: States the finding, not the variable names.
        subtitle: Sample, period and units.
    """
    fig.suptitle(
        title, fontsize=SIZE_TITLE, fontweight="semibold", x=0.012, ha="left", y=0.975, color=INK
    )
    fig.text(0.012, 0.932, subtitle, fontsize=SIZE_SUBTITLE, ha="left", color=MUTED)


def direct_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    **kwargs,
) -> None:
    """Write a series label on the plot instead of adding a legend entry."""
    ax.annotate(
        text,
        xy=(x, y),
        fontsize=SIZE_ANNOTATION,
        fontweight="semibold",
        color=color,
        va="center",
        **kwargs,
    )


def callout(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    color: str = INK,
) -> None:
    """Annotate the reader's takeaway with a short note and a leader line."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=SIZE_ANNOTATION,
        color=color,
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": MUTED,
            "linewidth": 0.9,
            "shrinkA": 2,
            "shrinkB": 4,
        },
    )


def finish(
    fig: plt.Figure,
    path: Path,
    source: str = DEFAULT_SOURCE,
    as_of: str = "",
    rect: tuple[float, float, float, float] = (0.012, 0.055, 0.988, 0.915),
) -> Path:
    """Stamp provenance, lay out with explicit margins, and write the figure.

    Successive figures use the same ``rect`` rather than ``bbox_inches="tight"``
    so that images sit at a consistent size in the README.

    Args:
        fig: Figure to write.
        path: Destination. The suffix selects the format; ``.png`` is written
            at 200 dpi, ``.svg`` as vector.
        source: Data-source line for the footer.
        as_of: As-of date for the underlying data.
        rect: ``(left, bottom, right, top)`` in figure coordinates.

    Returns:
        The path written.
    """
    footer = source if not as_of else f"{source} Data as of {as_of}."
    fig.text(
        0.012,
        0.014,
        f"{footer}  Generated by scripts/build_figures.py at commit {git_commit()}.",
        fontsize=SIZE_FOOTER,
        style="italic",
        color=MUTED,
        ha="left",
    )
    fig.tight_layout(rect=rect)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=RASTER_DPI if path.suffix == ".png" else None)
    plt.close(fig)
    return path
