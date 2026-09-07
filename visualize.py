"""Diagnostic plots for the holding-period return distributions.

``visualize(level, dist, summary)`` draws:
  1. underlying index level (log scale)
  2. total-return fan chart
  3. annualised-return fan chart
  4. moments of the ANNUALISED-return distribution vs holding period
  5. P(annualised return < threshold) for several thresholds

``visualize_from_disk(name, method)`` loads data/processed/* and plots one set.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kurtosis, skew

import config

_BASE_CMAP = plt.get_cmap("Blues")
_EDGES = np.asarray(config.PERCENTILE_EDGES)


# --- small helpers -------------------------------------------------------

def annualize(r_pct, h_days):
    """Total return (%) over h_days -> annualised return (% p.a.)."""
    return ((1 + r_pct / 100) ** (config.DAYS_PER_YEAR / h_days) - 1) * 100


def percentile_matrix(dist, qs, transform=None):
    """(-> horizons_in_days, matrix[len(qs), len(horizons)]) of percentiles."""
    horizons = np.array(sorted(dist))
    rows = [
        np.percentile(transform(dist[h], h) if transform else dist[h], qs)
        for h in horizons
    ]
    return horizons, np.vstack(rows).T


def moments_frame(dist, transform=None):
    """Per-horizon mean / variance / skewness / excess kurtosis of the returns."""
    horizons = np.array(sorted(dist))
    out = {"mean": [], "variance": [], "skewness": [], "excess_kurtosis": []}
    for h in horizons:
        r = transform(dist[h], h) if transform else dist[h]
        out["mean"].append(r.mean())
        out["variance"].append(r.var(ddof=1))
        out["skewness"].append(skew(r, bias=False))
        out["excess_kurtosis"].append(kurtosis(r, fisher=True, bias=False))
    return horizons, {k: np.asarray(v) for k, v in out.items()}


def fan_chart(ax, x, pmat, edges=_EDGES):
    """Discrete fan chart: one solid shade per percentile band (colour every step)."""
    edges = np.asarray(edges)
    step = edges[1] - edges[0]
    for i in range(len(edges) - 1):
        center = 0.5 * (edges[i] + edges[i + 1])
        shade = 0.15 + 0.85 * (1 - abs(center - 50) / 50)
        ax.fill_between(x, pmat[i], pmat[i + 1], color=_BASE_CMAP(shade), linewidth=0)

    ax.plot(x, pmat[np.where(edges == 50)[0][0]], color="black", lw=1)

    handles = [Line2D([0], [0], color="black", lw=1, label="median")]
    for lo in edges[edges < 50][::-1]:
        shade = 0.15 + 0.85 * (1 - abs(lo + step / 2 - 50) / 50)
        handles.append(Patch(facecolor=_BASE_CMAP(shade), label=f"{lo}th-{100 - lo}th pct"))
    ax.legend(handles=handles, title="Percentile band", loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _fig(title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)
    return fig, ax


# --- the visualiser -----------------------------------------------------

def visualize(level, dist, summary=None, *, edges=config.PERCENTILE_EDGES,
              min_days=config.MIN_DAYS, annual_ylim=config.ANNUAL_YLIM,
              annualised=True, show_level=True,
              total_ylabel="Total return (%)", title_prefix="US market"):
    """Diagnostic plots for one holding-period distribution set.

    `annualised=False` (for an arithmetic quantity like extra wealth vs cash, which
    has no meaningful compound-annual form) drops the annualised fan and computes
    the moments / loss-probability panels on the raw window values. `show_level`
    draws the underlying index panel (skip it when there is no single level series).
    `summary` is accepted but unused -- panels are recomputed from `dist`.
    """
    edges = np.asarray(edges)
    qs = edges.astype(float)
    yr = config.DAYS_PER_YEAR
    all_h = np.array(sorted(dist))
    kept = (all_h >= min_days) if annualised else np.ones(len(all_h), bool)
    tf = annualize if annualised else None

    # 1 -- underlying level --------------------------------------------------
    if show_level and level is not None:
        fig, ax = _fig(f"{title_prefix}: underlying index level (log scale)")
        ax.plot(level.index, level.values, color="#1f4e79", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("date")
        ax.set_ylabel("index level")
        ax.grid(True, alpha=0.3)
        plt.show()

    # 2 -- total (window) fan ---------------------------------------------
    h, pm = percentile_matrix(dist, qs)
    fig, ax = _fig(f"{title_prefix}: window {'excess wealth' if not annualised else 'total return'} "
                   f"distribution by holding period")
    fan_chart(ax, h / yr, pm, edges=edges)
    ax.set_xlabel("Time horizon (years)")
    ax.set_ylabel(total_ylabel)
    plt.show()

    # 3 -- annualised-return fan ----------------------------------------
    if annualised:
        h, pm = percentile_matrix(dist, qs, transform=annualize)
        keep = h >= min_days
        fig, ax = _fig(f"{title_prefix}: annualised return distribution by holding period")
        fan_chart(ax, h[keep] / yr, pm[:, keep], edges=edges)
        ax.set_xlabel("Time horizon (years)")
        ax.set_ylabel("Annualised return (% p.a.)")
        if annual_ylim is not None:
            ax.set_ylim(*annual_ylim)
        plt.show()

    # 4 -- moments of the distribution ------------------------------------
    h, mom = moments_frame(dist, transform=tf)
    keep = h >= min_days if annualised else np.ones(len(h), bool)
    kind = "annualised-return" if annualised else "window"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, col in zip(axes.flat, ["mean", "variance", "skewness", "excess_kurtosis"]):
        ax.plot(h[keep] / yr, mom[col][keep], color="#1f4e79")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time horizon (years)")
    fig.suptitle(f"{title_prefix}: moments of the {kind} distribution")
    fig.tight_layout()
    plt.show()

    # 5 -- P(value < 0) --------------------------------------------------
    # sign is invariant under annualising, so use the raw window values
    share = np.array([100 * np.mean(dist[k] < 0) for k in all_h[kept]])
    lbl = "negative annualised return" if annualised else "shortfall vs the benchmark"
    fig, ax = _fig(f"{title_prefix}: probability of a {lbl}")
    ax.plot(all_h[kept] / yr, share, color="#8c1d1d")
    ax.set_xlabel("Time horizon (years)")
    ax.set_ylabel("P(< 0)  (%)")
    ax.grid(True, alpha=0.3)
    plt.show()


_LABEL = {
    "excess_wealth": "US market vs 1M T-bill (extra wealth per $1)",
    "real_market": "US market (real, CPI-deflated)",
    "real_tbill": "1M T-bill (real, CPI-deflated)",
}

# per-series visualize() options
_VIZ_OPTS = {
    "excess_wealth": dict(annualised=False, show_level=False,
                          total_ylabel="Extra wealth vs T-bill (% of stake)"),
    "real_market": dict(annualised=True, show_level=True),
    "real_tbill": dict(annualised=True, show_level=True),
}


def visualize_from_disk(name: str = "excess_wealth", method: str = "common", **kwargs):
    """Load data/processed/* and plot one series from `config.SERIES`."""
    import process_data

    opts = {**_VIZ_OPTS[name], **kwargs}
    level = process_data.levels()[name] if opts.get("show_level", True) else None
    dist, summary = process_data.load_processed(name, method)
    prefix = f"{_LABEL[name]} ({method} start dates)"
    visualize(level, dist, summary, title_prefix=prefix, **opts)


if __name__ == "__main__":
    for _s in config.SERIES:
        visualize_from_disk(_s, "common")
