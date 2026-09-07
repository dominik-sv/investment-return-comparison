"""Turn a raw daily level series into holding-period return distributions.

Two sweeps, both returning ``(dist, summary)``:

* ``holding_period_returns_common_start``   -- PRIMARY. Every horizon is measured
  from the SAME set of start dates (those leaving room for the longest horizon),
  so differences between horizons reflect the holding period alone.
* ``holding_period_returns_expanding_start`` -- SECONDARY. Every horizon uses as
  many start dates as it can; long horizons silently lose the most recent starts,
  so curves are not directly comparable across horizons.

``dist``    : {holding_period_days -> ndarray of buy-and-hold returns in %}
``summary`` : DataFrame indexed by holding_period_days (n, moments, percentiles)

``compute_and_save`` / ``load_processed`` persist everything under data/processed/,
one set of files per series name ("sp500", "tbill").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tqdm import tqdm

import config


# --- monthly series -> daily step series on the equity calendar ------------

def _step_to_calendar(monthly: pd.Series, calendar: pd.DatetimeIndex,
                      pre: float) -> pd.Series:
    """Monthly series held flat between observations, sampled on `calendar`.
    Values before the first observation are set to `pre`."""
    grid = monthly.index.union(calendar)
    daily = monthly.reindex(grid).ffill().fillna(pre)
    return daily.reindex(calendar).astype(float)


def tbill_daily_level(tbill_monthly, calendar) -> pd.Series:
    """Month-end T-bill wealth held flat between month-ends. On any day between
    rolls your wealth is what it was at the last roll -- no partial accrual.
    Slightly conservative; prices the monthly illiquidity of holding to maturity.
    Before the first roll: hold cash at par (1.0)."""
    return _step_to_calendar(tbill_monthly, calendar, pre=1.0).rename("level")


def cpi_daily_level(cpi_monthly, calendar) -> pd.Series:
    """Monthly CPI held flat between observations, on the equity trading calendar."""
    return _step_to_calendar(cpi_monthly, calendar, pre=cpi_monthly.iloc[0]).rename("cpi")


# --- core sweep -------------------------------------------------------------

def _summary_row(hp: int, r: np.ndarray) -> dict:
    p = np.percentile(r, config.PERCENTILES)
    return {
        "holding_period": hp,
        "n": len(r),
        "mean": r.mean(),
        "variance": r.var(ddof=1),
        "skewness": skew(r, bias=False),
        "excess_kurtosis": kurtosis(r, fisher=True, bias=False),
        **{f"p{q}": v for q, v in zip(config.PERCENTILES, p)},
    }


def _sweep(level: pd.Series, day_short: int, day_long: int, common_start: bool,
           bench: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    """Buy-and-hold window returns for every holding period.

    With ``bench`` (aligned to the same index), each value is the *extra wealth*
    per unit invested vs. holding the benchmark instead:
    ``(level[end]/level[start]) - (bench[end]/bench[start])``, in %. (The -1s of
    the two gross returns cancel.)
    """
    dates = level.index
    prices = level.to_numpy(float)
    bench_p = bench.to_numpy(float) if bench is not None else None
    n = len(dates)

    n_fixed = None
    if common_start:
        # start rows for which even the LONGEST horizon still lands inside the data
        ends_long = dates.searchsorted(dates + pd.Timedelta(days=day_long))
        n_fixed = int(np.searchsorted(ends_long, n, side="left"))
        if n_fixed == 0:
            raise ValueError("history too short for the requested longest horizon")

    dist: dict[int, np.ndarray] = {}
    rows = []
    for hp in tqdm(range(day_short, day_long + 1)):
        ends = dates.searchsorted(dates + pd.Timedelta(days=hp))
        k = n_fixed if common_start else int(np.searchsorted(ends, n, side="left"))
        if k == 0:
            break
        gross = prices[ends[:k]] / prices[:k]
        if bench_p is None:
            r = (gross - 1.0) * 100
        else:
            r = (gross - bench_p[ends[:k]] / bench_p[:k]) * 100
        dist[hp] = r
        rows.append(_summary_row(hp, r))

    summary = pd.DataFrame(rows).set_index("holding_period")
    return dist, summary


def holding_period_returns_common_start(level, day_short=config.DAY_SHORT,
                                        day_long=config.DAY_LONG, bench=None):
    return _sweep(level, day_short, day_long, True, bench=bench)


def holding_period_returns_expanding_start(level, day_short=config.DAY_SHORT,
                                           day_long=config.DAY_LONG, bench=None):
    return _sweep(level, day_short, day_long, False, bench=bench)


# --- persistence ---------------------------------------------------------

def _pack(dist: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ragged dict -> (horizons, offsets, concatenated float32 values)."""
    horizons = np.array(sorted(dist), dtype=np.int64)
    arrays = [np.asarray(dist[int(h)], dtype=np.float32) for h in horizons]
    offsets = np.concatenate([[0], np.cumsum([len(a) for a in arrays])]).astype(np.int64)
    values = np.concatenate(arrays) if arrays else np.zeros(0, np.float32)
    return horizons, offsets, values


def _unpack(horizons, offsets, values) -> dict:
    values = np.asarray(values, dtype=np.float64)   # stored float32, compute float64
    return {int(h): values[offsets[i]:offsets[i + 1]]
            for i, h in enumerate(horizons)}


def compute_and_save(level: pd.Series, name: str, bench: pd.Series | None = None):
    """Run both sweeps for one series, write data/processed/*_{name}_*, and
    return ``(dist_common, summary_common, dist_expanding, summary_expanding)``.
    With ``bench`` the sweep yields extra wealth per unit invested vs. that
    benchmark instead of a plain return.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dist_c, summary_c = holding_period_returns_common_start(level, bench=bench)
    dist_e, summary_e = holding_period_returns_expanding_start(level, bench=bench)

    summary_c.to_csv(config.summary_csv(name, "common"))
    summary_e.to_csv(config.summary_csv(name, "expanding"))

    ch, co, cv = _pack(dist_c)
    eh, eo, ev = _pack(dist_e)
    npz = config.distributions_npz(name)
    np.savez_compressed(
        npz,
        common_horizons=ch, common_offsets=co, common_values=cv,
        expanding_horizons=eh, expanding_offsets=eo, expanding_values=ev,
    )
    print(f"[{name}] wrote summary_{name}_{{common,expanding}}.csv, "
          f"{npz.name} ({npz.stat().st_size / 1e6:.0f} MB)")
    return dist_c, summary_c, dist_e, summary_e


def load_summary(name: str, method: str = "common") -> pd.DataFrame:
    return pd.read_csv(config.summary_csv(name, method), index_col="holding_period")


def load_distributions(name: str) -> tuple[dict, dict]:
    """Return ``(dist_common, dist_expanding)`` for one series."""
    z = np.load(config.distributions_npz(name))
    dist_c = _unpack(z["common_horizons"], z["common_offsets"], z["common_values"])
    dist_e = _unpack(z["expanding_horizons"], z["expanding_offsets"], z["expanding_values"])
    return dist_c, dist_e


def load_processed(name: str, method: str = "common") -> tuple[dict, pd.DataFrame]:
    """Convenience: ``(dist, summary)`` for one series + method."""
    dist_c, dist_e = load_distributions(name)
    dist = dist_c if method == "common" else dist_e
    return dist, load_summary(name, method)


# --- cli ----------------------------------------------------------------

def levels() -> dict:
    """All level series on the same (equity) trading calendar. `config.SERIES`
    picks which get swept/visualised; the others are building blocks."""
    import get_data

    mkt = get_data.load_raw()
    cal = mkt.index
    tb = tbill_daily_level(get_data.load_tbill_monthly(), cal)
    cpi = cpi_daily_level(get_data.load_cpi_monthly(), cal)
    return {
        "usmkt": mkt,
        "tbill": tb,
        "cpi": cpi,
        "real_market": (mkt / cpi).astype(float).rename("level"),
        "real_tbill": (tb / cpi).astype(float).rename("level"),
    }


# series name -> (level source in levels(), benchmark source or None).
# A benchmark makes the sweep report extra wealth per unit invested vs. it.
SERIES_INPUTS = {
    "excess_wealth": ("usmkt", "tbill"),
    "real_market": ("real_market", None),
    "real_tbill": ("real_tbill", None),
}


def main() -> None:
    lv = levels()
    for name in config.SERIES:
        lsrc, bsrc = SERIES_INPUTS[name]
        level = lv[lsrc]
        bench = lv[bsrc] if bsrc else None
        print(f"[{name}] {level.index.min().date()} -> {level.index.max().date()} "
              f"({len(level)} obs)" + (f"  vs {bsrc}" if bsrc else ""))
        compute_and_save(level, name, bench=bench)


if __name__ == "__main__":
    main()
