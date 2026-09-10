"""Export compact JSON for the static website from data/processed/*.npz.

Reads the full return distributions and writes docs/data/*.json (~2-3 MB total):

* per series -- a fine percentile grid + P(window return < 0) + sample size, at
  daily horizon resolution;
* per series -- binned histograms at a coarser horizon stride (config.WEB_*).

No raw arrays leave this step; the website only ever sees these JSON files.

    python export_web.py            # after process_data.py has run
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

import config
import process_data

_LABEL = {
    "excess_wealth": "US market vs 1M T-bill — extra wealth per $1",
    "real_market": "US market — real (CPI-deflated) total return",
    "real_tbill": "1M T-bill — real (CPI-deflated) total return",
}
_UNIT = {
    "excess_wealth": "% of stake",
    "real_market": "%",
    "real_tbill": "%",
}
_VIEWS = {
    "excess": {
        "label": "Market excess of the risk-free rate",
        "series": ["excess_wealth"],
    },
    "real": {
        "label": "Real market vs real risk-free rate",
        "series": ["real_market", "real_tbill"],
    },
}


def _r(a, nd=3):
    """Round to a list of plain floats, NaN/inf -> None (JSON null)."""
    return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(a)]


def _series_payload(name: str, method: str) -> dict:
    dist, _ = process_data.load_processed(name, method)
    horizons = np.array(sorted(dist))
    levels = config.WEB_PCT_LEVELS

    pmat = np.vstack([np.percentile(dist[int(h)], levels) for h in horizons])   # (H, L)
    p_below_zero = np.array([100.0 * np.mean(dist[int(h)] < 0) for h in horizons])
    n = np.array([dist[int(h)].size for h in horizons])

    # histograms at a coarser stride (first + last horizon always included)
    stride = max(1, config.WEB_HIST_STRIDE_DAYS)
    idx = list(range(0, len(horizons), stride))
    if idx[-1] != len(horizons) - 1:
        idx.append(len(horizons) - 1)
    hh = horizons[idx]

    # ONE fixed bin grid for every horizon: 1st..99th pct at the reference horizon
    ref_days = config.WEB_HIST_RANGE_REF_YEARS * config.DAYS_PER_YEAR
    ref_h = int(horizons[np.argmin(np.abs(horizons - ref_days))])
    lo, hi = (float(x) for x in np.percentile(dist[ref_h], config.WEB_HIST_RANGE_PCT))
    edges = np.linspace(lo, hi, config.WEB_HIST_BINS + 1)

    def annualise(r, days):
        base = np.clip(1.0 + np.asarray(r) / 100.0, 1e-9, None)
        return (base ** (365.25 / days) - 1.0) * 100.0

    counts, below, above, hmean, hstd, hmed, hp10, hp90, hn = ([] for _ in range(9))
    hp1, hp99 = [], []
    ha_mean, ha_std, ha_p10, ha_p90, ha_p1, ha_p99 = ([] for _ in range(6))
    for h in hh:
        r = dist[int(h)]
        c, _ = np.histogram(r, bins=edges)
        counts.append([int(x) for x in c])
        below.append(100.0 * float(np.mean(r < lo)))
        above.append(100.0 * float(np.mean(r > hi)))
        p1, p10, p90, p99 = np.percentile(r, [1, 10, 90, 99])
        hmean.append(float(r.mean()))
        hstd.append(float(r.std(ddof=1)))
        hmed.append(float(np.median(r)))
        hp1.append(float(p1)); hp10.append(float(p10))
        hp90.append(float(p90)); hp99.append(float(p99))
        hn.append(int(r.size))

        a = annualise(r, int(h))                 # annualised-return distribution
        a1, a10, a90, a99 = np.percentile(a, [1, 10, 90, 99])
        ha_mean.append(float(a.mean())); ha_std.append(float(a.std(ddof=1)))
        ha_p1.append(float(a1)); ha_p10.append(float(a10))
        ha_p90.append(float(a90)); ha_p99.append(float(a99))

    return {
        "name": name,
        "label": _LABEL[name],
        "unit": _UNIT[name],
        "horizons_days": [int(x) for x in horizons],
        "horizons_years": _r(horizons / config.DAYS_PER_YEAR, 4),
        "n": [int(x) for x in n],
        "pct_levels": levels,
        "percentiles": {str(levels[j]): _r(pmat[:, j], 3) for j in range(len(levels))},
        "p_below_zero": _r(p_below_zero, 2),
        "hist": {
            "horizons_days": [int(x) for x in hh],
            "horizons_years": _r(hh / config.DAYS_PER_YEAR, 4),
            "n": hn,
            "range": _r([lo, hi], 3),
            "edges": _r(edges, 3),                   # shared across all horizons
            "counts": counts,
            "below_pct": _r(below, 2),               # mass left of the fixed range
            "above_pct": _r(above, 2),               # mass right of the fixed range
            "mean": _r(hmean, 3),
            "std": _r(hstd, 3),
            "median": _r(hmed, 3),
            "p1": _r(hp1, 3),
            "p10": _r(hp10, 3),
            "p90": _r(hp90, 3),
            "p99": _r(hp99, 3),
            "a_mean": _r(ha_mean, 3),
            "a_std": _r(ha_std, 3),
            "a_p1": _r(ha_p1, 3),
            "a_p10": _r(ha_p10, 3),
            "a_p90": _r(ha_p90, 3),
            "a_p99": _r(ha_p99, 3),
        },
    }


def main() -> None:
    config.WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    method = config.WEB_METHOD

    for name in config.SERIES:
        payload = _series_payload(name, method)
        path = config.WEB_DATA_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {path.name:24s} {path.stat().st_size / 1e3:6.0f} KB  "
              f"({len(payload['horizons_days'])} horizons, "
              f"{len(payload['hist']['horizons_days'])} hist)")

    meta = {
        "generated": dt.date.today().isoformat(),
        "method": method,
        "pct_levels": config.WEB_PCT_LEVELS,
        "series_labels": _LABEL,
        "units": _UNIT,
        "views": _VIEWS,
    }
    meta_path = config.WEB_DATA_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path.name}")


if __name__ == "__main__":
    main()
