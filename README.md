# Investment return comparison

How the distribution of long-horizon returns changes with the holding period, seen
three ways: the equity risk premium, real equity returns, and real cash returns.

## Layout

| file | role |
|---|---|
| `config.py` | all settings (data sources, history window, sweep + plot parameters) |
| `get_data.py` | fetch raw series → `data/raw/` |
| `process_data.py` | holding-period return distributions → `data/processed/` |
| `visualize.py` | diagnostic plots from `data/raw/` + `data/processed/` |
| `run.ipynb` | thin driver: fetch → process → visualise |
| `data/raw/` | `usmkt.csv`, `tbill_1m.csv`, `cpi_us.csv` |
| `data/processed/` | `summary_{series}_{method}.csv` + `distributions_{series}.npz` (~1.4 GB) |

## The three views (`config.SERIES`)

| series | what it is | question |
|---|---|---|
| **`excess_wealth`** | extra wealth per \$1 from the market vs. holding the 1m T-bill over the same window: `G_market − G_tbill` (arithmetic, in % of the stake) | vs cash — how much more money did I end up with? |
| **`real_market`** | US market total return deflated by CPI | vs inflation — did my purchasing power grow? |
| **`real_tbill`** | 1-month T-bill deflated by CPI | what did the "safe" option cost in purchasing power? |

`excess_wealth` is an arithmetic difference of two compound window returns, so it
has no meaningful compound-annual form — its plots skip the annualised fan and use
the raw window values (see `visualize`). `real_market` / `real_tbill` are genuine
deflated wealth indices and get the full annualised treatment.

## Data sources (plain HTTPS downloads, no credentials)

* **US market** — CRSP value-weighted total return (`Mkt-RF + RF` from Ken French's
  daily 3-factor file), daily from **1926-07-01**. Broader than the S&P 500 but
  ~0.99 correlated (both cap-weighted; the S&P 500 already covers ~80–85% of market
  cap).
* **1-month T-bill** — Ken French `RF` (monthly). Compounded into a wealth index at
  month-ends and **held flat between month-ends** — no partial accrual, slightly
  conservative to price the monthly illiquidity.
* **US CPI** — CPI-U, all items, not seasonally adjusted (BLS `CUUR0000SA0`),
  monthly from 1913, via a community mirror (`config.CPI_URL`; FRED `CPIAUCNS` is
  the primary source if reachable). Held flat between monthly observations.

All monthly series are sampled onto the equity trading calendar so "invest every
day" means the same days.

## Usage

```python
import get_data, process_data, config, visualize

get_data.main()                                   # 1. fetch → data/raw/
lv = process_data.levels()
for name in config.SERIES:
    process_data.compute_and_save(lv[name], name) # 2. process the three views
visualize.visualize_from_disk("excess", "common") # 3. plot
```

## The two sweeps

* **common** (`holding_period_returns_common_start`) — every horizon measured from
  the *same* start dates (those leaving room for the longest horizon), so
  differences across horizons reflect the holding period alone.
* **expanding** (`holding_period_returns_expanding_start`) — every horizon uses all
  the starts it can; long horizons silently lose the most recent ones (more data
  per point, not cross-comparable).

## Plots (`visualize.visualize`)

1. underlying index level, log scale (`show_level`; skipped for `excess_wealth`)
2. window fan chart — percentile bands, colour every 10 pts
3. annualised-return fan chart, y-axis fixed to `config.ANNUAL_YLIM` (default −15…30% p.a.)
   — skipped when `annualised=False`
4. moments (mean / variance / skewness / excess kurtosis) of the window (or
   annualised) distribution vs holding period
5. P(value < 0) vs holding period — for `excess_wealth`, the probability the market
   ended below cash

## Website

`export_web.py` reads `data/processed/*.npz` and writes `docs/data/*.json` (~2.3 MB:
a fine percentile grid + `P(return < 0)` at daily resolution, plus binned histograms
at a coarser stride). `docs/` is served by GitHub Pages (Settings → Pages → Deploy
from a branch → `main` → `/docs`); it also holds `.nojekyll` and, later, the
`index.html` frontend. The site renders only those JSON files — no backend, and
`visualize.py` is unrelated to it.

## Notes

History runs from 1926-07; `config.START_DATE` / `DAY_LONG` control the window.
Windows overlap heavily, so the long-horizon tail still rests on relatively few
independent observations even with the full history.
