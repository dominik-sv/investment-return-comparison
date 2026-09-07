"""Central configuration for the investment-return-comparison project.

Every script (get_data, process_data, visualize) imports its settings from here
so there is a single source of truth for paths, the data sources and the
analysis parameters.
"""
from pathlib import Path

# --- paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_USMKT_CSV = RAW_DIR / "usmkt.csv"
RAW_TBILL_CSV = RAW_DIR / "tbill_1m.csv"
RAW_CPI_CSV = RAW_DIR / "cpi_us.csv"

# series swept + visualised by process_data / visualize
#   excess_wealth : extra wealth per $1 from the US market vs. holding the 1m T-bill
#                   over the same window  =  G_market - G_tbill  (arithmetic, in %)
#   real_market   : US market total return deflated by US CPI
#   real_tbill    : 1-month T-bill deflated by US CPI
# (usmkt / tbill / cpi level series are built internally to construct these)
SERIES = ("excess_wealth", "real_market", "real_tbill")


def summary_csv(name: str, method: str) -> Path:
    return PROCESSED_DIR / f"summary_{name}_{method}.csv"


def distributions_npz(name: str) -> Path:
    return PROCESSED_DIR / f"distributions_{name}.npz"


# --- data sources (Ken French data library, plain HTTPS download) -----------
# Daily 3-factor file -> Mkt-RF (market minus 1m T-bill) and RF, daily from 1926-07.
# Monthly 3-factor file -> RF, monthly, for the T-bill roll model.
FF_FACTORS_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FF_FACTORS_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)
# US CPI-U, all items, not seasonally adjusted (BLS CUUR0000SA0), monthly from 1913.
# Community mirror of the BLS series, auto-updated (datahub.io / datasets).
# Primary source if reachable: https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS
CPI_URL = "https://raw.githubusercontent.com/datasets/cpi-us/master/data/cpiai.csv"
MKT_COL = "MktWealth"
TBILL_COL = "TBillWealth"
CPI_COL = "CPI"

START_DATE = "1926-07-01"          # full available history
END_DATE = "2025-12-31"

# --- holding-period sweep ---------------------------------------------------
DAY_SHORT = 1                       # shortest holding period (calendar days)
DAY_LONG = 365 * 10                 # longest holding period (calendar days)
PERCENTILES = [5, 25, 50, 75, 95]   # summary percentiles stored per horizon

# --- visualisation --------------------------------------------------------
MIN_DAYS = 365                      # drop sub-1y horizons from annualised views
PERCENTILE_EDGES = list(range(0, 101, 10))  # fan-chart bands (colour every 10 pts)
ANNUAL_YLIM = (-15, 30)            # y-axis limits for the annualised-return fan (% p.a.)

DAYS_PER_YEAR = 365.25

# --- website export (export_web.py) ---------------------------------------
# GitHub Pages serves this folder ("Deploy from a branch" -> main -> /docs).
WEB_DIR = ROOT / "docs"
WEB_DATA_DIR = WEB_DIR / "data"
WEB_METHOD = "common"                     # which sweep to export
WEB_PCT_LEVELS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
WEB_HIST_BINS = 60
WEB_HIST_STRIDE_DAYS = 7                  # histogram horizons: every N days
# fixed histogram x-range (same for every horizon): the 1st..99th percentile of the
# distribution at WEB_HIST_RANGE_REF_YEARS. Mass outside is reported per side.
WEB_HIST_RANGE_PCT = [1, 99]
WEB_HIST_RANGE_REF_YEARS = 5.0
