"""Fetch the raw series and store them under data/raw/.

Both series come from the Ken French data library (plain HTTPS download, no
credentials):

* US market total return  -> daily 3-factor file, Mkt-RF + RF
* 1-month T-bill           -> monthly 3-factor file, RF

    python get_data.py            # writes data/raw/usmkt.csv and tbill_1m.csv

The `load_*` helpers only read the stored files and have no network dependency,
so the rest of the pipeline runs offline once the data is downloaded.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

import config


# --- Fama-French factor download ------------------------------------------

def _fetch_ff_factors(url: str) -> pd.DataFrame:
    """Download a Fama-French factors zip and return the factor block as a
    DataFrame indexed by date, columns Mkt-RF / SMB / HML / RF (percent per
    period). Handles both the daily (YYYYMMDD) and monthly (YYYYMM) files.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    lines = zf.read(zf.namelist()[0]).decode("latin-1").splitlines()

    start = next(i for i, l in enumerate(lines)
                 if l.replace(" ", "").startswith(",Mkt-RF"))
    recs = []
    for line in lines[start + 1:]:
        tok = [x.strip() for x in line.split(",")]
        if len(tok) == 5 and tok[0].isdigit() and len(tok[0]) in (6, 8):
            recs.append(tok)
        elif recs:
            break                                   # blank line / annual block / footer

    df = pd.DataFrame(recs, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    if len(df["date"].iloc[0]) == 8:
        idx = pd.to_datetime(df["date"], format="%Y%m%d")
    else:
        idx = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    out = df[["Mkt-RF", "SMB", "HML", "RF"]].astype(float)
    out.index = idx
    return out


def _clip(s: pd.Series) -> pd.Series:
    if config.START_DATE:
        s = s[s.index >= config.START_DATE]
    if config.END_DATE:
        s = s[s.index <= config.END_DATE]
    return s


# --- the two series -----------------------------------------------------

def fetch_usmkt_daily() -> pd.DataFrame:
    """US market total-return wealth index (CRSP value-weighted), daily."""
    ff = _fetch_ff_factors(config.FF_FACTORS_DAILY_URL)
    total = (ff["Mkt-RF"] + ff["RF"]) / 100.0          # daily total market return
    wealth = _clip((1.0 + total).cumprod()).rename(config.MKT_COL)
    return wealth.to_frame().rename_axis("Date")


def fetch_tbill_1m() -> pd.DataFrame:
    """1-month T-bill wealth index from Ken French 'RF' (monthly). Compounded at
    month-ends; callers hold it flat between month-ends (no partial accrual)."""
    ff = _fetch_ff_factors(config.FF_FACTORS_MONTHLY_URL)
    wealth = (1.0 + ff["RF"] / 100.0).cumprod().rename(config.TBILL_COL)
    return wealth.to_frame().rename_axis("Date")


def fetch_cpi_us() -> pd.DataFrame:
    """US CPI-U, all items, not seasonally adjusted, monthly (BLS CUUR0000SA0)."""
    req = urllib.request.Request(config.CPI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text))          # columns: Date, Index, Inflation
    out = pd.DataFrame({config.CPI_COL: pd.to_numeric(df["Index"], errors="coerce")})
    out.index = pd.to_datetime(df["Date"])
    out.index.name = "Date"
    return out.dropna()


# --- io helpers -----------------------------------------------------------

def save_raw(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_raw(path=config.RAW_USMKT_CSV) -> pd.Series:
    """Read a stored daily level file as a single positive float Series 'level'."""
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    level = raw.iloc[:, 0].astype(float).rename("level")
    level.index.name = "date"
    level = level[level.notna() & (level > 0)].sort_index()
    return level


def load_tbill_monthly(path=config.RAW_TBILL_CSV) -> pd.Series:
    """Month-end T-bill wealth series."""
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s.index.name = "date"
    return s.rename("tbill_wealth").sort_index()


def load_cpi_monthly(path=config.RAW_CPI_CSV) -> pd.Series:
    """Monthly US CPI index."""
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index.name = "date"
    return s.rename("cpi").sort_index()


# --- cli ----------------------------------------------------------------

def main() -> None:
    mkt = fetch_usmkt_daily()
    save_raw(mkt, config.RAW_USMKT_CSV)
    print(f"saved {len(mkt)} rows -> {config.RAW_USMKT_CSV} "
          f"({mkt.index.min().date()} -> {mkt.index.max().date()})")

    tb = fetch_tbill_1m()
    save_raw(tb, config.RAW_TBILL_CSV)
    print(f"saved {len(tb)} rows -> {config.RAW_TBILL_CSV} "
          f"({tb.index.min().date()} -> {tb.index.max().date()})")

    cpi = fetch_cpi_us()
    save_raw(cpi, config.RAW_CPI_CSV)
    print(f"saved {len(cpi)} rows -> {config.RAW_CPI_CSV} "
          f"({cpi.index.min().date()} -> {cpi.index.max().date()})")


if __name__ == "__main__":
    main()
