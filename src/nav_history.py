"""
nav_history.py  –  Chunked reader and metric engine for the historical NAV
parquet file (mutual_fund_nav_history.parquet).

Data layout (21 M+ rows, 21 row groups)
----------------------------------------
  Index  : Scheme_Code  (int64)
  Columns: Date (datetime64[ns]), NAV (float64)

Computed metrics per scheme (reference date = 2024-12-31)
----------------------------------------------------------
  cagr_1yr    – Compound Annual Growth Rate over last 1 year
  cagr_3yr    – CAGR over last 3 years
  cagr_5yr    – CAGR over last 5 years
  vol_1yr     – Annualised volatility of daily log returns (last 252 tdays)
  sharpe_1yr  – Sharpe ratio  (cagr_1yr - 0.065) / vol_1yr
  sortino_1yr – Sortino ratio (cagr_1yr - 0.065) / downside_std
  max_drawdown– Maximum peak-to-trough decline (last 3 years)
  momentum_6m – 6-month rolling return
  nav_recency – Days since last NAV observation
  record_count– Total number of tradinxg days available

All results are cached to models/nav_metrics.csv so they are only
recomputed when the cache is missing or force_recompute=True.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODELS_DIR
from src.benchmark_features import load_nifty500_tri

# ─── paths ────────────────────────────────────────────────────────────────────
NAV_PARQUET_PATH = DATA_DIR  / "mutual_fund_nav_history.parquet"
NAV_METRICS_PATH = MODELS_DIR / "nav_metrics.csv"

# ─── constants ────────────────────────────────────────────────────────────────
# REFERENCE_DATE is resolved lazily on first use from the parquet max-date
# (falls back to pd.Timestamp.today() if the parquet file is unavailable).
RISK_FREE_RATE   = 0.065          # India 10-year G-sec proxy
TRADING_DAYS_YR  = 252

_NIFTY_TRI_CACHE: Optional[pd.Series] = None


def _get_nifty_tri_series() -> pd.Series:
    """Load and cache Nifty500 TRI indexed by Date for tracking metrics."""
    global _NIFTY_TRI_CACHE
    if _NIFTY_TRI_CACHE is not None:
        return _NIFTY_TRI_CACHE
    try:
        nifty = load_nifty500_tri()
        if nifty.empty:
            _NIFTY_TRI_CACHE = pd.Series(dtype=float)
        else:
            _NIFTY_TRI_CACHE = (
                nifty.set_index("Date")["TRI"]
                .sort_index()
                .dropna()
                .astype(float)
            )
    except Exception:
        _NIFTY_TRI_CACHE = pd.Series(dtype=float)
    return _NIFTY_TRI_CACHE


def _get_reference_date(parquet_path: Path = None) -> pd.Timestamp:
    """
    Return the latest NAV date available in the parquet file.
    Falls back to today's date if the file cannot be read.
    """
    try:
        if parquet_path is None:
            parquet_path = NAV_PARQUET_PATH
        pf  = pq.ParquetFile(str(parquet_path))
        # Read only the last row group to find the overall max date quickly
        dates_all = []
        for i in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(i, columns=["Date"]).to_pandas()
            rg["Date"] = pd.to_datetime(rg["Date"], errors="coerce")
            dates_all.append(rg["Date"].max())
        return max(dates_all)
    except Exception:
        return pd.Timestamp.today().normalize()


# Module-level reference date — resolved once on import from the actual data
REFERENCE_DATE: pd.Timestamp = None   # lazily resolved on first use (see _ref_date())


def _ref_date() -> pd.Timestamp:
    """Return (and cache) the reference date derived from the parquet max-date."""
    global REFERENCE_DATE
    if REFERENCE_DATE is None:
        REFERENCE_DATE = _get_reference_date()
        log.debug("[NavHistory] Reference date set to %s", REFERENCE_DATE.date())
    return REFERENCE_DATE

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  CHUNKED PARQUET READER
# ═══════════════════════════════════════════════════════════════════════════════

def iter_parquet_chunks(parquet_path: Path = NAV_PARQUET_PATH):
    """
    Yield one pandas DataFrame per row-group in the parquet file.
    Each chunk has Scheme_Code restored as a column (not index),
    and Date already cast to datetime.
    """
    pf = pq.ParquetFile(str(parquet_path))
    n  = pf.metadata.num_row_groups
    for i in range(n):
        df = pf.read_row_group(i).to_pandas().reset_index()   # Scheme_Code → col
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "NAV"])
        yield i, n, df


def read_full_nav(parquet_path: Path = NAV_PARQUET_PATH) -> pd.DataFrame:
    """
    Read the entire parquet in one call (uses pyarrow columnar IO).
    Returns a flat DataFrame with columns: Scheme_Code, Date, NAV.
    ~500 MB peak memory.  Use iter_parquet_chunks for low-memory paths.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    df    = table.to_pandas().reset_index()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "NAV"]).sort_values(["Scheme_Code", "Date"])


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  PER-SCHEME METRIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _nav_on_or_before(nav_series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    """Return NAV value closest to (but not after) `target` date."""
    sub = nav_series[nav_series.index <= target]
    return float(sub.iloc[-1]) if not sub.empty else None


def _cagr(nav_start: float, nav_end: float, years: float) -> float:
    """Safe CAGR; returns NaN for invalid input."""
    if nav_start is None or nav_end is None or nav_start <= 0 or years <= 0:
        return float("nan")
    return (nav_end / nav_start) ** (1.0 / years) - 1.0


def _compute_metrics_for_scheme(
    dates: pd.Series, navs: pd.Series, ref: pd.Timestamp = None
) -> dict:
    """
    Given a scheme's Date + NAV series (already sorted by date),
    return a dict of metrics keyed by metric name.
    """
    if ref is None:
        ref = _ref_date()
    # ── Set Date as index for fast .loc slicing ──
    ts = pd.Series(navs.values, index=pd.DatetimeIndex(dates.values), name="nav")
    ts = ts.sort_index()

    # last known NAV at / before reference date
    nav_ref = _nav_on_or_before(ts, ref)
    if nav_ref is None or nav_ref <= 0:
        return {}          # skip schemes with no data up to ref date

    # ── CAGR calculations ──
    nav_1y  = _nav_on_or_before(ts, ref - pd.DateOffset(years=1))
    nav_3y  = _nav_on_or_before(ts, ref - pd.DateOffset(years=3))
    nav_5y  = _nav_on_or_before(ts, ref - pd.DateOffset(years=5))

    cagr_1y = _cagr(nav_1y,  nav_ref, 1.0)
    cagr_3y = _cagr(nav_3y,  nav_ref, 3.0)
    cagr_5y = _cagr(nav_5y,  nav_ref, 5.0)

    # ── Volatility & Sharpe (last 252 trading days) ──
    window_1y = ts[ts.index >= ref - pd.DateOffset(days=400)]   # ~252 trading days
    if len(window_1y) >= 20:
        log_ret   = np.log(window_1y / window_1y.shift(1)).dropna()
        vol_1y    = float(log_ret.std() * math.sqrt(TRADING_DAYS_YR))
        vol_1y    = max(vol_1y, 0.002)   # floor: 0.2% avoids div-by-zero for stable debt funds
        neg_ret   = log_ret[log_ret < 0]
        downside  = float(neg_ret.std() * math.sqrt(TRADING_DAYS_YR)) if len(neg_ret) > 1 else 0.0
        downside  = max(downside, 0.002)
        sharpe_1y  = float(np.clip((cagr_1y - RISK_FREE_RATE) / vol_1y, -10.0, 10.0))
        sortino_1y = float(np.clip((cagr_1y - RISK_FREE_RATE) / downside, -10.0, 20.0))
    else:
        vol_1y = sharpe_1y = sortino_1y = float("nan")

    # ── Max Drawdown (last 3 years) ──
    window_3y = ts[ts.index >= ref - pd.DateOffset(years=3)]
    if len(window_3y) >= 2:
        roll_max = window_3y.cummax()
        drawdown = (window_3y - roll_max) / roll_max
        max_dd   = float(drawdown.min())           # negative number

        # B5: Recovery time — median trading days to recover from a 5%+ drawdown
        # A lower value means the fund bounces back quickly (resilience metric)
        trough_mask = drawdown < -0.05
        recovery_days_list = []
        in_drawdown = False
        trough_nav  = None
        for dt_idx, val in drawdown.items():
            if not in_drawdown and val < -0.05:
                in_drawdown = True
                trough_nav  = float(window_3y.loc[dt_idx])
                entry_date  = dt_idx
            elif in_drawdown and trough_nav is not None:
                current_nav = float(window_3y.loc[dt_idx])
                # recovered when NAV exceeds the pre-drawdown peak
                pre_peak = float(roll_max.loc[dt_idx])
                if current_nav >= pre_peak * 0.98:  # 2% tolerance
                    recovery_days_list.append((dt_idx - entry_date).days)
                    in_drawdown = False
                    trough_nav  = None
        recovery_time = float(np.median(recovery_days_list)) if recovery_days_list else float("nan")
    else:
        max_dd = float("nan")
        recovery_time = float("nan")

    # ── 6-month momentum ──
    nav_6m    = _nav_on_or_before(ts, ref - pd.DateOffset(months=6))
    momentum  = (nav_ref / nav_6m - 1.0) if (nav_6m and nav_6m > 0) else float("nan")

    # ── Consistency: % of rolling 1Y windows beating Nifty500 (last 5Y) ────
    # Falls back to positive-return consistency when aligned benchmark history
    # is unavailable for the lookback window.
    nifty_tri = _get_nifty_tri_series()
    window_5y = ts[ts.index >= ref - pd.DateOffset(years=5)]
    if len(window_5y) >= 260:
        roll_1y = window_5y / window_5y.shift(TRADING_DAYS_YR) - 1.0
        consistency_1y = float("nan")
        if not nifty_tri.empty:
            nifty_5y = nifty_tri[nifty_tri.index >= ref - pd.DateOffset(years=5)]
            nifty_roll_1y = nifty_5y / nifty_5y.shift(TRADING_DAYS_YR) - 1.0
            joined_roll = pd.concat(
                [roll_1y.rename("fund"), nifty_roll_1y.rename("nifty")], axis=1
            ).dropna()
            if not joined_roll.empty:
                consistency_1y = float((joined_roll["fund"] > joined_roll["nifty"]).mean())
        if np.isnan(consistency_1y):
            consistency_1y = float((roll_1y.dropna() > 0).mean()) if roll_1y.notna().any() else float("nan")
    else:
        consistency_1y = float("nan")

    # ── Tracking error vs Nifty500 TRI (last 1Y, annualized) ───────────────
    tracking_error_1y = float("nan")
    if len(window_1y) >= 20 and not nifty_tri.empty:
        fund_log_ret = np.log(window_1y / window_1y.shift(1)).dropna()
        nwin = nifty_tri[nifty_tri.index >= ref - pd.DateOffset(days=400)]
        nret = np.log(nwin / nwin.shift(1)).dropna()
        joined = pd.concat([fund_log_ret.rename("fund"), nret.rename("nifty")], axis=1).dropna()
        if len(joined) >= 20:
            active = joined["fund"] - joined["nifty"]
            tracking_error_1y = float(active.std() * math.sqrt(TRADING_DAYS_YR))

    # ── Recency & record count ──
    last_date    = ts.index.max()
    nav_recency  = (ref - last_date).days if last_date <= ref else 0
    record_count = len(ts)

    return {
        "cagr_1yr"     : cagr_1y,
        "cagr_3yr"     : cagr_3y,
        "cagr_5yr"     : cagr_5y,
        "vol_1yr"      : vol_1y,
        "sharpe_1yr"   : sharpe_1y,
        "sortino_1yr"  : sortino_1y,
        "max_drawdown" : max_dd,
        "recovery_time": recovery_time,   # B5: median days to recover from 5%+ drawdown
        "momentum_6m"  : momentum,
        "consistency_1y": consistency_1y,
        "tracking_error_1y": tracking_error_1y,
        "nav_recency"  : nav_recency,
        "record_count" : record_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  FULL METRIC COMPUTATION (CHUNKED)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    parquet_path: Path = NAV_PARQUET_PATH,
    ref_date    : pd.Timestamp = None,
    verbose     : bool = True,
) -> pd.DataFrame:
    """
    Iterate over all parquet row groups, accumulate per-scheme NAV histories,
    and compute the full metric set.

    ref_date defaults to the latest NAV date found in the parquet file itself.
    """
    if verbose:
        print("[NavHistory] Loading parquet in chunks …")

    # Resolve reference date from actual data max-date
    if ref_date is None:
        ref_date = _get_reference_date(parquet_path)
    if verbose:
        print(f"[NavHistory] Reference date (latest NAV in data): {ref_date.date()}")

    pf   = pq.ParquetFile(str(parquet_path))
    n_rg = pf.metadata.num_row_groups
    parts = []

    for i in range(n_rg):
        chunk = pf.read_row_group(i).to_pandas().reset_index()
        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce")
        chunk = chunk.dropna(subset=["Date", "NAV"])
        parts.append(chunk)
        if verbose:
            print(f"\r  chunk {i+1}/{n_rg}  ({len(chunk):,} rows)", end="", flush=True)

    if verbose:
        print()

    full = pd.concat(parts, ignore_index=True)
    del parts

    if verbose:
        print(f"[NavHistory] Total rows loaded: {len(full):,}")
        print(f"[NavHistory] Unique schemes   : {full['Scheme_Code'].nunique():,}")
        print(f"[NavHistory] Date range       : {full['Date'].min().date()} → {full['Date'].max().date()}")
        print("[NavHistory] Computing per-scheme metrics …")

    # ── Group and compute ──
    results = {}
    groups  = full.groupby("Scheme_Code")
    total   = len(groups)

    for idx, (code, grp) in enumerate(groups):
        metrics = _compute_metrics_for_scheme(grp["Date"], grp["NAV"], ref_date)
        if metrics:
            results[int(code)] = metrics
        if verbose and (idx % 2000 == 0):
            print(f"\r  processed {idx:,}/{total:,} schemes", end="", flush=True)

    if verbose:
        print(f"\r  processed {total:,}/{total:,} schemes — done.")

    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.index.name = "Scheme_Code"

    if verbose:
        print(f"[NavHistory] Metrics computed for {len(metrics_df):,} schemes.")
        _print_summary(metrics_df)

    return metrics_df


def _print_summary(df: pd.DataFrame) -> None:
    print("\n[NavHistory] === Metric Summary ===")
    for col in df.columns:
        s = df[col].dropna()
        if len(s):
            print(f"  {col:<15}: mean={s.mean():+.4f}  std={s.std():.4f}  "
                  f"min={s.min():+.4f}  max={s.max():+.4f}  non-null={len(s):,}")


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  CACHE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def load_nav_metrics(
    force_recompute: bool = False,
    verbose        : bool = True,
) -> pd.DataFrame:
    """
    Load cached nav_metrics.csv if it exists, else compute and cache.

    Parameters
    ----------
    force_recompute : bool  – ignore cache and recompute from parquet
    verbose         : bool  – print progress

    Returns
    -------
    DataFrame indexed by Scheme_Code with computed NAV metrics.
    """
    cache = Path(NAV_METRICS_PATH)

    if cache.exists() and not force_recompute:
        if verbose:
            print(f"[NavHistory] Loading cached metrics from {cache} …")
        df = pd.read_csv(cache, index_col="Scheme_Code")
        if verbose:
            print(f"[NavHistory] Loaded {len(df):,} scheme metrics from cache.")
        return df

    if not Path(NAV_PARQUET_PATH).exists():
        raise FileNotFoundError(
            f"Parquet file not found: {NAV_PARQUET_PATH}\n"
            "Download it from: https://github.com/InertExpert2911/Mutual_Fund_Data"
        )

    df = compute_all_metrics(verbose=verbose)
    df.to_csv(cache)
    if verbose:
        print(f"[NavHistory] Metrics cached → {cache}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  INSIGHTS HELPER  (used in notebooks)
# ═══════════════════════════════════════════════════════════════════════════════

def get_top_performers(
    metrics_df : pd.DataFrame,
    mf_df      : Optional[pd.DataFrame] = None,
    metric     : str = "sharpe_1yr",
    n          : int = 20,
    min_records: int = 100,
) -> pd.DataFrame:
    """
    Return top-N schemes by the chosen metric, optionally joined with mf_df names.
    """
    ranked = (
        metrics_df[metrics_df["record_count"] >= min_records]
        .dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(n)
        .copy()
    )
    if mf_df is not None:
        info = mf_df[["Scheme_Code", "Scheme_Name", "AMC", "Scheme_Category"]].copy()
        info["Scheme_Code"] = info["Scheme_Code"].astype(int)
        ranked = ranked.merge(info, left_index=True, right_on="Scheme_Code", how="left")
        ranked = ranked.set_index("Scheme_Code")
    return ranked


def get_category_stats(
    metrics_df: pd.DataFrame,
    mf_df     : pd.DataFrame,
) -> pd.DataFrame:
    """
    Return mean metrics per Scheme_Category (joined from mf_df).
    """
    info = mf_df[["Scheme_Code", "Scheme_Category"]].copy()
    info["Scheme_Code"] = info["Scheme_Code"].astype(int)
    merged = metrics_df.merge(info, left_index=True, right_on="Scheme_Code", how="inner")
    cat_stats = (
        merged.groupby("Scheme_Category")[
            ["cagr_1yr", "cagr_3yr", "cagr_5yr", "sharpe_1yr", "vol_1yr", "max_drawdown"]
        ]
        .mean()
        .sort_values("cagr_3yr", ascending=False)
    )
    return cat_stats


def nav_history_quick_stats(parquet_path: Path = NAV_PARQUET_PATH) -> dict:
    """
    Read only metadata + first row group to return quick dataset-level stats.
    Does NOT load the full file.
    """
    pf   = pq.ParquetFile(str(parquet_path))
    meta = pf.metadata
    rg0  = pf.read_row_group(0).to_pandas().reset_index()
    rg0["Date"] = pd.to_datetime(rg0["Date"], errors="coerce")

    return {
        "total_rows"       : meta.num_rows,
        "num_row_groups"   : meta.num_row_groups,
        "rows_per_group_approx": meta.num_rows // meta.num_row_groups,
        "columns"          : list(pf.schema_arrow.names),
        "sample_date_min"  : rg0["Date"].min(),
        "sample_date_max"  : rg0["Date"].max(),
        "sample_schemes"   : int(rg0["Scheme_Code"].nunique()),
        "file_path"        : str(parquet_path),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  MARKET REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_market_regime(
    parquet_path: Path = NAV_PARQUET_PATH,
    ref_date: Optional[pd.Timestamp] = None,
) -> str:
    """
    Classify the overall market regime at ref_date from NAV history.

    Method
    ------
    Sample the first 3 parquet row groups (~3 M rows, representative cross-section).
    Compute the median cross-scheme NAV return over [ref_date − 12m → ref_date].
    Classify:
      • > +15%  → 'bull'
      • <   0%  → 'bear'
      • otherwise → 'sideways'

    Returns
    -------
    'bull' | 'bear' | 'sideways'
    """
    if ref_date is None:
        ref_date = _ref_date()

    try:
        pf   = pq.ParquetFile(str(parquet_path))
        n_rg = pf.metadata.num_row_groups
        parts = []
        for i in range(min(3, n_rg)):
            chunk = pf.read_row_group(i).to_pandas().reset_index()
            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce")
            chunk = chunk.dropna(subset=["Date", "NAV"])
            parts.append(chunk)

        sample  = pd.concat(parts, ignore_index=True)
        ref_m12 = ref_date - pd.DateOffset(years=1)

        def _median_ret(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
            nav_start = (
                df[df["Date"] <= start].sort_values("Date")
                .groupby("Scheme_Code")["NAV"].last()
            )
            nav_end = (
                df[df["Date"] <= end].sort_values("Date")
                .groupby("Scheme_Code")["NAV"].last()
            )
            common = nav_start.index.intersection(nav_end.index)
            if common.empty:
                return float("nan")
            rets = (nav_end[common] / nav_start[common] - 1.0).dropna()
            return float(rets.median())

        ret = _median_ret(sample, ref_m12, ref_date)

        if np.isnan(ret):
            return "sideways"
        if ret > 0.15:
            return "bull"
        if ret < 0.00:
            return "bear"
        return "sideways"

    except Exception:
        return "sideways"


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  TWO-STEP DATA-DRIVEN FUND RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fund_risk_bands(
    mf_df: pd.DataFrame,
    nav_metrics: pd.DataFrame,
    w_vol: float = 0.50,
    w_dd:  float = 0.50,
    min_category_size: int = 5,
) -> pd.Series:
    """
    Assign each fund a realized risk tier using a two-step mechanism.

    Step 1 – Category-normalized risk score
    ----------------------------------------
    For every fund that has NAV history, compute:

        z_vol(i) = (vol_1yr_i  − vol_mean_cat) / vol_std_cat
        z_dd(i)  = (|dd_i|     − dd_mean_cat)  / dd_std_cat
        risk_score(i) = w_vol × z_vol(i)  +  w_dd × z_dd(i)

    Working within each Scheme_Category peer group removes the absolute
    level difference between debt and equity funds — a liquid fund in the
    top 20 % of *its* peers is "risky for a liquid fund", not "risky overall".

    Step 2 – Percentile bands within category
    ------------------------------------------
    Within each Scheme_Category, discretize risk_score at P20/P40/P60/P80:

        risk_score ≤ P20  →  Very_Low
        P20 < score ≤ P40 →  Low
        P40 < score ≤ P60 →  Medium
        P60 < score ≤ P80 →  High
        score > P80        →  Very_High

    Step 3 – SEBI-aligned floor/ceiling override
    ----------------------------------------------
    Apply CATEGORY_RISK_BOUNDS from config.py to clamp each band so that
    e.g. a "Very_Low" small-cap fund (uniform low-vol batch) is still at
    least High, and an "Overnight" fund never exceeds Very_Low.
    First keyword match in CATEGORY_RISK_BOUNDS wins (ordered dict).

    Parameters
    ----------
    mf_df             : fund catalogue DataFrame (Scheme_Code, Scheme_Category)
    nav_metrics       : per-scheme metrics from load_nav_metrics() —
                        indexed by Scheme_Code; must contain vol_1yr & max_drawdown
    w_vol             : weight for volatility z-score    (default 0.50)
    w_dd              : weight for max-drawdown z-score  (default 0.50)
    min_category_size : min funds with valid history required to use
                        percentile bands; smaller groups fall back to
                        keyword-based classification                  (default 5)

    Returns
    -------
    pd.Series indexed by mf_df.index with risk_tier strings.
    Funds with no NAV history AND no keyword match return pd.NA.
    """
    from config import CATEGORY_RISK_BOUNDS, RISK_CLASSES

    RISK_IDX = {r: i for i, r in enumerate(RISK_CLASSES)}
    PERC     = [20, 40, 60, 80]

    # ── Step 0: build a working frame aligned to mf_df's index ──────────────
    nm = nav_metrics[["vol_1yr", "max_drawdown"]].copy()
    nm.index = nm.index.astype(int)
    nm["abs_dd"] = nm["max_drawdown"].abs()          # positive = larger drawdown

    wf = mf_df[["Scheme_Code", "Scheme_Category"]].copy()
    wf["_code"] = pd.to_numeric(wf["Scheme_Code"], errors="coerce").astype("Int64")

    nm_vol = nm["vol_1yr"].to_dict()
    nm_dd  = nm["abs_dd"].to_dict()
    wf["vol_1yr"] = wf["_code"].map(nm_vol)
    wf["abs_dd"]  = wf["_code"].map(nm_dd)

    # ── Steps 1 & 2: per-category z-score → percentile band ─────────────────
    hist_tier = pd.Series(pd.NA, index=wf.index, dtype=object)

    for cat, grp in wf.groupby("Scheme_Category"):
        valid = grp[grp["vol_1yr"].notna() & grp["abs_dd"].notna()]
        if len(valid) < min_category_size:
            continue                                  # fall back to keyword tier

        vol = valid["vol_1yr"].astype(float)
        dd  = valid["abs_dd"].astype(float)

        def _z(s: pd.Series) -> pd.Series:
            """Safe z-score; returns 0 if std is near-zero (uniform category)."""
            sigma = s.std()
            return (s - s.mean()) / sigma if sigma > 1e-9 else pd.Series(0.0, index=s.index)

        risk_score = (w_vol * _z(vol) + w_dd * _z(dd)).reindex(valid.index)

        # percentile thresholds computed from THIS category's score distribution
        p20, p40, p60, p80 = np.percentile(risk_score.dropna().values, PERC)

        def _assign_band(s: float) -> str:
            if s <= p20: return RISK_CLASSES[0]
            if s <= p40: return RISK_CLASSES[1]
            if s <= p60: return RISK_CLASSES[2]
            if s <= p80: return RISK_CLASSES[3]
            return RISK_CLASSES[4]

        hist_tier.loc[valid.index] = risk_score.map(_assign_band)

    wf["hist_risk_tier"] = hist_tier

    # ── Step 3: SEBI floor/ceiling per category keyword ──────────────────────
    def _apply_bounds(row) -> object:
        tier = row["hist_risk_tier"]
        if pd.isna(tier):
            return pd.NA
        cat      = (row["Scheme_Category"] or "").lower()
        tier_idx = RISK_IDX.get(tier, 2)
        for kw, (floor_idx, ceil_idx) in CATEGORY_RISK_BOUNDS.items():
            if kw in cat:
                tier_idx = max(floor_idx, min(ceil_idx, tier_idx))
                break                                 # first match wins
        return RISK_CLASSES[tier_idx]

    final = wf.apply(_apply_bounds, axis=1)
    final.index = mf_df.index
    return final
