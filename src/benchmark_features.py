"""
benchmark_features.py  -  Real-data benchmark feature engine.

Combines:
1) CRISIL FundPerf Excel exports (Data/FundPerf/*.xlsx)
2) Nifty 500 TRI history CSVs     (Data/Nifty500/*.csv)

Outputs per-fund benchmark-aware features used by recommender/ensemble:
- riskometer_mapped_tier
- benchmarked_flag / benchmark_status
- fund return snapshots (1y/3y/5y)
- benchmark return snapshots (1y/3y/5y)
- excess return snapshots (fund - benchmark)
- nifty500_3y_cagr / nifty500_5y_cagr and excess-vs-nifty features
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import FUNDPERF_DIR, NIFTY500_DIR, FUND_BENCHMARK_FEATURES_CSV


_RISKOMETER_TO_TIER = {
    "low": "Very_Low",
    "low to moderate": "Low",
    "moderately low": "Low",
    "moderate": "Medium",
    "moderately high": "High",
    "high": "Very_High",
    "very high": "Very_High",
}

_EQUITY_BENCHMARK_KWS = [
    "equity",
    "large cap",
    "mid cap",
    "small cap",
    "multi cap",
    "flexi cap",
    "elss",
    "sectoral",
    "thematic",
    "index",
    "etf",
    "aggressive hybrid",
    "balanced advantage",
    "dynamic asset allocation",
    "equity savings",
]

_RET_FUND_COLS = ["fund_return_1y", "fund_return_3y", "fund_return_5y"]
_RET_BENCH_COLS = ["benchmark_return_1y", "benchmark_return_3y", "benchmark_return_5y"]


def _norm_name(s: object) -> str:
    text = str(s or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\s*[-–]\s*(regular|direct|growth|dividend|idcw|option|plan).*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _find_col(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for p in patterns:
        rgx = re.compile(p, flags=re.I)
        for low, orig in cols.items():
            if rgx.search(low):
                return orig
    return None


def _read_fundperf_file(path: Path) -> pd.DataFrame:
    # Header row is row 5 in the provided CRISIL exports.
    df = pd.read_excel(path, header=4)
    if "Scheme Name" not in df.columns:
        return pd.DataFrame()

    keep = [
        "Scheme Name",
        "Benchmark",
        "Riskometer Scheme",
        "Riskometer Benchmark",
        "NAV Date",
        "Daily AUM (Cr.)",
        "Return 1 Year (%) Regular",
        "Return 3 Year (%) Regular",
        "Return 5 Year (%) Regular",
        "Return 1 Year (%) Benchmark",
        "Return 3 Year (%) Benchmark",
        "Return 5 Year (%) Benchmark",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["_source_file"] = path.name
    return out


def load_fundperf_data(folder: Path = FUNDPERF_DIR) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.xlsx"))
    parts = []
    for f in files:
        try:
            p = _read_fundperf_file(f)
            if not p.empty:
                parts.append(p)
        except Exception:
            continue

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["Scheme Name"]).copy()
    df["scheme_name_key"] = df["Scheme Name"].apply(_norm_name)

    if "NAV Date" in df.columns:
        df["NAV Date"] = pd.to_datetime(df["NAV Date"], errors="coerce")
    else:
        df["NAV Date"] = pd.NaT

    for c in [
        "Daily AUM (Cr.)",
        "Return 1 Year (%) Regular",
        "Return 3 Year (%) Regular",
        "Return 5 Year (%) Regular",
        "Return 1 Year (%) Benchmark",
        "Return 3 Year (%) Benchmark",
        "Return 5 Year (%) Benchmark",
    ]:
        if c in df.columns:
            df[c] = _coerce_num(df[c])

    # keep latest NAV date per scheme key
    df = df.sort_values("NAV Date").groupby("scheme_name_key", as_index=False).last()

    risk_text = df.get("Riskometer Scheme", pd.Series([""] * len(df)))
    risk_norm = risk_text.astype(str).str.strip().str.lower()
    df["riskometer_mapped_tier"] = risk_norm.map(_RISKOMETER_TO_TIER)

    # standard output schema
    out = pd.DataFrame({
        "scheme_name_key": df["scheme_name_key"],
        "fundperf_scheme_name": df.get("Scheme Name"),
        "fundperf_benchmark_name": df.get("Benchmark"),
        "fundperf_riskometer_scheme": df.get("Riskometer Scheme"),
        "fundperf_riskometer_benchmark": df.get("Riskometer Benchmark"),
        "fundperf_nav_date": df.get("NAV Date"),
        "fundperf_daily_aum_cr": df.get("Daily AUM (Cr.)"),
        "fund_return_1y": df.get("Return 1 Year (%) Regular"),
        "fund_return_3y": df.get("Return 3 Year (%) Regular"),
        "fund_return_5y": df.get("Return 5 Year (%) Regular"),
        "benchmark_return_1y": df.get("Return 1 Year (%) Benchmark"),
        "benchmark_return_3y": df.get("Return 3 Year (%) Benchmark"),
        "benchmark_return_5y": df.get("Return 5 Year (%) Benchmark"),
        "riskometer_mapped_tier": df["riskometer_mapped_tier"],
    })

    for h in [1, 3, 5]:
        out[f"excess_return_{h}y"] = out[f"fund_return_{h}y"] - out[f"benchmark_return_{h}y"]

    # Forward-looking quality controls: recency and benchmark data completeness.
    for col in _RET_FUND_COLS + _RET_BENCH_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=-95.0, upper=300.0)

    nav_date = pd.to_datetime(out["fundperf_nav_date"], errors="coerce")
    ref_date = nav_date.max()
    if pd.notna(ref_date):
        out["fundperf_recency_days"] = (ref_date - nav_date).dt.days.clip(lower=0)
    else:
        out["fundperf_recency_days"] = np.nan

    benchmark_obs = out[_RET_BENCH_COLS].notna().sum(axis=1)
    fund_obs = out[_RET_FUND_COLS].notna().sum(axis=1)
    out["benchmark_data_quality"] = (benchmark_obs / len(_RET_BENCH_COLS)).astype(float)
    out["fundperf_data_quality"] = (fund_obs / len(_RET_FUND_COLS)).astype(float)

    return out


def load_nifty500_tri(folder: Path = NIFTY500_DIR) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"))
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        date_col = _find_col(df, [r"^date$"])
        tri_col = _find_col(df, [r"total returns index", r"tri"]) or _find_col(df, [r"close", r"index value"])
        if not date_col or not tri_col:
            continue

        part = pd.DataFrame({
            "Date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
            "TRI": _coerce_num(df[tri_col]),
        }).dropna(subset=["Date", "TRI"])
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["Date", "TRI", "ret"])

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date")
    out["ret"] = out["TRI"].pct_change()
    return out


def _calc_cagr(series: pd.Series, years: int) -> float:
    if len(series) < 2:
        return np.nan
    end_date = series.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    sub = series[series.index >= start_date]
    if len(sub) < 2:
        return np.nan
    v0 = float(sub.iloc[0])
    v1 = float(sub.iloc[-1])
    if v0 <= 0:
        return np.nan
    return (v1 / v0) ** (1.0 / years) - 1.0


def compute_nifty_benchmark_summary(nifty_df: pd.DataFrame) -> dict:
    if nifty_df.empty:
        return {"nifty500_3y_cagr": np.nan, "nifty500_5y_cagr": np.nan}

    tri = nifty_df.set_index("Date")["TRI"].sort_index()
    return {
        "nifty500_3y_cagr": _calc_cagr(tri, 3),
        "nifty500_5y_cagr": _calc_cagr(tri, 5),
    }


def build_fund_benchmark_features() -> pd.DataFrame:
    fp = load_fundperf_data()
    nifty = load_nifty500_tri()
    nifty_summary = compute_nifty_benchmark_summary(nifty)

    if fp.empty:
        return pd.DataFrame()

    fp["nifty500_3y_cagr"] = nifty_summary["nifty500_3y_cagr"]
    fp["nifty500_5y_cagr"] = nifty_summary["nifty500_5y_cagr"]
    fp["excess_vs_nifty_3y"] = (fp["fund_return_3y"] / 100.0) - fp["nifty500_3y_cagr"]
    fp["excess_vs_nifty_5y"] = (fp["fund_return_5y"] / 100.0) - fp["nifty500_5y_cagr"]

    fp.to_csv(FUND_BENCHMARK_FEATURES_CSV, index=False)
    return fp


def _is_equity_benchmark_candidate(cat: object, scheme_type: object) -> bool:
    c = str(cat or "").lower()
    st = str(scheme_type or "").lower()
    return ("open" in st) and any(k in c for k in _EQUITY_BENCHMARK_KWS)


def attach_benchmark_features(mf_df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Join FundPerf + Nifty500 features into the mutual-fund catalogue.
    """
    if benchmark_df is None:
        if Path(FUND_BENCHMARK_FEATURES_CSV).exists():
            benchmark_df = pd.read_csv(FUND_BENCHMARK_FEATURES_CSV)
        else:
            benchmark_df = build_fund_benchmark_features()

    out = mf_df.copy()
    out["scheme_name_key"] = out["Scheme_Name"].apply(_norm_name)

    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_df = benchmark_df.drop_duplicates(subset=["scheme_name_key"], keep="last")
        out = out.merge(benchmark_df, on="scheme_name_key", how="left")

    out["benchmark_data_quality"] = pd.to_numeric(
        out.get("benchmark_data_quality", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["fundperf_data_quality"] = pd.to_numeric(
        out.get("fundperf_data_quality", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["fundperf_recency_days"] = pd.to_numeric(
        out.get("fundperf_recency_days", pd.Series(np.nan, index=out.index)), errors="coerce"
    ).fillna(365.0).clip(lower=0.0)

    eligible = out.apply(
        lambda r: 1 if _is_equity_benchmark_candidate(r.get("Scheme_Category"), r.get("Scheme_Type")) else 0,
        axis=1,
    ).astype(int)
    has_quality_data = (out["benchmark_data_quality"] >= 0.34).astype(int)

    out["benchmark_eligible_flag"] = eligible
    out["benchmarked_flag"] = (eligible * has_quality_data).astype(int)
    out["benchmark_status"] = np.select(
        [out["benchmarked_flag"] == 1, out["benchmark_eligible_flag"] == 1],
        ["Benchmarked_With_Data", "Eligible_No_Data"],
        default="Not_Eligible",
    )

    # Riskometer-based tier available from FundPerf should override keyword tier when present.
    if "riskometer_mapped_tier" in out.columns:
        mask = out["riskometer_mapped_tier"].isin(["Very_Low", "Low", "Medium", "High", "Very_High"])
        out.loc[mask, "risk_tier"] = out.loc[mask, "riskometer_mapped_tier"]
        out.loc[mask, "risk_tier_source"] = "fundperf_riskometer"

    out.drop(columns=["scheme_name_key"], inplace=True)
    return out
