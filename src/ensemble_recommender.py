"""
ensemble_recommender.py  –  Fund Selection via XGBoost + Random Forest + LightGBM Ensemble.

Reference: §7.3.2 — Fund Selection via Ensemble Modelling — Fund Selection Ensemble Modelling
    "We propose to use XGBoost, Random Forest, and LightGBM in the ensemble
     modeling. XGBoost provides accuracy through gradient boosting, Random
     Forest provides variance reduction through bagging, and LightGBM adds
     leaf-wise growth for improved generalisation."

Architecture (v2)
-----------------
1.  Feature engineering from fund catalogue metadata (AUM, NAV, fund age,
    recency, category, AMC size, scheme type).
2.  Forward-looking target when panel labels are available; otherwise a
    rule-based fallback quality target for low-information funds.
3.  XGBRegressor  — gradient boosting (accuracy / efficiency).
4.  RandomForestRegressor — bagging trees (variance reduction / robustness).
5.  LGBMRegressor — leaf-wise growth (faster training / better generalisation).
6.  Soft ensemble: final score = 0.40 × XGB + 0.35 × RF + 0.25 × LGBM.
7.  Saved artefacts: rf_fund_model.joblib, xgb_fund_model.joblib,
                     lgbm_fund_model.joblib, fund_feature_cols.joblib,
                     ensemble_meta.json

Usage
-----
    from src.ensemble_recommender import fit_fund_ensemble, score_funds_ensemble

    # Training (done once in train.py):
    metrics = fit_fund_ensemble(mf_df)

    # Inference (in recommender.py):
    mf_df["ensemble_score"] = score_funds_ensemble(mf_df)
"""

from __future__ import annotations

import json
import re
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge  # Meta-learner for stacking
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pyarrow.parquet as pq

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn(
        "xgboost not installed.  Install with: conda install xgboost\n"
        "Falling back to GradientBoostingRegressor from sklearn.",
        stacklevel=2,
    )
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    warnings.warn(
        "lightgbm not installed.  Install with: pip install lightgbm\n"
        "Ensemble will use XGBoost + RF only (2-model fallback).",
        stacklevel=2,
    )

try:
    from catboost import CatBoostRegressor
    _CAT_AVAILABLE = True
except ImportError:
    _CAT_AVAILABLE = False
    warnings.warn(
        "catboost not installed.  Install with: pip install catboost\n"
        "Ensemble will skip CatBoost (optional enhancement).",
        stacklevel=2,
    )

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, RANDOM_SEED, RISK_CLASSES, NAV_HISTORY_PARQUET
from src.benchmark_features import load_nifty500_tri

# ─── Artefact paths ───────────────────────────────────────────────────────────
RF_FUND_MODEL_PATH   = MODELS_DIR / "rf_fund_model.joblib"
XGB_FUND_MODEL_PATH  = MODELS_DIR / "xgb_fund_model.joblib"
LGBM_FUND_MODEL_PATH = MODELS_DIR / "lgbm_fund_model.joblib"
FUND_FEAT_COLS_PATH  = MODELS_DIR / "fund_feature_cols.joblib"
ENSEMBLE_META_PATH   = MODELS_DIR / "ensemble_fund_meta.json"

RF_FUND_MODEL_EQ_PATH   = MODELS_DIR / "rf_fund_model_equity.joblib"
XGB_FUND_MODEL_EQ_PATH  = MODELS_DIR / "xgb_fund_model_equity.joblib"
LGBM_FUND_MODEL_EQ_PATH = MODELS_DIR / "lgbm_fund_model_equity.joblib"
RF_FUND_MODEL_NEQ_PATH   = MODELS_DIR / "rf_fund_model_nonequity.joblib"
XGB_FUND_MODEL_NEQ_PATH  = MODELS_DIR / "xgb_fund_model_nonequity.joblib"
LGBM_FUND_MODEL_NEQ_PATH = MODELS_DIR / "lgbm_fund_model_nonequity.joblib"

# ─── Enhanced models (optional) ───────────────────────────────────────────────
CAT_FUND_MODEL_PATH  = MODELS_DIR / "cat_fund_model.joblib"
ET_FUND_MODEL_PATH   = MODELS_DIR / "et_fund_model.joblib"
META_RIDGE_MODEL_PATH = MODELS_DIR / "meta_ridge_stacking.joblib"  # Ridge meta-learner

# ─── Feature engineering ──────────────────────────────────────────────────────

# Reference date: latest NAV available (resolved lazily from parquet)
# Falls back to today if parquet not yet loaded.
def _get_ref_date() -> pd.Timestamp:
    """Return the latest NAV date from the parquet, or today as fallback."""
    try:
        from src.nav_history import _ref_date
        return _ref_date()
    except Exception:
        return pd.Timestamp.today().normalize()


# Top AMC labels retained as one-hot (rest → "Other")
_TOP_AMC_COUNT = 15

# Historical metrics columns sourced from nav_history.load_nav_metrics().
# tracking_error_1y removed (Pearson r=0.861 with vol_1yr — redundant).
_HIST_FEATURE_COLS = [
    "cagr_1yr",
    "cagr_3yr",
    "cagr_5yr",
    "vol_1yr",
    "sharpe_1yr",
    "max_drawdown",
    "recovery_time",
    "momentum_6m",
    "consistency_1y",
]

_FUNDPERF_FEAT_COLS = [
    "fund_return_1y",
    "fund_return_3y",
    "fund_return_5y",
    "benchmark_return_1y",
    "benchmark_return_3y",
    "benchmark_return_5y",
    "excess_return_1y",
    "excess_return_3y",
    "excess_return_5y",
    "excess_vs_nifty_3y",
    "excess_vs_nifty_5y",
    "fundperf_daily_aum_cr",
    "fundperf_recency_days",
    "benchmark_data_quality",
    "fundperf_data_quality",
    "benchmark_eligible_flag",
    "benchmark_missing_flag",
]

_BENCH_DERIVED_COLS = [
    "benchmark_return_1y",
    "benchmark_return_3y",
    "benchmark_return_5y",
    "excess_return_1y",
    "excess_return_3y",
    "excess_return_5y",
    "excess_vs_nifty_3y",
    "excess_vs_nifty_5y",
]


def _parse_date_col(series: pd.Series) -> pd.Series:
    """Coerce a column to datetime, returning NaT for failures."""
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def build_fund_features(
    df: pd.DataFrame,
    nav_metrics: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derive a numeric feature matrix from the mutual fund catalogue.

    Parameters
    ----------
    df          : fund catalogue DataFrame (from load_mutual_funds)
    nav_metrics : optional per-scheme historical metrics from nav_history.py
                  (columns: cagr_1yr, cagr_3yr, cagr_5yr, vol_1yr,
                   sharpe_1yr, max_drawdown, momentum_6m)
                  If supplied, these are merged on Scheme_Code and appended
                  as additional features — making the ensemble model use real
                  return-history rather than only catalogue metadata.

    Returns
    -------
    X_df        : DataFrame of numeric features (one row per fund)
    feat_cols   : ordered list of column names in X_df
    """
    d = df.copy()
    d_raw = df.copy()

    # ── Numeric base features ──────────────────────────────────────────────
    d["log_aum"] = np.log1p(pd.to_numeric(d.get("Average_AUM_Cr", 0), errors="coerce").fillna(0))
    d["nav_log"] = np.log1p(pd.to_numeric(d.get("NAV", 0), errors="coerce").fillna(0))
    d["min_inv_log"] = np.log1p(
        pd.to_numeric(d.get("Scheme_Min_Amt", 500), errors="coerce").fillna(500)
    )

    # ── Date-derived features ──────────────────────────────────────────────
    launch     = _parse_date_col(d.get("Launch_Date", pd.NaT))
    latest_nav = _parse_date_col(d.get("Latest_NAV_Date", pd.NaT))

    _ref = _get_ref_date()
    d["fund_age_years"]   = (((_ref - launch).dt.days) / 365.25).clip(lower=0).fillna(0)
    d["nav_recency_days"] = ((_ref - latest_nav).dt.days).clip(lower=0).fillna(365)

    # ── Boolean / categorical flags ────────────────────────────────────────
    scheme_type = d.get("Scheme_Type", pd.Series([""] * len(d), index=d.index)).fillna("")
    d["is_open_ended"] = scheme_type.str.contains("Open", case=False, na=False).astype(int)
    d["is_active"]     = (d.get("Closure_Date", pd.Series([np.nan] * len(d), index=d.index))
                           .isna()).astype(int)
    nav_name = d.get("Scheme_NAV_Name", pd.Series([""] * len(d), index=d.index)).fillna("")
    d["is_growth"]     = nav_name.str.contains("Growth", case=False, na=False).astype(int)

    # ── Risk tier ordinal (0–4) ────────────────────────────────────────────
    tier_map = {r: i for i, r in enumerate(RISK_CLASSES)}
    d["risk_tier_ord"] = d.get("risk_tier", pd.Series([np.nan] * len(d), index=d.index)).map(tier_map).fillna(2)

    # ── AMC frequency encoding (large AMCs → higher weight) ───────────────
    amc_series = d.get("AMC", pd.Series(["Other"] * len(d), index=d.index)).fillna("Other")
    top_amcs   = amc_series.value_counts().head(_TOP_AMC_COUNT).index.tolist()
    d["amc_is_top"] = amc_series.isin(top_amcs).astype(int)
    # Rank of AMC by fund count (normalised)
    amc_counts   = amc_series.map(amc_series.value_counts()).fillna(1)
    d["amc_size_norm"] = amc_counts / amc_counts.max()

    # ── Expense ratio: use real TER when available, otherwise median-impute ──
    # No synthetic TER is generated anywhere in the scoring pipeline.
    expense_ratio_real = pd.to_numeric(
        d.get("expense_ratio_real", pd.Series(np.nan, index=d.index)), errors="coerce"
    )
    if "expense_ratio" in d.columns:
        expense_ratio_model = pd.to_numeric(d["expense_ratio"], errors="coerce")
    else:
        if "Scheme_Category" in d.columns:
            cat_median = expense_ratio_real.groupby(d["Scheme_Category"]).transform("median")
        else:
            cat_median = pd.Series(np.nan, index=d.index)
        global_median = expense_ratio_real.median()
        if pd.isna(global_median):
            global_median = 1.0
        expense_ratio_model = expense_ratio_real.fillna(cat_median).fillna(global_median)
        d["expense_ratio"] = expense_ratio_model.round(2)
    d["expense_ratio_norm"] = (expense_ratio_model / 2.50).clip(0.0, 1.0).fillna(0.0).astype(np.float32)

    # TER missingness is modeled explicitly so imputed TER contributes less than real TER.
    if "ter_missing_flag" in d.columns:
        d["ter_missing_flag"] = pd.to_numeric(d["ter_missing_flag"], errors="coerce").fillna(1).astype(np.float32)
    else:
        d["ter_missing_flag"] = expense_ratio_real.isna().astype(np.float32)

    if "expense_ratio_real" in d.columns:
        d["expense_ratio_real_norm"] = (
            expense_ratio_real / 2.50
        ).clip(0.0, 1.0).fillna(0.0).astype(np.float32)
    else:
        d["expense_ratio_real_norm"] = 0.0

    # ── Numeric feature list (deterministic order) ─────────────────────────
    # IMPROVEMENT: Remove redundant benchmark flags and low-quality data_quality fields
    base_feat_cols = [
        "log_aum",
        "nav_log",
        "min_inv_log",
        "fund_age_years",
        "nav_recency_days",
        "is_open_ended",
        "is_active",
        "is_growth",
        "risk_tier_ord",
        "amc_is_top",
        "amc_size_norm",
        "expense_ratio_norm",       # Keep normalized proxy (helps with universality)
        "expense_ratio_real_norm",  # Both are useful despite r=0.944
        "ter_missing_flag",
        # Removed: has_nav_history (duplicate), benchmark_missing_flag (r=1.0 with benchmarked),
        #         fundperf_data_quality, benchmark_data_quality (high redundancy r>0.96)
        "has_fundperf_returns",
        "data_quality_flag",
        "regime_beta",  # ENHANCED: Regime beta (systematic risk exposure)
    ]

    # ── Merge historical return metrics if provided ─────────────────────────
    hist_feat_cols: List[str] = []
    has_nav_history = pd.Series(0.0, index=d.index, dtype=np.float32)
    if nav_metrics is not None and len(nav_metrics) > 0:
        nm = nav_metrics.reindex(columns=[c for c in _HIST_FEATURE_COLS if c in nav_metrics.columns]).copy()
        for c in _HIST_FEATURE_COLS:
            if c not in nm.columns:
                nm[c] = np.nan
        nm = nm[_HIST_FEATURE_COLS]
        nm.index = nm.index.astype(int)
        nm.columns = [f"{c}_hist" for c in nm.columns]   # avoid name clashes
        hist_feat_cols = list(nm.columns)

        # align on original DataFrame index via Scheme_Code
        scheme_codes = pd.to_numeric(
            d.get("Scheme_Code", pd.Series([np.nan] * len(d), index=d.index)),
            errors="coerce",
        ).astype("Int64")

        if hist_feat_cols:
            ref_col = "cagr_3yr_hist" if "cagr_3yr_hist" in hist_feat_cols else hist_feat_cols[0]
            has_nav_history = scheme_codes.map(nm[ref_col].to_dict()).notna().astype(np.float32)

        for col in hist_feat_cols:
            mapped = scheme_codes.map(nm[col].to_dict())
            # fill NaN with median (schemes with no history)
            med = nm[col].median()
            d[col] = pd.to_numeric(mapped, errors="coerce").fillna(med).astype(np.float32)

    ret_src_cols = [c for c in ["fund_return_1y", "fund_return_3y", "fund_return_5y"] if c in d_raw.columns]
    if ret_src_cols:
        has_fundperf_returns = pd.concat(
            [pd.to_numeric(d_raw[c], errors="coerce") for c in ret_src_cols], axis=1
        ).notna().any(axis=1).astype(np.float32)
    else:
        has_fundperf_returns = pd.Series(0.0, index=d.index, dtype=np.float32)

    d["has_nav_history"] = has_nav_history
    d["has_fundperf_returns"] = has_fundperf_returns
    d["data_quality_flag"] = np.select(
        [d["has_nav_history"] > 0.5, d["has_fundperf_returns"] > 0.5],
        [2.0, 1.0],
        default=0.0,
    ).astype(np.float32)

    # ── FundPerf + Nifty benchmark features (if already attached in mf_df) ─
    # IMPROVEMENT: Skip redundant data_quality fields, keep only most predictive ones
    fundperf_feat_cols: List[str] = []
    benchmarked_series = pd.to_numeric(d.get("benchmarked_flag", pd.Series(0, index=d.index)), errors="coerce").fillna(0)
    non_bench_mask = benchmarked_series <= 0

    # Skip redundant: fundperf_data_quality, benchmark_data_quality (too correlated r=0.965)
    skip_fundperf_cols = [
        "fundperf_data_quality",      # Redundant with benchmark_data_quality
        "benchmark_data_quality",      # Redundant feature - skip it
    ]
    
    for col in _FUNDPERF_FEAT_COLS:
        if col in skip_fundperf_cols:
            continue  # Skip redundant data quality fields
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            # For non-benchmarked funds, benchmark-relative fields are treated as unavailable
            # and hard-zeroed with benchmark flags to avoid synthetic cross-asset comparisons.
            if col in _BENCH_DERIVED_COLS:
                d.loc[non_bench_mask, col] = np.nan
                d[col] = d[col].fillna(0.0).astype(np.float32)
            else:
                med = d[col].median()
                d[col] = d[col].fillna(med if pd.notna(med) else 0.0).astype(np.float32)
            # convert percentage fields to decimal to stabilize scales
            if re.search(r"return", col, flags=re.I):
                d[col] = d[col] / 100.0
            fundperf_feat_cols.append(col)

    if "benchmarked_flag" in d.columns:
        d["benchmarked_flag"] = pd.to_numeric(d["benchmarked_flag"], errors="coerce").fillna(0).astype(np.float32)
        fundperf_feat_cols.append("benchmarked_flag")

    # IMPROVEMENT: Skip benchmark_missing_flag (r=1.0 with benchmarked_flag, perfect redundancy)
    # d["benchmark_missing_flag"] is still computed for internal use but NOT added to model features

    risk_map = {r: i for i, r in enumerate(RISK_CLASSES)}
    if "riskometer_mapped_tier" in d.columns:
        d["riskometer_ord"] = d["riskometer_mapped_tier"].map(risk_map).fillna(2).astype(np.float32)
        fundperf_feat_cols.append("riskometer_ord")

    # ENHANCED: Regime beta (systematic risk exposure proxy)
    # Beta = Excess Return / Volatility (approximation)
    if "excess_return_1y" in d.columns and "vol_1yr_hist" in d.columns:
        excess_1y = pd.to_numeric(d.get("excess_return_1y", pd.Series(0.0, index=d.index)), errors="coerce").fillna(0.0)
        vol_1yr = pd.to_numeric(d.get("vol_1yr_hist", pd.Series(0.05, index=d.index)), errors="coerce").fillna(0.05)
        d["regime_beta"] = (excess_1y / (vol_1yr + 1e-6)).clip(-2.0, 2.0).fillna(0.0).astype(np.float32)
    else:
        d["regime_beta"] = 0.0

    feat_cols = base_feat_cols + hist_feat_cols + fundperf_feat_cols
    X_df = d[feat_cols].copy().astype(np.float32)
    return X_df, feat_cols


def _build_alpha_target(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    nav_metrics: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    ENHANCED: Build alpha target (forward 1Y excess return over NIFTY 500 TRI, net TER).
    Alpha = Fund Return - NIFTY Return - TER (risk-free adjustment).
    
    When NAV metrics available: Use historical excess returns.
    Fallback: Use quality score as proxy.
    """
    if nav_metrics is not None and "cagr_1yr_hist" in X_df.columns:
        # Real alpha proxy from historical excess returns
        ret_fwd = X_df["cagr_1yr_hist"].clip(-0.5, 1.5).fillna(0.0)
        bench_ret = X_df.get("benchmark_return_1y", pd.Series(0.0, index=X_df.index)).clip(-0.5, 1.5).fillna(0.0)
        ter_penalty = (X_df.get("expense_ratio_norm", pd.Series(0.5, index=X_df.index)) * 0.025).fillna(0.0)
        
        # Alpha = Fund Return - Benchmark - TER drag
        alpha = (ret_fwd - bench_ret - ter_penalty).clip(-0.5, 0.5)
        return alpha.values.astype(np.float32)
    else:
        # Fallback to quality score if no history
        return _build_quality_target(df, X_df, nav_metrics)


def _build_quality_target(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    nav_metrics: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Build the quality score (0-1) used as training target.

    When nav_metrics is available (real historical return data):
        q = 0.35 * cagr_3yr_pct  +  0.25 * sharpe_1yr_pct
          + 0.25 * (-max_drawdown_pct)  +  0.15 * momentum_6m_pct

    Otherwise (synthetic fallback — no return data):
        q = 0.50 * AUM_pct  +  0.30 * Recency_pct  +  0.20 * FundAge_pct
    """
    def _pct(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / (rng + 1e-9)

    # ── Real-returns target from NAV-history metrics ──────────────────────
    if nav_metrics is not None and "cagr_3yr_hist" in X_df.columns:
        cagr3   = _pct(X_df["cagr_3yr_hist"].clip(-0.5, 1.0))
        sharpe  = _pct(X_df["sharpe_1yr_hist"].clip(-3.0, 5.0))
        neg_dd  = _pct(-X_df["max_drawdown_hist"])     # less negative = better
        mom     = _pct(X_df["momentum_6m_hist"].clip(-0.5, 1.0))
        # B1: Recency-weighted CAGR: blend (2×1yr + 1×3yr) / 3 so latest perf dominates
        if "cagr_1yr_hist" in X_df.columns:
            cagr1      = _pct(X_df["cagr_1yr_hist"].clip(-0.5, 1.0))
            cagr_blend = 0.667 * cagr1 + 0.333 * cagr3
        else:
            cagr_blend = cagr3
        # Expense ratio penalty: high TER erodes real returns (SEBI cap = 2.5%)
        if "expense_ratio_norm" in X_df.columns:
            er_pen  = _pct(X_df["expense_ratio_norm"])
            miss = X_df.get("ter_missing_flag", pd.Series(1.0, index=X_df.index)).clip(0.0, 1.0)
            # Down-weight TER penalty when TER was imputed instead of real.
            er_weight = 0.10 * (1.0 - 0.5 * miss)
            quality = 0.35 * cagr_blend + 0.25 * sharpe + 0.25 * neg_dd + 0.15 * mom - er_weight * er_pen
            quality = quality.clip(0.0, 1.0)
        else:
            quality = 0.35 * cagr_blend + 0.25 * sharpe + 0.25 * neg_dd + 0.15 * mom

        # Add a forward-leaning calibration component when FundPerf is available.
        if "fund_return_1y" in X_df.columns:
            f1 = _pct(X_df.get("fund_return_1y", pd.Series(0.0, index=X_df.index)).clip(-0.5, 1.5))
            ex1 = _pct(X_df.get("excess_return_1y", pd.Series(0.0, index=X_df.index)).clip(-0.8, 1.5))
            bq = X_df.get("benchmark_data_quality", pd.Series(0.0, index=X_df.index)).clip(0.0, 1.0)
            fresh = 1.0 - _pct(X_df.get("fundperf_recency_days", pd.Series(365.0, index=X_df.index)).clip(0.0, 3650.0))
            fwd = (0.6 * f1 + 0.4 * ex1) * (0.70 + 0.20 * bq + 0.10 * fresh)
            quality = (0.75 * quality + 0.25 * fwd).clip(0.0, 1.0)
        return quality.values.astype(np.float32)

    # ── Real-returns target from FundPerf snapshots (if NAV metrics absent) ─
    if "fund_return_3y" in X_df.columns:
        f1 = _pct(X_df.get("fund_return_1y", X_df["fund_return_3y"]).clip(-0.5, 1.5))
        f3 = _pct(X_df["fund_return_3y"].clip(-0.5, 1.5))
        f5 = _pct(X_df.get("fund_return_5y", X_df["fund_return_3y"]).clip(-0.5, 2.0))

        ex1 = _pct(X_df.get("excess_return_1y", pd.Series(0.0, index=X_df.index)).clip(-0.8, 1.5))
        ex3 = _pct(X_df.get("excess_return_3y", pd.Series(0.0, index=X_df.index)).clip(-0.5, 1.5))
        ex5 = _pct(X_df.get("excess_return_5y", pd.Series(0.0, index=X_df.index)).clip(-0.5, 1.5))
        exn = _pct(X_df.get("excess_vs_nifty_3y", pd.Series(0.0, index=X_df.index)).clip(-1.0, 1.0))

        er_pen = _pct(X_df.get("expense_ratio_norm", pd.Series(0.5, index=X_df.index)))
        miss = X_df.get("ter_missing_flag", pd.Series(1.0, index=X_df.index)).clip(0.0, 1.0)
        bench_quality = X_df.get("benchmark_data_quality", pd.Series(0.0, index=X_df.index)).clip(0.0, 1.0)
        fresh = 1.0 - _pct(X_df.get("fundperf_recency_days", pd.Series(365.0, index=X_df.index)).clip(0.0, 3650.0))

        # Forward-looking emphasis: recent 1Y and excess returns carry the highest influence.
        core_quality = (
            0.24 * f1 + 0.16 * f3 + 0.10 * f5 +
            0.22 * ex1 + 0.14 * ex3 + 0.08 * ex5 + 0.06 * exn
        )
        er_weight = 0.10 * (1.0 - 0.5 * miss)
        quality = core_quality - er_weight * er_pen

        # Confidence calibration: discount stale/low-quality benchmark rows.
        confidence = 0.70 + 0.20 * bench_quality + 0.10 * fresh
        quality = quality * confidence
        return quality.clip(0.0, 1.0).values.astype(np.float32)

    # ── Low-information fallback (no return-history family) ───────────────
    aum_pct      = _pct(X_df["log_aum"])
    recency_pct  = _pct(-X_df["nav_recency_days"])   # lower recency_days = better
    age_pct      = _pct(X_df["fund_age_years"])
    er_pen       = _pct(X_df.get("expense_ratio_norm", pd.Series(0.5, index=X_df.index)))
    miss = X_df.get("ter_missing_flag", pd.Series(1.0, index=X_df.index)).clip(0.0, 1.0)
    er_weight = 0.10 * (1.0 - 0.5 * miss)
    quality = 0.50 * aum_pct + 0.30 * recency_pct + 0.20 * age_pct - er_weight * er_pen
    quality = quality.clip(0.0, 1.0)
    return quality.values.astype(np.float32)


def _value_on_or_before(idx: pd.DatetimeIndex, vals: np.ndarray, t: pd.Timestamp) -> Optional[float]:
    pos = int(idx.searchsorted(t, side="right") - 1)
    if pos < 0:
        return None
    return float(vals[pos])


def _value_on_or_after(idx: pd.DatetimeIndex, vals: np.ndarray, t: pd.Timestamp) -> Optional[float]:
    pos = int(idx.searchsorted(t, side="left"))
    if pos >= len(idx):
        return None
    return float(vals[pos])


def _build_forward_panel_dataset(
    mf_df: pd.DataFrame,
    X_df: pd.DataFrame,
    feat_cols: List[str],
    verbose: bool = True,
) -> Optional[Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]]:
    """
    Build (scheme, as_of_date) panel rows and forward-looking best-fund labels.

    label_best=1 if next-3Y future performance is top-20% inside
    (as_of_date, risk_tier, Scheme_Category):
      - benchmarked funds: by future excess vs Nifty500
      - others: by future absolute return
    """
    if not Path(NAV_HISTORY_PARQUET).exists():
        return None

    # Map static features by Scheme_Code (base row copied, hist cols overwritten by as_of snapshots).
    scheme_codes = pd.to_numeric(mf_df.get("Scheme_Code", pd.Series(np.nan, index=mf_df.index)), errors="coerce").astype("Int64")
    row_idx_map = {
        int(code): int(idx)
        for idx, code in scheme_codes.items()
        if pd.notna(code)
    }
    if not row_idx_map:
        return None

    # Read monthly NAV closes by scheme (memory-efficient row-group pass).
    pf = pq.ParquetFile(str(NAV_HISTORY_PARQUET))
    monthly = {}
    for i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(i).to_pandas().reset_index()
        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce")
        chunk = chunk.dropna(subset=["Scheme_Code", "Date", "NAV"])
        if chunk.empty:
            continue
        chunk["as_of_date"] = chunk["Date"].dt.to_period("M").dt.to_timestamp("M")
        g = chunk.sort_values("Date").groupby(["Scheme_Code", "as_of_date"], as_index=False)["NAV"].last()
        for _, r in g.iterrows():
            code = int(r["Scheme_Code"])
            monthly.setdefault(code, []).append((pd.Timestamp(r["as_of_date"]), float(r["NAV"])))

    if not monthly:
        return None

    nifty = load_nifty500_tri()
    nifty_idx = pd.DatetimeIndex([])
    nifty_vals = np.array([], dtype=np.float64)
    if not nifty.empty:
        nifty_tri = nifty.set_index("Date")["TRI"].sort_index().dropna()
        if not nifty_tri.index.is_unique:
            nifty_tri = nifty_tri.groupby(level=0).last()
        nifty_idx = pd.DatetimeIndex(nifty_tri.index)
        nifty_vals = nifty_tri.values.astype(np.float64)

    hist_keys = [
        "cagr_1yr_hist", "cagr_3yr_hist", "cagr_5yr_hist", "vol_1yr_hist", "sharpe_1yr_hist",
        "max_drawdown_hist", "recovery_time_hist", "momentum_6m_hist", "consistency_1y_hist", "tracking_error_1y_hist",
    ]
    hist_positions = {k: feat_cols.index(k) for k in hist_keys if k in feat_cols}

    launch_series = pd.to_datetime(mf_df.get("Launch_Date", pd.NaT), errors="coerce")
    recency_pos = feat_cols.index("nav_recency_days") if "nav_recency_days" in feat_cols else None
    age_pos = feat_cols.index("fund_age_years") if "fund_age_years" in feat_cols else None

    panel_rows: List[np.ndarray] = []
    panel_meta: List[Dict[str, object]] = []

    for code, arr in monthly.items():
        if code not in row_idx_map:
            continue
        ridx = row_idx_map[code]
        base = X_df.iloc[ridx].values.astype(np.float32)
        scheme_is_benchmarked = float(
            pd.to_numeric(mf_df.iloc[ridx].get("benchmarked_flag", 0), errors="coerce") or 0
        ) > 0.5
        arr = sorted(arr, key=lambda x: x[0])
        dedup = {}
        for d, v in arr:
            dedup[d] = v
        s_idx = pd.DatetimeIndex(sorted(dedup.keys()))
        s_vals = np.array([dedup[d] for d in s_idx], dtype=np.float64)
        if len(s_idx) < 48:
            continue

        as_of_candidates = [d for d in s_idx if d.month in (3, 6, 9, 12)]
        for as_of in as_of_candidates:
            nav_now = _value_on_or_before(s_idx, s_vals, as_of)
            nav_fut = _value_on_or_after(s_idx, s_vals, as_of + pd.DateOffset(years=3))
            if nav_now is None or nav_fut is None or nav_now <= 0:
                continue
            fut_ret = nav_fut / nav_now - 1.0

            # Snapshot metrics computed only from data <= as_of
            hist = pd.Series(s_vals[s_idx <= as_of], index=s_idx[s_idx <= as_of])
            if len(hist) < 24:
                continue
            hvals: Dict[str, float] = {}

            def _cagr(y: int) -> float:
                start = _value_on_or_before(hist.index, hist.values, as_of - pd.DateOffset(years=y))
                end = _value_on_or_before(hist.index, hist.values, as_of)
                if start is None or end is None or start <= 0:
                    return np.nan
                return (end / start) ** (1.0 / y) - 1.0

            c1 = _cagr(1)
            c3 = _cagr(3)
            c5 = _cagr(5)
            hvals["cagr_1yr_hist"] = c1
            hvals["cagr_3yr_hist"] = c3
            hvals["cagr_5yr_hist"] = c5

            w1 = hist[hist.index >= as_of - pd.DateOffset(days=400)]
            if len(w1) >= 20:
                lret = np.log(w1 / w1.shift(1)).dropna()
                vol = float(lret.std() * np.sqrt(252))
                vol = max(vol, 0.002)
                hvals["vol_1yr_hist"] = vol
                hvals["sharpe_1yr_hist"] = float(np.clip((c1 - 0.065) / vol, -10.0, 10.0)) if pd.notna(c1) else np.nan
                hvals["tracking_error_1y_hist"] = np.nan
                if len(nifty_idx) > 0:
                    n_now = _value_on_or_before(nifty_idx, nifty_vals, as_of)
                    _ = n_now
                    n_w = pd.Series(nifty_vals, index=nifty_idx)
                    n_w = n_w[n_w.index >= as_of - pd.DateOffset(days=400)]
                    n_lret = np.log(n_w / n_w.shift(1)).dropna()
                    j = pd.concat([lret.rename("f"), n_lret.rename("n")], axis=1).dropna()
                    if len(j) >= 20:
                        hvals["tracking_error_1y_hist"] = float((j["f"] - j["n"]).std() * np.sqrt(252))
            else:
                hvals["vol_1yr_hist"] = np.nan
                hvals["sharpe_1yr_hist"] = np.nan
                hvals["tracking_error_1y_hist"] = np.nan

            w3 = hist[hist.index >= as_of - pd.DateOffset(years=3)]
            if len(w3) >= 2:
                roll_max = w3.cummax()
                dd = (w3 - roll_max) / roll_max
                hvals["max_drawdown_hist"] = float(dd.min())

                rec_days = []
                in_dd = False
                start_dt = None
                for dt_i, dd_v in dd.items():
                    if not in_dd and dd_v < -0.05:
                        in_dd = True
                        start_dt = dt_i
                    elif in_dd:
                        if w3.loc[dt_i] >= roll_max.loc[dt_i] * 0.98 and start_dt is not None:
                            rec_days.append((dt_i - start_dt).days)
                            in_dd = False
                            start_dt = None
                hvals["recovery_time_hist"] = float(np.median(rec_days)) if rec_days else np.nan
            else:
                hvals["max_drawdown_hist"] = np.nan
                hvals["recovery_time_hist"] = np.nan

            nav_6m = _value_on_or_before(hist.index, hist.values, as_of - pd.DateOffset(months=6))
            nav_asof = _value_on_or_before(hist.index, hist.values, as_of)
            hvals["momentum_6m_hist"] = (nav_asof / nav_6m - 1.0) if (nav_asof and nav_6m and nav_6m > 0) else np.nan

            w5 = hist[hist.index >= as_of - pd.DateOffset(years=5)]
            if len(w5) >= 260:
                r1y = w5 / w5.shift(252) - 1.0
                consistency = np.nan
                if scheme_is_benchmarked and len(nifty_idx) > 0:
                    nifty_5y = pd.Series(nifty_vals, index=nifty_idx)
                    nifty_5y = nifty_5y[
                        (nifty_5y.index >= as_of - pd.DateOffset(years=5)) &
                        (nifty_5y.index <= as_of)
                    ]
                    nr1y = nifty_5y / nifty_5y.shift(252) - 1.0
                    joined_r1y = pd.concat([r1y.rename("fund"), nr1y.rename("nifty")], axis=1).dropna()
                    if not joined_r1y.empty:
                        consistency = float((joined_r1y["fund"] > joined_r1y["nifty"]).mean())
                if np.isnan(consistency):
                    consistency = float((r1y.dropna() > 0).mean()) if r1y.notna().any() else np.nan
                hvals["consistency_1y_hist"] = consistency
            else:
                hvals["consistency_1y_hist"] = np.nan

            row = base.copy()
            for hk, pos in hist_positions.items():
                row[pos] = float(hvals.get(hk, np.nan))

            if recency_pos is not None:
                row[recency_pos] = 0.0
            if age_pos is not None:
                launch = launch_series.iloc[ridx]
                if pd.notna(launch):
                    row[age_pos] = max((as_of - launch).days / 365.25, 0.0)

            if np.isnan(row).any():
                med = np.nanmedian(row)
                row = np.nan_to_num(row, nan=float(0.0 if np.isnan(med) else med))

            nifty_fut = np.nan
            if len(nifty_idx) > 0:
                n0 = _value_on_or_before(nifty_idx, nifty_vals, as_of)
                n3 = _value_on_or_after(nifty_idx, nifty_vals, as_of + pd.DateOffset(years=3))
                if n0 and n3 and n0 > 0:
                    nifty_fut = n3 / n0 - 1.0

            panel_rows.append(row.astype(np.float32))
            panel_meta.append(
                {
                    "scheme_code": code,
                    "as_of_date": as_of,
                    "future_return_3y": fut_ret,
                    "future_excess_3y": fut_ret - nifty_fut if pd.notna(nifty_fut) else np.nan,
                    "benchmarked_flag": float(scheme_is_benchmarked),
                    "risk_tier": str(mf_df.iloc[ridx].get("risk_tier", "Medium")),
                    "Scheme_Category": str(mf_df.iloc[ridx].get("Scheme_Category", "Unknown")),
                }
            )

    if not panel_rows:
        return None

    pm = pd.DataFrame(panel_meta)
    pm["as_of_date"] = pd.to_datetime(pm["as_of_date"])

    grp_cols = ["as_of_date", "risk_tier", "Scheme_Category"]
    equity_mask = pm["benchmarked_flag"] > 0.5

    pm["_metric"] = pm["future_return_3y"]
    pm.loc[equity_mask & pm["future_excess_3y"].notna(), "_metric"] = pm.loc[
        equity_mask & pm["future_excess_3y"].notna(), "future_excess_3y"
    ]
    q80 = pm.groupby(grp_cols)["_metric"].transform(lambda s: s.quantile(0.80) if len(s) > 2 else s.max())
    pm["label_best"] = (pm["_metric"] >= q80).astype(np.float32)

    X_panel = pd.DataFrame(np.vstack(panel_rows), columns=feat_cols)
    y_panel = pm["label_best"].values.astype(np.float32)

    if verbose:
        pos_rate = float(pm["label_best"].mean()) if len(pm) else 0.0
        print(f"[Ensemble] Forward panel rows: {len(pm):,}, positive-rate={pos_rate:.3f}")

    meta = {
        "panel_rows": int(len(pm)),
        "panel_positive": int(pm["label_best"].sum()),
        "panel_unique_funds": int(pm["scheme_code"].nunique()),
    }
    return X_panel, y_panel, meta

def normalize_importance(imp_dict):
    total = sum(imp_dict.values())
    return {k: v / total if total > 0 else 0 for k, v in imp_dict.items()}
# ─── Model fitting ────────────────────────────────────────────────────────────

def fit_fund_ensemble(
    mf_df       : pd.DataFrame,
    n_estimators: int = 200,
    verbose     : bool = True,
    nav_metrics_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Train XGBoost + Random Forest + LightGBM on fund features.

    Parameters
    ----------
    mf_df          : DataFrame returned by load_mutual_funds()
    n_estimators   : tree count for all models
    verbose        : print progress
    nav_metrics_df : optional per-scheme historical metrics

    Returns
    -------
    metrics dict with keys: rf_rmse, xgb_rmse, lgbm_rmse, rf_r2, xgb_r2,
                            lgbm_r2, cv scores, n_funds, uses_history
    """
    # ── Auto-load nav metrics if parquet exists and not provided ─────────
    if nav_metrics_df is None:
        try:
            from src.nav_history import load_nav_metrics, NAV_PARQUET_PATH
            if Path(NAV_PARQUET_PATH).exists():
                if verbose:
                    print("[Ensemble] Loading historical NAV metrics …")
                nav_metrics_df = load_nav_metrics(verbose=verbose)
        except Exception as _e:
            if verbose:
                print(f"[Ensemble] Historical metrics unavailable ({_e}); using synthetic target.")

    uses_history = nav_metrics_df is not None
    if verbose:
        mode = "alpha" if uses_history else "synthetic (AUM/recency/age)"
        print(f"[Ensemble] Building fund features … (target: {mode})")

    X_df, feat_cols = build_fund_features(mf_df, nav_metrics_df)
    # ENHANCED: Try alpha target first, then fallback to quality
    y_alpha = _build_alpha_target(mf_df, X_df, nav_metrics_df)
    y_rule = _build_quality_target(mf_df, X_df, nav_metrics_df)

    panel_meta: Dict[str, int] = {"panel_rows": 0, "panel_positive": 0, "panel_unique_funds": 0}
    X_train_df = X_df
    panel_pack = _build_forward_panel_dataset(mf_df, X_df, feat_cols, verbose=verbose)
    if panel_pack is not None:
        X_train_df, y, panel_meta = panel_pack
        X = X_train_df.values.astype(np.float32)
        train_target = "forward_label_next_3y_top20"
    else:
        X = X_df.values.astype(np.float32)
        y = y_rule.astype(np.float32)
        train_target = "rule_based_quality_fallback"

    data_quality_modes = {"A_nav_history": 0, "B_fundperf_only": 0, "C_low_information": 0}
    case_c_hotspots: Dict[str, float] = {}
    if "data_quality_flag" in X_df.columns:
        mode_counts = X_df["data_quality_flag"].round().astype(int).value_counts().to_dict()
        data_quality_modes = {
            "A_nav_history": int(mode_counts.get(2, 0)),
            "B_fundperf_only": int(mode_counts.get(1, 0)),
            "C_low_information": int(mode_counts.get(0, 0)),
        }
        if "Scheme_Category" in mf_df.columns:
            qtmp = pd.DataFrame(
                {
                    "Scheme_Category": mf_df["Scheme_Category"].astype(str).values,
                    "data_quality_flag": X_df["data_quality_flag"].round().astype(int).values,
                }
            )
            by_cat = qtmp.groupby("Scheme_Category")["data_quality_flag"].apply(lambda s: float((s == 0).mean()))
            case_c_hotspots = by_cat[by_cat >= 0.40].sort_values(ascending=False).head(20).round(3).to_dict()

    n_funds = len(X)
    if verbose:
        print(f"[Ensemble] Training rows={n_funds:,} | features={len(feat_cols)} | target={train_target}")
        print(
            "[Ensemble] Data quality modes: "
            f"A={data_quality_modes['A_nav_history']}, "
            f"B={data_quality_modes['B_fundperf_only']}, "
            f"C={data_quality_modes['C_low_information']}"
        )

    # ── Random Forest (Bagging) ────────────────────────────────────────────
    # Balanced: Reduce overfitting while maintaining learning capacity
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,           # Moderate: avoid extreme leaves
        min_samples_split=10,         # Require more samples to split
        max_depth=None,               # Allow full depth but rely on min_samples_leaf
        bootstrap=True,
        oob_score=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        warm_start=False,
    )
    rf.fit(X, y)

    # ── XGBoost (Gradient Boosting) ───────────────────────────────────────
    # Optimized: 400 iterations at 0.02 learning rate with light regularization
    if _XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=400,                     # Optimal iterations
            learning_rate=0.02,                   # Optimal learning rate
            max_depth=6,                          # Deeper trees for capacity
            min_child_weight=2,                   # Allow splits
            subsample=0.8,                        # Good sampling
            colsample_bytree=0.8,                 # Good feature sampling
            colsample_bylevel=0.8,
            gamma=0.1,                            # Split penalty
            reg_alpha=0.05,                       # Light L1 regularization
            reg_lambda=0.5,                       # Light L2 regularization
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F811
        xgb = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.02,
            max_depth=6,
            min_samples_leaf=3,
            min_samples_split=5,
            subsample=0.8,
            max_features="sqrt",
            random_state=RANDOM_SEED,
        )
    xgb.fit(X, y)

    # ── LightGBM (Leaf-wise Growth) ──────────────────────────────────────
    # Optimized: 400 iterations at 0.02 learning rate with light regularization
    lgbm = None
    if _LGBM_AVAILABLE:
        lgbm = LGBMRegressor(
            n_estimators=400,                     # Optimal iterations
            learning_rate=0.02,                   # Optimal learning rate
            max_depth=7,                          # Deeper trees
            num_leaves=31,                        # Optimal leaf count
            min_child_samples=5,                  # Allow splits
            subsample=0.8,                        # Good sampling
            colsample_bytree=0.8,                 # Good feature sampling
            subsample_freq=1,                     # Apply subsample every iteration
            reg_alpha=0.05,                       # Light L1 regularization
            reg_lambda=0.5,                       # Light L2 regularization
            min_split_gain=0.01,                  # Split penalty
            feature_fraction_bynode=0.9,          # Good feature sampling
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
            force_col_wise=True,
        )
        lgbm.fit(X, y)

    # ── ENHANCED: CatBoost (Categorical gradient boosting) ────────────────
    cat = None
    if _CAT_AVAILABLE:
        try:
            cat = CatBoostRegressor(
                iterations=500,                    # Higher iterations for depth
                learning_rate=0.015,               # Slightly lower for stability
                depth=7,                           # Similar to LGBM
                l2_leaf_reg=3.0,                   # L2 regularization
                random_seed=RANDOM_SEED,
                verbose=False,
                thread_count=-1,
            )
            cat.fit(X, y)
            if verbose:
                print("[Ensemble] CatBoost trained ✓")
        except Exception as e:
            if verbose:
                print(f"[Ensemble] CatBoost training failed: {e}")
            cat = None
    
    # ── ENHANCED: ExtraTreesRegressor (Diversity via random splits) ───────
    et = ExtraTreesRegressor(
        n_estimators=500,                      # More estimators for ensemble
        max_features="sqrt",
        min_samples_leaf=5,
        min_samples_split=10,
        max_depth=None,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    et.fit(X, y)
    if verbose:
        print("[Ensemble] ExtraTreesRegressor trained ✓")

    # ── ENHANCED: Stacking meta-learner (Ridge on base predictions) ───────
    # Stack base model predictions and train Ridge regression
    base_preds_list = [
        rf.predict(X),
        xgb.predict(X),
        lgbm.predict(X) if lgbm is not None else np.zeros(len(y)),
        cat.predict(X) if cat is not None else np.zeros(len(y)),
        et.predict(X),
    ]
    base_preds = np.column_stack(base_preds_list)
    
    meta_ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    meta_ridge.fit(base_preds, y)
    if verbose:
        print(f"[Ensemble] Ridge meta-learner trained with weights: {meta_ridge.coef_.round(3)}")

    # In-sample meta-ensemble score
    meta_pred = meta_ridge.predict(base_preds)
    meta_rmse = float(np.sqrt(mean_squared_error(y, meta_pred)))
    meta_r2   = float(r2_score(y, meta_pred))
    if verbose:
        print(f"[Ensemble] Meta (stacked): RMSE={meta_rmse:.4f}, R²={meta_r2:.4f}")

    # ── In-sample metrics ─────────────────────────────────────────────────
    rf_pred  = rf.predict(X)
    xgb_pred = xgb.predict(X)

    rf_rmse  = float(np.sqrt(mean_squared_error(y, rf_pred)))
    xgb_rmse = float(np.sqrt(mean_squared_error(y, xgb_pred)))
    rf_r2    = float(r2_score(y, rf_pred))
    xgb_r2   = float(r2_score(y, xgb_pred))

    lgbm_rmse, lgbm_r2 = 0.0, 0.0
    lgbm_cv_mean, lgbm_cv_std = 0.0, 0.0
    if lgbm is not None:
        lgbm_pred = lgbm.predict(X)
        lgbm_rmse = float(np.sqrt(mean_squared_error(y, lgbm_pred)))
        lgbm_r2   = float(r2_score(y, lgbm_pred))

    # ── 5-fold CV R² ──────────────────────────────────────────────────────
    rf_cv  = cross_val_score(rf,  X, y, cv=5, scoring="r2", n_jobs=-1)
    xgb_cv = cross_val_score(xgb, X, y, cv=5, scoring="r2", n_jobs=-1)
    if lgbm is not None:
        lgbm_cv = cross_val_score(lgbm, X, y, cv=5, scoring="r2", n_jobs=-1)
        lgbm_cv_mean = float(lgbm_cv.mean())
        lgbm_cv_std  = float(lgbm_cv.std())

    if verbose:
        print(f"[Ensemble] Random Forest   — RMSE={rf_rmse:.4f}  R²={rf_r2:.4f}  CV-R²={rf_cv.mean():.4f}±{rf_cv.std():.4f}")
        print(f"[Ensemble] XGBoost         — RMSE={xgb_rmse:.4f}  R²={xgb_r2:.4f}  CV-R²={xgb_cv.mean():.4f}±{xgb_cv.std():.4f}")
        if lgbm is not None:
            print(f"[Ensemble] LightGBM        — RMSE={lgbm_rmse:.4f}  R²={lgbm_r2:.4f}  CV-R²={lgbm_cv_mean:.4f}±{lgbm_cv_std:.4f}")
        else:
            print("[Ensemble] LightGBM        — SKIPPED (not installed)")

    # ── Feature importance ────────────────────────────────────────────────
    xgb_importance = dict(zip(feat_cols, xgb.feature_importances_.tolist()))
    rf_importance  = dict(zip(feat_cols, rf.feature_importances_.tolist()))
    lgbm_importance = {}
    if lgbm is not None:
        lgbm_importance = dict(zip(feat_cols, lgbm.feature_importances_.tolist()))

    # print(xgb_importance)
    # print(rf_importance)
    # print(lgbm_importance)
    xgb_importance  = normalize_importance(xgb_importance)
    rf_importance   = normalize_importance(rf_importance)
    lgbm_importance = normalize_importance(lgbm_importance)

    # ── Persist artefacts ─────────────────────────────────────────────────
    joblib.dump(rf,        RF_FUND_MODEL_PATH)
    joblib.dump(xgb,       XGB_FUND_MODEL_PATH)
    if lgbm is not None:
        joblib.dump(lgbm,  LGBM_FUND_MODEL_PATH)
    if cat is not None:
        joblib.dump(cat,   CAT_FUND_MODEL_PATH)
    joblib.dump(et,        ET_FUND_MODEL_PATH)
    joblib.dump(meta_ridge, META_RIDGE_MODEL_PATH)  # ENHANCED: Save meta-learner
    joblib.dump(feat_cols, FUND_FEAT_COLS_PATH)
    if verbose:
        print(f"[Ensemble] Models saved → {MODELS_DIR}")

    # ENHANCED: Compute metrics for new models
    cat_rmse = cat_r2 = cat_cv_mean = cat_cv_std = 0.0
    if cat is not None:
        cat_pred = cat.predict(X)
        cat_rmse = float(np.sqrt(mean_squared_error(y, cat_pred)))
        cat_r2   = float(r2_score(y, cat_pred))
        cat_cv = cross_val_score(cat, X, y, cv=5, scoring="r2", n_jobs=-1)
        cat_cv_mean = float(cat_cv.mean())
        cat_cv_std  = float(cat_cv.std())

    et_pred = et.predict(X)
    et_rmse = float(np.sqrt(mean_squared_error(y, et_pred)))
    et_r2   = float(r2_score(y, et_pred))
    et_cv = cross_val_score(et, X, y, cv=5, scoring="r2", n_jobs=-1)
    et_cv_mean = float(et_cv.mean())
    et_cv_std  = float(et_cv.std())

    metrics = {
        "n_funds":        n_funds,
        "n_features":     len(feat_cols),
        "n_models":       5,  # ENHANCED: RF + XGB + LGBM + CAT + ET + META
        "uses_history":   uses_history,
        "train_target":   train_target,
        "forward_panel":  panel_meta,
        "feat_cols":      feat_cols,
        # Individual model metrics
        "rf_rmse":        round(rf_rmse,  4),
        "xgb_rmse":       round(xgb_rmse, 4),
        "lgbm_rmse":      round(lgbm_rmse, 4),
        "cat_rmse":       round(cat_rmse, 4),
        "et_rmse":        round(et_rmse,  4),
        "meta_rmse":      round(meta_rmse, 4),  # ENHANCED: Meta-learner RMSE
        "rf_r2":          round(rf_r2,    4),
        "xgb_r2":         round(xgb_r2,   4),
        "lgbm_r2":        round(lgbm_r2,  4),
        "cat_r2":         round(cat_r2,   4),
        "et_r2":          round(et_r2,    4),
        "meta_r2":        round(meta_r2,  4),  # ENHANCED: Meta-learner R²
        # CV scores
        "rf_cv_r2_mean":  round(float(rf_cv.mean()),  4),
        "rf_cv_r2_std":   round(float(rf_cv.std()),   4),
        "xgb_cv_r2_mean": round(float(xgb_cv.mean()), 4),
        "xgb_cv_r2_std":  round(float(xgb_cv.std()),  4),
        "lgbm_cv_r2_mean": round(lgbm_cv_mean, 4),
        "lgbm_cv_r2_std":  round(lgbm_cv_std,  4),
        "cat_cv_r2_mean":  round(cat_cv_mean, 4),
        "cat_cv_r2_std":   round(cat_cv_std,  4),
        "et_cv_r2_mean":   round(et_cv_mean, 4),
        "et_cv_r2_std":    round(et_cv_std,  4),
        # Feature importance
        "xgb_feature_importance": xgb_importance,
        "rf_feature_importance":  rf_importance,
        "lgbm_feature_importance": lgbm_importance,
        "cat_feature_importance": dict(zip(feat_cols, cat.feature_importances_.tolist())) if cat is not None else {},
        "et_feature_importance": dict(zip(feat_cols, et.feature_importances_.tolist())),
        # Meta-learner weights
        "meta_weights":   {
            "rf": float(meta_ridge.coef_[0]) if len(meta_ridge.coef_) > 0 else 0.2,
            "xgb": float(meta_ridge.coef_[1]) if len(meta_ridge.coef_) > 1 else 0.2,
            "lgbm": float(meta_ridge.coef_[2]) if len(meta_ridge.coef_) > 2 else 0.2,
            "cat": float(meta_ridge.coef_[3]) if len(meta_ridge.coef_) > 3 else 0.2,
            "et": float(meta_ridge.coef_[4]) if len(meta_ridge.coef_) > 4 else 0.2,
        },
        "model_type":     "RF+XGB+LGBM+CAT+ET+META",
        "ensemble_mode":  "stacking",  # ENHANCED: Stacking meta-learner
        "data_quality_modes": data_quality_modes,
        "case_c_hotspots": case_c_hotspots,
    }
    with open(ENSEMBLE_META_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[Ensemble] Artefacts saved → {MODELS_DIR}")

    # ── Segment-specific ensembles (equity-like vs non-equity) ───────────
    segment_info = {"enabled": False, "equity_n": 0, "nonequity_n": 0}
    try:
        is_equity = pd.to_numeric(
            X_train_df.get("benchmarked_flag", pd.Series(0, index=X_train_df.index)),
            errors="coerce",
        ).fillna(0) > 0
        eq_idx = np.where(is_equity.values)[0]
        neq_idx = np.where(~is_equity.values)[0]
        segment_info["equity_n"] = int(len(eq_idx))
        segment_info["nonequity_n"] = int(len(neq_idx))

        def _fit_segment(
            seg_idx: np.ndarray,
            rf_path: Path,
            xgb_path: Path,
            lgbm_path: Path,
            drop_benchmark_cols: bool = False,
        ) -> bool:
            if len(seg_idx) < 200:
                return False
            Xs = X[seg_idx]
            ys = y[seg_idx]

            if drop_benchmark_cols:
                bench_cols = [
                    "benchmark_return_1y", "benchmark_return_3y", "benchmark_return_5y",
                    "excess_return_1y", "excess_return_3y", "excess_return_5y",
                    "excess_vs_nifty_3y", "excess_vs_nifty_5y", "tracking_error_1y_hist",
                ]
                bench_pos = [feat_cols.index(c) for c in bench_cols if c in feat_cols]
                if bench_pos:
                    Xs = Xs.copy()
                    Xs[:, bench_pos] = 0.0
            
            # Segment-specific Random Forest
            rf_s = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_leaf=5,
                min_samples_split=10,
                max_depth=None,
                bootstrap=True,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            rf_s.fit(Xs, ys)
            
            # Segment-specific XGBoost
            if _XGB_AVAILABLE:
                xgb_s = XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.02,
                    max_depth=6,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    colsample_bylevel=0.8,
                    gamma=0.1,
                    reg_alpha=0.05,
                    reg_lambda=0.5,
                    random_state=RANDOM_SEED,
                    verbosity=0,
                    n_jobs=-1,
                    tree_method="hist",
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                xgb_s = GradientBoostingRegressor(
                    n_estimators=400,
                    learning_rate=0.02,
                    max_depth=6,
                    min_samples_leaf=3,
                    min_samples_split=5,
                    subsample=0.8,
                    max_features="sqrt",
                    random_state=RANDOM_SEED,
                )
            xgb_s.fit(Xs, ys)
            
            # Segment-specific LightGBM
            lgbm_s = None
            if _LGBM_AVAILABLE:
                lgbm_s = LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.02,
                    max_depth=7,
                    num_leaves=31,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    subsample_freq=1,
                    reg_alpha=0.05,
                    reg_lambda=0.5,
                    min_split_gain=0.01,
                    feature_fraction_bynode=0.9,
                    random_state=RANDOM_SEED,
                    verbose=-1,
                    n_jobs=-1,
                    force_col_wise=True,
                )
                lgbm_s.fit(Xs, ys)

            joblib.dump(rf_s, rf_path)
            joblib.dump(xgb_s, xgb_path)
            if lgbm_s is not None:
                joblib.dump(lgbm_s, lgbm_path)
            elif lgbm_path.exists():
                lgbm_path.unlink()
            return True

        ok_eq = _fit_segment(eq_idx, RF_FUND_MODEL_EQ_PATH, XGB_FUND_MODEL_EQ_PATH, LGBM_FUND_MODEL_EQ_PATH, drop_benchmark_cols=False)
        ok_neq = _fit_segment(neq_idx, RF_FUND_MODEL_NEQ_PATH, XGB_FUND_MODEL_NEQ_PATH, LGBM_FUND_MODEL_NEQ_PATH, drop_benchmark_cols=True)
        segment_info["enabled"] = bool(ok_eq and ok_neq)
        if verbose:
            print(
                f"[Ensemble] Segment models: enabled={segment_info['enabled']} "
                f"(equity={segment_info['equity_n']}, non_equity={segment_info['nonequity_n']})"
            )
    except Exception as seg_e:
        if verbose:
            print(f"[Ensemble] Segment model training skipped: {seg_e}")

    metrics["segment_models"] = segment_info
    with open(ENSEMBLE_META_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─── Inference ────────────────────────────────────────────────────────────────

def score_funds_ensemble(
    mf_df: pd.DataFrame,
    rf_path: Path = RF_FUND_MODEL_PATH,
    xgb_path: Path = XGB_FUND_MODEL_PATH,
    lgbm_path: Path = LGBM_FUND_MODEL_PATH,
    feat_path: Path = FUND_FEAT_COLS_PATH,
    use_stacking: bool = True,  # ENHANCED: Use meta-learner stacking by default
) -> np.ndarray:
    """
    ENHANCED: Score all funds using the saved ensemble with optional stacking.
    Returns a 1-D numpy array of ensemble scores (higher = better quality).

    If use_stacking=True and meta-learner available: Uses Ridge stacking on all base models.
    Otherwise: Uses traditional weighted ensemble.
    
    Ensemble weights (with LightGBM): 0.40 × XGB + 0.35 × RF + 0.25 × LGBM
    Ensemble weights (without LGBM):  0.50 × XGB + 0.50 × RF
    Stacking: Ridge meta-learner on RF + XGB + LGBM + CAT + ET predictions
    """
    rf        = joblib.load(rf_path)
    xgb       = joblib.load(xgb_path)

    # Keep inference feature construction aligned with training: if training
    # used history-derived columns, auto-load nav metrics again.
    feat_cols = joblib.load(feat_path)
    need_hist = any(c.endswith("_hist") for c in feat_cols)
    nav_metrics = None
    if need_hist:
        try:
            from src.nav_history import load_nav_metrics, NAV_PARQUET_PATH
            if Path(NAV_PARQUET_PATH).exists():
                nav_metrics = load_nav_metrics(verbose=False)
        except Exception:
            nav_metrics = None

    X_df, _ = build_fund_features(mf_df, nav_metrics=nav_metrics)
    X_df = X_df.reindex(columns=feat_cols, fill_value=0.0)
    X    = X_df.values.astype(np.float32)

    # ENHANCED: Try to load new models and meta-learner for stacking
    meta_ridge = None
    cat = None
    et = None
    lgbm = None
    
    if use_stacking and META_RIDGE_MODEL_PATH.exists():
        try:
            meta_ridge = joblib.load(META_RIDGE_MODEL_PATH)
            lgbm = joblib.load(lgbm_path) if lgbm_path.exists() else None
            if CAT_FUND_MODEL_PATH.exists():
                cat = joblib.load(CAT_FUND_MODEL_PATH)
            if ET_FUND_MODEL_PATH.exists():
                et = joblib.load(ET_FUND_MODEL_PATH)
        except Exception:
            meta_ridge = None  # Fall back to weighted ensemble

    # Prefer segment-specific models when available.
    can_segment = (
        RF_FUND_MODEL_EQ_PATH.exists() and XGB_FUND_MODEL_EQ_PATH.exists() and
        RF_FUND_MODEL_NEQ_PATH.exists() and XGB_FUND_MODEL_NEQ_PATH.exists()
    )
    use_segment = False
    try:
        if ENSEMBLE_META_PATH.exists():
            with open(ENSEMBLE_META_PATH) as f:
                meta = json.load(f)
            use_segment = bool(meta.get("segment_models", {}).get("enabled", False))
    except Exception:
        use_segment = False

    # ENHANCED: Use stacking meta-learner if available
    if meta_ridge is not None and not (can_segment and use_segment):
        # Stacking ensemble: predict with all base models, then use meta-learner
        base_preds = np.column_stack([
            rf.predict(X),
            xgb.predict(X),
            lgbm.predict(X) if lgbm is not None else np.zeros(len(X)),
            cat.predict(X) if cat is not None else np.zeros(len(X)),
            et.predict(X) if et is not None else np.zeros(len(X)),
        ])
        ensemble_score = meta_ridge.predict(base_preds).astype(np.float32)
        return np.clip(ensemble_score, -0.5, 1.0)  # Clip to reasonable bounds

    if can_segment and use_segment and "benchmarked_flag" in mf_df.columns:
        is_eq = pd.to_numeric(mf_df["benchmarked_flag"], errors="coerce").fillna(0).values > 0
        ensemble_score = np.zeros(len(mf_df), dtype=np.float32)

        rf_eq = joblib.load(RF_FUND_MODEL_EQ_PATH)
        xgb_eq = joblib.load(XGB_FUND_MODEL_EQ_PATH)
        lgbm_eq = joblib.load(LGBM_FUND_MODEL_EQ_PATH) if LGBM_FUND_MODEL_EQ_PATH.exists() else None

        rf_neq = joblib.load(RF_FUND_MODEL_NEQ_PATH)
        xgb_neq = joblib.load(XGB_FUND_MODEL_NEQ_PATH)
        lgbm_neq = joblib.load(LGBM_FUND_MODEL_NEQ_PATH) if LGBM_FUND_MODEL_NEQ_PATH.exists() else None

        if is_eq.any():
            Xe = X[is_eq]
            s_rf = rf_eq.predict(Xe)
            s_xgb = xgb_eq.predict(Xe)
            if lgbm_eq is not None:
                s = 0.40 * s_xgb + 0.35 * s_rf + 0.25 * lgbm_eq.predict(Xe)
            else:
                s = 0.50 * s_xgb + 0.50 * s_rf
            ensemble_score[is_eq] = s.astype(np.float32)

        if (~is_eq).any():
            Xn = X[~is_eq]
            s_rf = rf_neq.predict(Xn)
            s_xgb = xgb_neq.predict(Xn)
            if lgbm_neq is not None:
                s = 0.40 * s_xgb + 0.35 * s_rf + 0.25 * lgbm_neq.predict(Xn)
            else:
                s = 0.50 * s_xgb + 0.50 * s_rf
            ensemble_score[~is_eq] = s.astype(np.float32)
    else:
        rf_scores  = rf.predict(X)
        xgb_scores = xgb.predict(X)
        if lgbm_path.exists():
            lgbm = joblib.load(lgbm_path)
            lgbm_scores = lgbm.predict(X)
            ensemble_score = 0.40 * xgb_scores + 0.35 * rf_scores + 0.25 * lgbm_scores
        else:
            ensemble_score = 0.50 * rf_scores + 0.50 * xgb_scores

    return np.clip(ensemble_score, 0.0, 1.0)


def load_ensemble_meta() -> Dict:
    """Load saved training metrics from JSON."""
    with open(ENSEMBLE_META_PATH) as f:
        return json.load(f)


def is_ensemble_trained() -> bool:
    """Return True if at least RF + XGB artefacts exist on disk."""
    return (
        RF_FUND_MODEL_PATH.exists() and
        XGB_FUND_MODEL_PATH.exists() and
        FUND_FEAT_COLS_PATH.exists()
    )


# ─── Feature importance plot ──────────────────────────────────────────────────

def plot_ensemble_importance(
    meta: Dict,
    save_dir: Optional[Path] = None,
) -> None:
    """Bar chart comparing XGBoost vs RF vs LightGBM feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir) if save_dir else MODELS_DIR
    xgb_imp  = meta.get("xgb_feature_importance", {})
    rf_imp   = meta.get("rf_feature_importance",  {})
    lgbm_imp = meta.get("lgbm_feature_importance", {})
    features = list(xgb_imp.keys())

    n_models = 3 if lgbm_imp else 2
    x = np.arange(len(features))
    w = 0.25 if n_models == 3 else 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, [xgb_imp[f] for f in features], w,
           label="XGBoost",       color="#EF5350", alpha=0.85, edgecolor="white")
    ax.bar(x,     [rf_imp[f]  for f in features], w,
           label="Random Forest", color="#42A5F5", alpha=0.85, edgecolor="white")
    if lgbm_imp:
        ax.bar(x + w, [lgbm_imp.get(f, 0) for f in features], w,
               label="LightGBM",     color="#66BB6A", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha="right", fontsize=9)
    model_label = meta.get("model_type", "XGB+RF")
    cv_info = (f"XGB CV-R²={meta.get('xgb_cv_r2_mean', 0):.3f}  |  "
               f"RF CV-R²={meta.get('rf_cv_r2_mean', 0):.3f}")
    if lgbm_imp:
        cv_info += f"  |  LGBM CV-R²={meta.get('lgbm_cv_r2_mean', 0):.3f}"
    ax.set_title(f"Ensemble Feature Importances — Fund Scoring ({model_label})\n{cv_info}")
    ax.set_ylabel("Importance")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = save_dir / "plot_ensemble_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Ensemble] Feature importance plot saved → {out}")



