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
2.  Synthetic quality target:  0.50 × AUM_percentile
                              + 0.30 × Recency_percentile
                              + 0.20 × FundAge_percentile
    (analogous to 1/3/5-yr return scoring when actual return data is absent)
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
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, RANDOM_SEED, RISK_CLASSES

# ─── Artefact paths ───────────────────────────────────────────────────────────
RF_FUND_MODEL_PATH   = MODELS_DIR / "rf_fund_model.joblib"
XGB_FUND_MODEL_PATH  = MODELS_DIR / "xgb_fund_model.joblib"
LGBM_FUND_MODEL_PATH = MODELS_DIR / "lgbm_fund_model.joblib"
FUND_FEAT_COLS_PATH  = MODELS_DIR / "fund_feature_cols.joblib"
ENSEMBLE_META_PATH   = MODELS_DIR / "ensemble_fund_meta.json"

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

# Historical metrics columns sourced from nav_history.load_nav_metrics()
_HIST_FEATURE_COLS = [
    "cagr_1yr",
    "cagr_3yr",
    "cagr_5yr",
    "vol_1yr",
    "sharpe_1yr",
    "max_drawdown",
    "momentum_6m",
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

    # ── Expense ratio (synthetic; no TER column in CSV) ─────────────────────
    # Use synth_expense_ratio from mf_df if pre-computed (by load_mutual_funds),
    # otherwise derive from fund traits: Index/ETF ≈ 0.1–0.3%, equity active ≈ 1.0–1.75%
    if "synth_expense_ratio" in d.columns:
        d["expense_ratio_norm"] = (d["synth_expense_ratio"] / 2.50).clip(0.0, 1.0).astype(np.float32)
    else:
        nav_name_s = d.get("Scheme_NAV_Name", pd.Series([""] * len(d), index=d.index)).fillna("").str.lower()
        is_index   = nav_name_s.str.contains("index|etf|nifty|sensex", regex=True).astype(float)
        is_debt_er = (d["risk_tier_ord"] <= 1.0).astype(float)
        base_er    = 1.50 - 1.20 * is_index - 0.70 * is_debt_er
        er_discount = 0.30 * d["amc_size_norm"]
        d["synth_expense_ratio"] = (base_er - er_discount).clip(0.05, 2.50).astype(np.float32)
        d["expense_ratio_norm"]  = (d["synth_expense_ratio"] / 2.50).clip(0.0, 1.0).astype(np.float32)

    # ── Numeric feature list (deterministic order) ─────────────────────────
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
        "expense_ratio_norm",
    ]

    # ── Merge historical return metrics if provided ─────────────────────────
    hist_feat_cols: List[str] = []
    if nav_metrics is not None and len(nav_metrics) > 0:
        nm = nav_metrics[_HIST_FEATURE_COLS].copy()
        nm.index = nm.index.astype(int)
        nm.columns = [f"{c}_hist" for c in nm.columns]   # avoid name clashes
        hist_feat_cols = list(nm.columns)

        # align on original DataFrame index via Scheme_Code
        scheme_codes = pd.to_numeric(
            d.get("Scheme_Code", pd.Series([np.nan] * len(d), index=d.index)),
            errors="coerce",
        ).astype("Int64")

        for col in hist_feat_cols:
            mapped = scheme_codes.map(nm[col].to_dict())
            # fill NaN with median (schemes with no history)
            med = nm[col].median()
            d[col] = pd.to_numeric(mapped, errors="coerce").fillna(med).astype(np.float32)

    feat_cols = base_feat_cols + hist_feat_cols
    X_df = d[feat_cols].copy().astype(np.float32)
    return X_df, feat_cols


def _build_quality_target(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    nav_metrics: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Build the quality score (0-1) used as training target.

    When nav_metrics is available (real historical return data):
        q = 0.35 × cagr_3yr_pct  +  0.25 × sharpe_1yr_pct
          + 0.25 × (-max_drawdown_pct)  +  0.15 × momentum_6m_pct

    Otherwise (synthetic fallback — no return data):
        q = 0.50 × AUM_pct  +  0.30 × Recency_pct  +  0.20 × FundAge_pct
    """
    def _pct(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / (rng + 1e-9)

    # ── Real-returns target ────────────────────────────────────────────────
    if nav_metrics is not None and "cagr_3yr_hist" in X_df.columns:
        cagr3   = _pct(X_df["cagr_3yr_hist"].clip(-0.5, 1.0))
        sharpe  = _pct(X_df["sharpe_1yr_hist"].clip(-3.0, 5.0))
        neg_dd  = _pct(-X_df["max_drawdown_hist"])     # less negative = better
        mom     = _pct(X_df["momentum_6m_hist"].clip(-0.5, 1.0))
        # Expense ratio penalty: high TER erodes real returns (SEBI cap = 2.5%)
        if "expense_ratio_norm" in X_df.columns:
            er_pen  = _pct(X_df["expense_ratio_norm"])
            quality = 0.35 * cagr3 + 0.25 * sharpe + 0.25 * neg_dd + 0.15 * mom - 0.10 * er_pen
            quality = quality.clip(0.0, 1.0)
        else:
            quality = 0.35 * cagr3 + 0.25 * sharpe + 0.25 * neg_dd + 0.15 * mom
        return quality.values.astype(np.float32)

    # ── Synthetic fallback (no history data) ──────────────────────────────
    aum_pct      = _pct(X_df["log_aum"])
    recency_pct  = _pct(-X_df["nav_recency_days"])   # lower recency_days = better
    age_pct      = _pct(X_df["fund_age_years"])
    er_pen       = _pct(X_df.get("expense_ratio_norm", pd.Series(0.5, index=X_df.index)))
    quality = 0.50 * aum_pct + 0.30 * recency_pct + 0.20 * age_pct - 0.10 * er_pen
    quality = quality.clip(0.0, 1.0)
    return quality.values.astype(np.float32)


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
        mode = "real-return history" if uses_history else "synthetic (AUM/recency/age)"
        print(f"[Ensemble] Building fund features … (target: {mode})")

    X_df, feat_cols = build_fund_features(mf_df, nav_metrics_df)
    y = _build_quality_target(mf_df, X_df, nav_metrics_df)
    X = X_df.values

    n_funds = len(X)
    if verbose:
        print(f"[Ensemble] Training on {n_funds} funds | {len(feat_cols)} features")

    # ── Random Forest (Bagging) ────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # ── XGBoost (Gradient Boosting) ───────────────────────────────────────
    if _XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F811
        xgb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )
    xgb.fit(X, y)

    # ── LightGBM (Leaf-wise Growth) ──────────────────────────────────────
    lgbm = None
    if _LGBM_AVAILABLE:
        lgbm = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
        lgbm.fit(X, y)

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

    # ── Persist artefacts ─────────────────────────────────────────────────
    joblib.dump(rf,        RF_FUND_MODEL_PATH)
    joblib.dump(xgb,       XGB_FUND_MODEL_PATH)
    if lgbm is not None:
        joblib.dump(lgbm,  LGBM_FUND_MODEL_PATH)
    joblib.dump(feat_cols, FUND_FEAT_COLS_PATH)

    metrics = {
        "n_funds":        n_funds,
        "n_features":     len(feat_cols),
        "n_models":       3 if lgbm is not None else 2,
        "uses_history":   uses_history,
        "feat_cols":      feat_cols,
        "rf_rmse":        round(rf_rmse,  4),
        "xgb_rmse":       round(xgb_rmse, 4),
        "lgbm_rmse":      round(lgbm_rmse, 4),
        "rf_r2":          round(rf_r2,    4),
        "xgb_r2":         round(xgb_r2,   4),
        "lgbm_r2":        round(lgbm_r2,  4),
        "rf_cv_r2_mean":  round(float(rf_cv.mean()),  4),
        "rf_cv_r2_std":   round(float(rf_cv.std()),   4),
        "xgb_cv_r2_mean": round(float(xgb_cv.mean()), 4),
        "xgb_cv_r2_std":  round(float(xgb_cv.std()),  4),
        "lgbm_cv_r2_mean": round(lgbm_cv_mean, 4),
        "lgbm_cv_r2_std":  round(lgbm_cv_std,  4),
        "xgb_feature_importance": xgb_importance,
        "rf_feature_importance":  rf_importance,
        "lgbm_feature_importance": lgbm_importance,
        "model_type":     "XGB+RF+LGBM" if lgbm is not None else ("XGB+RF" if _XGB_AVAILABLE else "GBR+RF"),
        "ensemble_weights": {"xgb": 0.40, "rf": 0.35, "lgbm": 0.25} if lgbm is not None else {"xgb": 0.50, "rf": 0.50},
    }
    with open(ENSEMBLE_META_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[Ensemble] Artefacts saved → {MODELS_DIR}")

    return metrics


# ─── Inference ────────────────────────────────────────────────────────────────

def score_funds_ensemble(
    mf_df: pd.DataFrame,
    rf_path: Path = RF_FUND_MODEL_PATH,
    xgb_path: Path = XGB_FUND_MODEL_PATH,
    lgbm_path: Path = LGBM_FUND_MODEL_PATH,
    feat_path: Path = FUND_FEAT_COLS_PATH,
) -> np.ndarray:
    """
    Score all funds using the saved ensemble (XGB + RF + optionally LGBM).
    Returns a 1-D numpy array of ensemble scores (higher = better quality).

    Ensemble weights (with LightGBM): 0.40 × XGB + 0.35 × RF + 0.25 × LGBM
    Ensemble weights (without LGBM):  0.50 × XGB + 0.50 × RF
    """
    rf        = joblib.load(rf_path)
    xgb       = joblib.load(xgb_path)
    feat_cols = joblib.load(feat_path)

    X_df, _ = build_fund_features(mf_df)
    X_df = X_df.reindex(columns=feat_cols, fill_value=0.0)
    X    = X_df.values.astype(np.float32)

    rf_scores  = rf.predict(X)
    xgb_scores = xgb.predict(X)

    # Try loading LightGBM model
    if lgbm_path.exists():
        lgbm = joblib.load(lgbm_path)
        lgbm_scores = lgbm.predict(X)
        ensemble_score = 0.40 * xgb_scores + 0.35 * rf_scores + 0.25 * lgbm_scores
    else:
        ensemble_score = 0.50 * rf_scores + 0.50 * xgb_scores

    return ensemble_score


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
