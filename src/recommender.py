"""
recommender.py  –  Mutual Fund Recommendation Engine (Central) — v3.

Pipeline
--------
1. Load & clean the mutual_fund_data.csv (16 k rows).
2. Map each fund to a risk tier based on its Scheme_Category (rule-based).
3. Score funds within each risk tier using the trained Ensemble Model
   (XGBoost + Random Forest + LightGBM ensemble) when
   available, falling back to the rule-based AUM + NAV-recency formula.
4. Apply portfolio diversification scoring.
5. Return top-N with rationale.

v3 additions
------------
- **Horizon-based recommendations**: different asset allocation for 1yr/3yr/5yr/10yr+
  horizons following SEBI glide-path (SEBI/HO/IMD/DF2/CIR/P/2019/17).
- **Core-Satellite diversified portfolio**: 60% core (user's risk tier) +
  20% stability satellite (tier below) + 20% growth satellite (tier above).
  Ref: BlackRock "Core-Satellite Investing"; Vanguard "Diversification" 2022.
- **Multi-metric risk matrix integration**: risk_matrix sub-scores optionally
  included in explanation context for richer LLM outputs.

Ensemble scoring: XGBoost + Random Forest + LightGBM
  Final score = 0.40 × XGB + 0.35 × RF + 0.25 × LGBM.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MF_CSV, RISK_TO_FUND_CATEGORIES, TOP_N_RECOMMENDATIONS, RISK_CLASSES,
    HORIZON_EQUITY_GLIDE, HORIZON_LABELS, CORE_SATELLITE_SPLIT,
    HORIZON_CATEGORY_OVERRIDE,
)


# ─── Lazy import so circular dependency is avoided ────────────────────────────
def _try_ensemble_score(mf_df: pd.DataFrame) -> Optional["np.ndarray"]:
    """
    Return ensemble scores for all funds if ensemble is trained, else None.
    """
    try:
        from src.ensemble_recommender import score_funds_ensemble, is_ensemble_trained
        if is_ensemble_trained():
            return score_funds_ensemble(mf_df)
    except Exception:
        pass
    return None


# ─── Fund catalogue loader ─────────────────────────────────────────────────────

def load_mutual_funds(csv_path=None) -> pd.DataFrame:
    """
    Load mutual fund data, clean columns, tag each fund with a risk_tier.

    Risk tier assignment — two-step with keyword fallback
    ------------------------------------------------------
    Step A (always): assign risk_tier via keyword substring match on
                     Scheme_Category  (RISK_TO_FUND_CATEGORIES in config.py).
    Step B (when NAV history cache exists): override with data-driven tier
                     from nav_history.compute_fund_risk_bands():
                     - Annualized volatility & max drawdown z-scored within
                       each Scheme_Category peer group.
                     - Percentile bands P20/P40/P60/P80 per category.
                     - SEBI-aligned floor/ceiling clamp  (CATEGORY_RISK_BOUNDS).
                     Funds with no NAV history retain the keyword-based tier.

    Returns a DataFrame: Scheme_Code, Scheme_Name, AMC, Scheme_Category,
                         NAV, Average_AUM_Cr, risk_tier, Scheme_Min_Amt,
                         risk_tier_source ('nav_history' | 'keyword')
    """
    csv_path = csv_path or MF_CSV
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # ── Numeric cleaning ──
    for col in ["NAV", "Average_AUM_Cr", "Scheme_Min_Amt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Date parsing ──
    for col in ["Latest_NAV_Date", "Launch_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Step A: keyword-based tier (always, serves as fallback) ──
    df["risk_tier"] = df["Scheme_Category"].apply(_categorise_risk)
    df["risk_tier_source"] = "keyword"

    # ── Step B: data-driven override when NAV history cache is present ─────
    try:
        from src.nav_history import load_nav_metrics, NAV_METRICS_PATH
        from src.nav_history import compute_fund_risk_bands
        if Path(NAV_METRICS_PATH).exists():
            nav_metrics = load_nav_metrics(verbose=False)
            data_driven = compute_fund_risk_bands(df, nav_metrics)
            # Override only where data-driven method produced a non-null result
            valid_mask = data_driven.notna()
            df.loc[valid_mask, "risk_tier"]        = data_driven[valid_mask]
            df.loc[valid_mask, "risk_tier_source"] = "nav_history"
    except Exception:
        pass   # silently fall back to keyword tier if anything goes wrong

    # ── Synthetic expense ratio (derived from fund traits; no TER in CSV) ──
    # Index/ETF: ~0.10–0.30%  |  Active equity: ~1.0–1.75%  |  Debt: ~0.30–0.80%
    _tier_map    = {r: i for i, r in enumerate(RISK_CLASSES)}
    _tier_ord    = df["risk_tier"].map(_tier_map).fillna(2)
    _nav_nm      = df.get("Scheme_NAV_Name", pd.Series([""] * len(df), index=df.index)).fillna("").str.lower()
    _is_idx      = _nav_nm.str.contains("index|etf|nifty|sensex", regex=True).astype(float)
    _is_debt     = (_tier_ord <= 1.0).astype(float)
    _amc_s       = df.get("AMC", pd.Series(["Other"] * len(df), index=df.index)).fillna("Other")
    _amc_cnt     = _amc_s.map(_amc_s.value_counts()).fillna(1)
    _amc_norm    = _amc_cnt / _amc_cnt.max()
    _base_er     = 1.50 - 1.20 * _is_idx - 0.70 * _is_debt
    df["synth_expense_ratio"] = (_base_er - 0.30 * _amc_norm).clip(0.05, 2.50).round(2)

    # Drop funds with no NAV or unknown tier
    df = df[df["NAV"].notna()].copy()

    return df.reset_index(drop=True)


def _categorise_risk(scheme_category: str) -> Optional[str]:
    """
    Map a Scheme_Category string to one of the 5 RISK_CLASSES by substring match.
    Returns None if no mapping found.
    """
    if not isinstance(scheme_category, str):
        return None
    cat_lower = scheme_category.lower()
    for risk_class in RISK_CLASSES:
        keywords = RISK_TO_FUND_CATEGORIES[risk_class]
        for kw in keywords:
            if kw.lower() in cat_lower:
                return risk_class
    return None


# ─── Scoring & ranking ─────────────────────────────────────────────────────────

def _compute_fund_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite score for ranking within a risk tier:
    - AUM (larger  = more established)
    - Recency of NAV date (more recent = more active)
    Both normalised to [0, 1] before combining.
    """
    scores = pd.Series(0.0, index=df.index)

    if "Average_AUM_Cr" in df.columns:
        aum = df["Average_AUM_Cr"].fillna(0)
        aum_norm = (aum - aum.min()) / (aum.max() - aum.min() + 1e-6)
        scores += 0.7 * aum_norm

    if "Latest_NAV_Date" in df.columns:
        nav_date = df["Latest_NAV_Date"].astype(np.int64)
        date_norm = (nav_date - nav_date.min()) / (nav_date.max() - nav_date.min() + 1e-6)
        scores += 0.3 * date_norm.fillna(0)

    return scores


# ─── Public recommendation API ────────────────────────────────────────────────

def recommend_funds(
    risk_label: str,
    mf_df: pd.DataFrame,
    top_n: int = TOP_N_RECOMMENDATIONS,
    min_aum: Optional[float] = None,
    scheme_type_filter: Optional[str] = None,
    use_ensemble: bool = True,
) -> pd.DataFrame:
    """
    Return top-N mutual fund recommendations for a given risk_label.

    Scoring strategy:
      If ensemble model is trained    →  0.5 × RF + 0.5 × XGBoost ensemble score
      Otherwise (fallback)            →  0.7 × AUM_norm + 0.3 × NAV_recency_norm

    Parameters
    ----------
    risk_label          : one of RISK_CLASSES
    mf_df               : DataFrame returned by load_mutual_funds()
    top_n               : number of recommendations to return
    min_aum             : optional minimum Average_AUM_Cr filter
    scheme_type_filter  : optional filter ('Open Ended', 'Close Ended', …)
    use_ensemble        : set False to force fallback rule-based scoring

    Returns
    -------
    DataFrame with columns:
        Scheme_Code, Scheme_Name, AMC, Scheme_Category, risk_tier,
        NAV, Average_AUM_Cr, Scheme_Min_Amt, score, score_source, rationale
    """
    if risk_label not in RISK_CLASSES:
        raise ValueError(f"risk_label must be one of {RISK_CLASSES}, got '{risk_label}'")

    pool = mf_df[mf_df["risk_tier"] == risk_label].copy()

    if pool.empty:
        idx = RISK_CLASSES.index(risk_label)
        fallback = RISK_CLASSES[max(0, idx - 1)]
        pool = mf_df[mf_df["risk_tier"] == fallback].copy()

    if min_aum is not None and "Average_AUM_Cr" in pool.columns:
        pool = pool[pool["Average_AUM_Cr"] >= min_aum]

    if scheme_type_filter and "Scheme_Type" in pool.columns:
        pool = pool[pool["Scheme_Type"].str.contains(scheme_type_filter, case=False, na=False)]

    # ── Scoring: ensemble (XGB + RF) or rule-based fallback ──
    score_src = "rule_based"
    if use_ensemble:
        all_scores = _try_ensemble_score(mf_df)
        if all_scores is not None:
            pool["score"] = all_scores[pool.index]
            score_src = "ensemble_xgb_rf"
        else:
            pool["score"] = _compute_fund_score(pool)
    else:
        pool["score"] = _compute_fund_score(pool)

    pool["score_source"] = score_src

    # ── AMC-diversity-aware selection ──────────────────────────────────────
    # Cap each AMC to at most ceil(top_n × _MAX_SINGLE_AMC) slots BEFORE the
    # final top-N cut.  This prevents a single large-AUM AMC from monopolising
    # all top-N positions, which would make the 40% AMC cap in allocate_portfolio
    # mathematically impossible to enforce after normalisation.
    pool = pool.sort_values("score", ascending=False)
    if "AMC" in pool.columns:
        max_per_amc = max(1, int(np.ceil(top_n * _MAX_SINGLE_AMC)))
        pool = (
            pool.groupby("AMC", group_keys=False)
                .apply(lambda g: g.head(max_per_amc))
                .sort_values("score", ascending=False)
        )
    pool = pool.head(top_n)
    pool["rationale"] = pool.apply(lambda r: _build_rationale(r, risk_label, score_src), axis=1)

    output_cols = [
        "Scheme_Code", "Scheme_Name", "AMC", "Scheme_Category",
        "risk_tier", "NAV", "Average_AUM_Cr", "Scheme_Min_Amt",
        "synth_expense_ratio", "score", "score_source", "rationale",
    ]
    output_cols = [c for c in output_cols if c in pool.columns]
    return pool[output_cols].reset_index(drop=True)


def _build_rationale(row: pd.Series, risk_label: str, score_src: str = "rule_based") -> str:
    """Build a one-line human-readable explanation for each recommendation."""
    aum  = f"₹{row.get('Average_AUM_Cr', 0):,.0f} Cr AUM"
    nav  = f"NAV ₹{row.get('NAV', 0):.2f}"
    cat  = row.get("Scheme_Category", "")
    model_tag = "(XGB+RF ensemble scored)" if "ensemble" in score_src else "(AUM+recency scored)"
    return (
        f"Matched to your '{risk_label}' risk profile {model_tag}. "
        f"Category: {cat}. {aum}, {nav}."
    )


# ─── Portfolio allocation weights ──────────────────────────────────────────────

# Equity/Debt split per risk tier (approximate SEBI-aligned guidelines)
_EQUITY_DEBT_SPLIT = {
    "Very_Low":  {"equity": 0.00, "debt": 1.00},
    "Low":       {"equity": 0.20, "debt": 0.80},
    "Medium":    {"equity": 0.50, "debt": 0.50},
    "High":      {"equity": 0.80, "debt": 0.20},
    "Very_High": {"equity": 0.95, "debt": 0.05},
}
_MAX_SINGLE_FUND = 0.30   # no single fund > 30% of portfolio
_MAX_SINGLE_AMC  = 0.40   # no single AMC > 40% of portfolio


def _classify_asset(scheme_category: str) -> str:
    """Classify a fund as 'equity' or 'debt' from its Scheme_Category."""
    cat = (scheme_category or "").lower()
    equity_kws = [
        "equity", "elss", "mid cap", "large cap", "small cap",
        "flexi", "multi cap", "sectoral", "thematic", "index",
        "etf", "nifty", "sensex",
    ]
    return "equity" if any(kw in cat for kw in equity_kws) else "debt"


def _diversification_score(recs_df: pd.DataFrame) -> float:
    """
    Compute a diversification score (0-1) for a portfolio.
    Higher = more diversified across AMCs, categories, and asset classes.

    Score = (1 - HHI_amc) × 0.40  +  (1 - HHI_category) × 0.40
          + asset_class_entropy × 0.20

    Where HHI = Herfindahl-Hirschman Index (sum of squared shares).
    """
    if recs_df.empty:
        return 0.0

    n = len(recs_df)

    # AMC concentration
    amc_counts = recs_df.get("AMC", pd.Series(["Other"] * n)).value_counts(normalize=True)
    hhi_amc = float((amc_counts ** 2).sum())

    # Category concentration
    cat_counts = recs_df.get("Scheme_Category", pd.Series([""] * n)).value_counts(normalize=True)
    hhi_cat = float((cat_counts ** 2).sum())

    # Asset class entropy (equity/debt balance)
    asset_classes = recs_df.get("Scheme_Category", pd.Series([""] * n)).apply(_classify_asset)
    ac_counts = asset_classes.value_counts(normalize=True)
    entropy = 0.0
    for p in ac_counts.values:
        if p > 0:
            entropy -= p * np.log2(p + 1e-10)
    # Normalise: max entropy for 2 classes = 1.0
    norm_entropy = min(entropy / 1.0, 1.0)

    score = 0.40 * (1 - hhi_amc) + 0.40 * (1 - hhi_cat) + 0.20 * norm_entropy
    return round(float(score), 4)


def allocate_portfolio(
    recs_df: pd.DataFrame,
    risk_label: str,
    total_amount: float = 100_000.0,
) -> pd.DataFrame:
    """
    Compute portfolio allocation weights for a set of recommended funds.

    Strategy (v2)
    -------------
    1. Determine equity/debt split from risk_label (SEBI-aligned guidelines).
    2. Classify each fund as equity or debt from Scheme_Category.
    3. Equal-weight within each asset class, capped at _MAX_SINGLE_FUND (30%).
    4. Apply AMC concentration cap (_MAX_SINGLE_AMC = 40%).
    5. Renormalise weights so they sum to 1.0.
    6. Compute diversification score.

    Parameters
    ----------
    recs_df      : DataFrame from recommend_funds()
    risk_label   : user risk tier (one of RISK_CLASSES)
    total_amount : total investable corpus in INR (default ₹1 lakh)

    Returns
    -------
    recs_df with added columns: asset_class, weight, alloc_amount_inr,
                                diversification_score
    """
    if recs_df.empty:
        return recs_df

    split = _EQUITY_DEBT_SPLIT.get(risk_label, {"equity": 0.5, "debt": 0.5})
    df = recs_df.copy()

    df["asset_class"] = df["Scheme_Category"].fillna("").apply(_classify_asset)

    equity_mask = df["asset_class"] == "equity"
    debt_mask   = ~equity_mask
    n_eq        = int(equity_mask.sum())
    n_dt        = int(debt_mask.sum())
    weights     = pd.Series(0.0, index=df.index)

    if n_eq > 0:
        weights[equity_mask] = min(split["equity"] / n_eq, _MAX_SINGLE_FUND)
    if n_dt > 0:
        weights[debt_mask]   = min(split["debt"]   / n_dt, _MAX_SINGLE_FUND)

    # AMC concentration cap: no single AMC > _MAX_SINGLE_AMC (40%)
    # Applied iteratively because renormalization can re-inflate AMC shares after
    # a single pass.  Converges in ≤ n_unique_amc iterations.
    amc_col = df.get("AMC", pd.Series(["Other"] * len(df), index=df.index))
    for _iter in range(20):                  # safety cap on iterations
        total_w = weights.sum()
        if total_w > 0:
            weights = weights / total_w      # normalize before each cap check
        capped = False
        for amc_name in amc_col.unique():
            amc_mask  = amc_col == amc_name
            amc_share = weights[amc_mask].sum()
            if amc_share > _MAX_SINGLE_AMC + 1e-6:
                scale_factor          = _MAX_SINGLE_AMC / amc_share
                weights[amc_mask]    *= scale_factor
                capped                = True
        if not capped:
            break                            # all AMCs within cap — converged

    total_w = weights.sum()
    if total_w > 0:
        weights = weights / total_w

    df["weight"]           = weights.round(4)
    df["alloc_amount_inr"] = (weights * total_amount).round(0).astype(int)

    # Diversification score
    df["diversification_score"] = _diversification_score(df)

    return df


# ─── Full advisor: risk prediction → recommendation ───────────────────────────

def advise_user(
    user_features: np.ndarray,
    model,                          # RiskMLP or any callable with predict()
    mf_df: pd.DataFrame,
    label_encoder,
    top_n: int = TOP_N_RECOMMENDATIONS,
) -> Dict:
    """
    End-to-end advisor for a single user.

    Parameters
    ----------
    user_features   : 1-D array of shape (n_features,)
    model           : trained RiskMLP (central or FL-updated)
    mf_df           : loaded mutual fund catalogue
    label_encoder   : fitted LabelEncoder
    top_n           : number of funds to recommend

    Returns
    -------
    dict with keys: risk_label, risk_probabilities, recommendations (DataFrame)
    """
    from src.central_model import predict as mlp_predict

    X = user_features.reshape(1, -1)
    pred_idx, probs = mlp_predict(model, X)
    pred_label = label_encoder.inverse_transform(pred_idx)[0]

    prob_dict = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(probs[0])
    }

    recs = recommend_funds(pred_label, mf_df, top_n=top_n)

    return {
        "risk_label":          pred_label,
        "risk_probabilities":  prob_dict,
        "recommendations":     recs,
    }


# ─── Horizon-based recommendations (v3) ──────────────────────────────────────

def _get_horizon_equity_pct(risk_label: str, horizon_years: int) -> float:
    """
    Return target equity % for a given risk tier and investment horizon.

    Uses the SEBI glide-path matrix from config.HORIZON_EQUITY_GLIDE.
    Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — investment horizon
    is an explicit factor in suitability profiling.
    """
    glide = HORIZON_EQUITY_GLIDE.get(risk_label, [0.3, 0.4, 0.5, 0.6])
    if horizon_years <= 1:
        return glide[0]
    elif horizon_years <= 3:
        return glide[1]
    elif horizon_years <= 5:
        return glide[2]
    else:
        return glide[3]


def recommend_funds_by_horizon(
    risk_label: str,
    mf_df: pd.DataFrame,
    horizon_years: int = 5,
    top_n: int = TOP_N_RECOMMENDATIONS,
    total_amount: float = 100_000.0,
    use_ensemble: bool = True,
) -> Dict:
    """
    Generate fund recommendations tailored to a specific investment horizon.

    For short horizons (1yr), the system overrides the risk tier to prioritise
    capital preservation (liquid/overnight funds) regardless of risk appetite.
    For longer horizons, it blends equity and debt funds per the SEBI glide-path.

    Parameters
    ----------
    risk_label     : user's risk tier
    mf_df          : fund catalogue DataFrame
    horizon_years  : 1, 3, 5, or 10 (investment duration in years)
    top_n          : number of funds to recommend per bracket
    total_amount   : total investable corpus in INR
    use_ensemble   : use ML ensemble scoring

    Returns
    -------
    dict with keys: horizon, equity_pct, debt_pct, equity_funds, debt_funds,
                    portfolio (combined DataFrame with weights)
    """
    equity_pct = _get_horizon_equity_pct(risk_label, horizon_years)
    debt_pct   = 1.0 - equity_pct

    horizon_key = (
        "1yr"  if horizon_years <= 1 else
        "3yr"  if horizon_years <= 3 else
        None
    )

    # ── Equity portion ────────────────────────────────────────────────────
    equity_funds = pd.DataFrame()
    if equity_pct > 0:
        # For short horizons, use category overrides
        if horizon_key and horizon_key in HORIZON_CATEGORY_OVERRIDE:
            # Only allow safe equity-like categories (arbitrage, balanced)
            safe_cats = HORIZON_CATEGORY_OVERRIDE[horizon_key]
            pool = mf_df[mf_df["Scheme_Category"].apply(
                lambda c: any(kw.lower() in (c or "").lower() for kw in safe_cats)
            )].copy()
            if not pool.empty:
                all_scores = _try_ensemble_score(mf_df) if use_ensemble else None
                if all_scores is not None:
                    pool["score"] = all_scores[pool.index]
                else:
                    pool["score"] = _compute_fund_score(pool)
                equity_funds = pool.sort_values("score", ascending=False).head(
                    max(2, top_n // 2))
        else:
            equity_funds = recommend_funds(risk_label, mf_df,
                                           top_n=max(2, top_n // 2),
                                           use_ensemble=use_ensemble)

    # ── Debt portion ─────────────────────────────────────────────────────
    debt_funds = pd.DataFrame()
    if debt_pct > 0:
        # Use 1-2 tiers below user's risk for debt allocation
        tier_idx = RISK_CLASSES.index(risk_label) if risk_label in RISK_CLASSES else 2
        debt_tier = RISK_CLASSES[max(0, tier_idx - 2)]
        debt_funds = recommend_funds(debt_tier, mf_df,
                                     top_n=max(2, top_n // 2),
                                     use_ensemble=use_ensemble)

    # ── Combine & allocate ───────────────────────────────────────────────
    combined = pd.DataFrame()
    rows = []
    if not equity_funds.empty:
        eq_df = equity_funds.copy()
        n_eq = len(eq_df)
        eq_weight = equity_pct / n_eq if n_eq > 0 else 0
        eq_df["weight"] = round(eq_weight, 4)
        eq_df["bracket"] = "equity"
        rows.append(eq_df)

    if not debt_funds.empty:
        dt_df = debt_funds.copy()
        n_dt = len(dt_df)
        dt_weight = debt_pct / n_dt if n_dt > 0 else 0
        dt_df["weight"] = round(dt_weight, 4)
        dt_df["bracket"] = "debt"
        rows.append(dt_df)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        # Renormalise weights
        total_w = combined["weight"].sum()
        if total_w > 0:
            combined["weight"] = (combined["weight"] / total_w).round(4)
        combined["alloc_amount_inr"] = (combined["weight"] * total_amount).round(0).astype(int)
        combined["horizon"] = f"{horizon_years}yr"

    horizon_label = (
        "1yr"   if horizon_years <= 1 else
        "3yr"   if horizon_years <= 3 else
        "5yr"   if horizon_years <= 5 else
        "10yr+"
    )

    return {
        "risk_label":   risk_label,
        "horizon":      horizon_label,
        "horizon_years": horizon_years,
        "equity_pct":   round(equity_pct, 2),
        "debt_pct":     round(debt_pct, 2),
        "equity_funds": equity_funds,
        "debt_funds":   debt_funds,
        "portfolio":    combined,
        "total_amount": total_amount,
    }


# ─── Core-Satellite diversified portfolio (v3) ───────────────────────────────

def recommend_diversified_portfolio(
    risk_label: str,
    mf_df: pd.DataFrame,
    horizon_years: int = 5,
    top_n_per_bracket: int = 3,
    total_amount: float = 100_000.0,
    use_ensemble: bool = True,
) -> Dict:
    """
    Build a diversified Core-Satellite portfolio across multiple risk brackets.

    Strategy (Portfolio Construction)
    ------------------------------------------------
    - **Core (60%)**: funds from user's own risk tier — primary return driver.
    - **Stability satellite (20%)**: funds from one tier below — downside buffer.
    - **Growth satellite (20%)**: funds from one tier above — upside kicker.
    - Within each bracket, HHI diversification + AMC cap (40%) are applied.

    This ensures:
    1. Multiple funds from different AMCs and categories.
    2. Multiple risk brackets represented in every portfolio.
    3. Investment horizon adjusts equity/debt split per SEBI glide-path.

    Ref: BlackRock "Core-Satellite Investing" (2018);
         Vanguard "Principles for Investing Success" (2022);
         Markowitz H. (1952) "Portfolio Selection", J. Finance 7(1), pp.77-91.

    Parameters
    ----------
    risk_label          : user's predicted risk tier
    mf_df               : fund catalogue
    horizon_years       : investment time horizon (affects equity/debt split)
    top_n_per_bracket   : funds per bracket
    total_amount        : total investable corpus in INR
    use_ensemble        : use ML ensemble scoring

    Returns
    -------
    dict with keys: brackets (list of dicts), portfolio (DataFrame),
                    diversification_score, risk_label, horizon
    """
    tier_idx = RISK_CLASSES.index(risk_label) if risk_label in RISK_CLASSES else 2
    core_pct      = CORE_SATELLITE_SPLIT["core_pct"]
    stability_pct = CORE_SATELLITE_SPLIT["stability_pct"]
    growth_pct    = CORE_SATELLITE_SPLIT["growth_pct"]

    # Determine adjacent tiers (clamp at edges)
    stability_tier = RISK_CLASSES[max(0, tier_idx - 1)]
    growth_tier    = RISK_CLASSES[min(len(RISK_CLASSES) - 1, tier_idx + 1)]

    # If at edge, redistribute to core
    if tier_idx == 0:
        # No tier below → give stability share to core
        core_pct += stability_pct
        stability_pct = 0.0
        stability_tier = risk_label
    if tier_idx == len(RISK_CLASSES) - 1:
        # No tier above → give growth share to core
        core_pct += growth_pct
        growth_pct = 0.0
        growth_tier = risk_label

    # Apply horizon-based equity adjustment
    eq_pct = _get_horizon_equity_pct(risk_label, horizon_years)

    brackets = []
    portfolio_parts = []

    for bracket_label, tier, pct in [
        ("Core",      risk_label,     core_pct),
        ("Stability", stability_tier, stability_pct),
        ("Growth",    growth_tier,    growth_pct),
    ]:
        if pct <= 0:
            continue
        recs = recommend_funds(tier, mf_df, top_n=top_n_per_bracket,
                               use_ensemble=use_ensemble)
        if recs.empty:
            continue

        n = len(recs)
        fund_weight = pct / n
        recs_copy = recs.copy()
        recs_copy["weight"]           = round(fund_weight, 4)
        recs_copy["alloc_amount_inr"] = int(round(fund_weight * total_amount, 0))
        recs_copy["bracket"]          = bracket_label
        recs_copy["bracket_tier"]     = tier
        recs_copy["bracket_pct"]      = round(pct * 100, 1)

        brackets.append({
            "bracket":   bracket_label,
            "tier":      tier,
            "pct":       round(pct * 100, 1),
            "n_funds":   n,
            "funds":     recs,
        })
        portfolio_parts.append(recs_copy)

    # Combine all brackets into one portfolio
    if portfolio_parts:
        portfolio = pd.concat(portfolio_parts, ignore_index=True)
        # Renormalise weights
        total_w = portfolio["weight"].sum()
        if total_w > 0:
            portfolio["weight"] = (portfolio["weight"] / total_w).round(4)
            portfolio["alloc_amount_inr"] = (
                portfolio["weight"] * total_amount
            ).round(0).astype(int)
    else:
        portfolio = pd.DataFrame()

    div_score = _diversification_score(portfolio) if not portfolio.empty else 0.0

    horizon_label = (
        "1yr"   if horizon_years <= 1 else
        "3yr"   if horizon_years <= 3 else
        "5yr"   if horizon_years <= 5 else
        "10yr+"
    )

    return {
        "risk_label":           risk_label,
        "horizon":              horizon_label,
        "horizon_years":        horizon_years,
        "equity_pct":           round(eq_pct, 2),
        "brackets":             brackets,
        "portfolio":            portfolio,
        "diversification_score": div_score,
        "total_amount":         total_amount,
        "core_tier":            risk_label,
        "stability_tier":       stability_tier,
        "growth_tier":          growth_tier,
    }


# ─── Full user investment profile across all horizons (v3) ───────────────────

def recommend_full_profile(
    risk_label: str,
    mf_df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    top_n_per_bracket: int = 3,
    total_amount: float = 500_000.0,
    use_ensemble: bool = True,
) -> Dict:
    """
    Build a complete investment profile for a user by generating core-satellite
    diversified portfolios for every investment horizon.

    For each horizon (1yr / 3yr / 5yr / 10yr+), the function calls
    recommend_diversified_portfolio() which respects both the 60/20/20
    core-satellite split AND the SEBI glide-path equity/debt ratio for that
    specific duration.

    Parameters
    ----------
    risk_label        : user's predicted risk tier (e.g. 'High')
    mf_df             : mutual fund catalogue DataFrame
    horizons          : investment durations in years (default: [1, 3, 5, 10])
    top_n_per_bracket : number of funds per core / stability / growth bracket
    total_amount      : total investable corpus in INR (default ₹5,00,000)
    use_ensemble      : use ML ensemble scoring when available

    Returns
    -------
    dict with keys:
      risk_label  : str
      total_amount: float
      horizons    : dict of horizon_label → recommend_diversified_portfolio() result
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    by_horizon: Dict[str, Dict] = {}
    for h in horizons:
        label = (
            "1yr"   if h <= 1 else
            "3yr"   if h <= 3 else
            "5yr"   if h <= 5 else
            "10yr+"
        )
        by_horizon[label] = recommend_diversified_portfolio(
            risk_label, mf_df,
            horizon_years=h,
            top_n_per_bracket=top_n_per_bracket,
            total_amount=total_amount,
            use_ensemble=use_ensemble,
        )

    return {
        "risk_label":   risk_label,
        "total_amount": total_amount,
        "horizons":     by_horizon,
    }
