"""
drift_detector.py  –  Customer Feature Drift Detection for Smart Fund Advisor.

Purpose
-------
Detect when a customer's financial profile has shifted significantly since
their last risk assessment.  A large shift means the cached risk label may
be stale and the customer should be re-evaluated through the FL global model.

Statistical Methods (v2)
------------------------
1. **Max absolute delta** (per-feature, legacy) — simple threshold check.
2. **Population Stability Index** (PSI) — industry-standard measure of
   distribution shift between baseline and current feature vectors.
   PSI < 0.10: no significant shift; 0.10–0.25: moderate; > 0.25: significant.
   Ref: Siddiqi (2006), "Credit Risk Scorecards".
3. **Kolmogorov-Smirnov** (KS) test — non-parametric two-sample test
   comparing CDFs of baseline vs current across a batch of customers.
   Ref: scipy.stats.ks_2samp with α=0.05.

Usage
-----
    from src.drift_detector import detect_drift, drift_summary

    flag, score, top_feature, details = detect_drift(
        new_features      = new_row_scaled,
        original_features = orig_row_scaled,
        threshold         = 0.25,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy import stats


# ─── PSI calculation ──────────────────────────────────────────────────────────

def _psi_single(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index for a single feature.
    PSI = Σ (p_i - q_i) · ln(p_i / q_i)  where p = current, q = baseline.
    Both arrays should be 1-D with at least ~30 samples for reliable estimates.

    Bins are defined on the baseline distribution (quantile-based)
    so that each bin has roughly equal baseline mass.
    """
    eps = 1e-8
    # quantile-based bins from baseline
    breakpoints = np.unique(np.percentile(baseline, np.linspace(0, 100, n_bins + 1)))
    if len(breakpoints) < 3:
        breakpoints = np.linspace(baseline.min() - eps, baseline.max() + eps, n_bins + 1)

    base_counts, _ = np.histogram(baseline, bins=breakpoints)
    curr_counts, _ = np.histogram(current,  bins=breakpoints)

    base_pct = (base_counts + eps) / (base_counts.sum() + eps * len(base_counts))
    curr_pct = (curr_counts + eps) / (curr_counts.sum() + eps * len(curr_counts))

    psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    return max(psi, 0.0)


# ─── Core drift function (single customer) ───────────────────────────────────

def detect_drift(
    new_features: np.ndarray,
    original_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.25,
) -> Tuple[bool, float, str, Dict]:
    """
    Detect significant shift between a customer's current and historical features.

    Parameters
    ----------
    new_features      : 1-D array — current feature vector (MinMax-scaled to [0,1])
    original_features : 1-D array — historical feature vector at last assessment
    feature_names     : optional list of feature names (length = n_features)
    threshold         : maximum absolute delta on any feature to trigger drift flag

    Returns
    -------
    (drifted, max_delta, top_feature_name, details)
        drifted       : True if any feature moved more than ``threshold``
        max_delta     : largest absolute change (0 → 1 scale)
        top_feature   : name of the most-drifted feature
        details       : dict with per-feature deltas
    """
    new_arr  = np.asarray(new_features,      dtype=np.float32).ravel()
    orig_arr = np.asarray(original_features, dtype=np.float32).ravel()

    if new_arr.shape != orig_arr.shape:
        raise ValueError(
            f"Feature arrays must have the same shape; "
            f"got {new_arr.shape} vs {orig_arr.shape}"
        )

    deltas    = np.abs(new_arr - orig_arr)
    max_delta = float(deltas.max())
    top_idx   = int(deltas.argmax())

    if feature_names and len(feature_names) > top_idx:
        top_feature = feature_names[top_idx]
    else:
        top_feature = f"feature_{top_idx}"

    drifted = max_delta > threshold

    # Per-feature delta details
    per_feat = {}
    for i, d in enumerate(deltas):
        fname = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
        per_feat[fname] = round(float(d), 4)

    return drifted, max_delta, top_feature, per_feat


# ─── Batch drift with PSI + KS ───────────────────────────────────────────────

def drift_summary(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Compute per-customer drift flags across a batch.

    Returns
    -------
    DataFrame with columns:
        customer_id, max_delta, top_feature, drifted (bool)
    sorted by max_delta descending.
    """
    common_idx = current_df.index.intersection(baseline_df.index)
    records = []
    for cid in common_idx:
        new_vec  = current_df.loc[cid, feature_cols].values.astype(np.float32)
        orig_vec = baseline_df.loc[cid, feature_cols].values.astype(np.float32)
        drifted, max_delta, top_feat, _ = detect_drift(
            new_vec, orig_vec, feature_names=feature_cols, threshold=threshold
        )
        records.append({
            "customer_id": cid,
            "max_delta":   round(max_delta, 4),
            "top_feature": top_feat,
            "drifted":     drifted,
        })

    result = pd.DataFrame(records).sort_values("max_delta", ascending=False)
    return result.reset_index(drop=True)


def population_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Population-level drift analysis using PSI + KS test over a batch.

    Parameters
    ----------
    baseline_df   : DataFrame with scaled feature columns (reference period)
    current_df    : DataFrame with scaled feature columns (current period)
    feature_cols  : list of feature column names
    psi_threshold : PSI > this → significant drift (industry standard: 0.25)
    ks_alpha      : KS p-value < this → reject null of same distribution

    Returns
    -------
    DataFrame indexed by feature with columns:
        psi, psi_flag, ks_statistic, ks_pvalue, ks_flag, drift_severity
    """
    records = []
    for col in feature_cols:
        base_vals = baseline_df[col].dropna().values.astype(np.float64)
        curr_vals = current_df[col].dropna().values.astype(np.float64)

        # PSI
        psi = _psi_single(base_vals, curr_vals) if len(base_vals) >= 10 else np.nan

        # KS test
        if len(base_vals) >= 10 and len(curr_vals) >= 10:
            ks_stat, ks_pval = stats.ks_2samp(base_vals, curr_vals)
        else:
            ks_stat, ks_pval = np.nan, np.nan

        # Severity classification
        if psi > psi_threshold:
            severity = "HIGH"
        elif psi > 0.10:
            severity = "MODERATE"
        else:
            severity = "LOW"

        records.append({
            "feature":      col,
            "psi":          round(float(psi), 4) if not np.isnan(psi) else None,
            "psi_flag":     psi > psi_threshold if not np.isnan(psi) else False,
            "ks_statistic": round(float(ks_stat), 4) if not np.isnan(ks_stat) else None,
            "ks_pvalue":    round(float(ks_pval), 4) if not np.isnan(ks_pval) else None,
            "ks_flag":      ks_pval < ks_alpha if not np.isnan(ks_pval) else False,
            "drift_severity": severity,
        })

    return pd.DataFrame(records).set_index("feature")


# ─── Human-readable drift report ─────────────────────────────────────────────

def drift_report(
    customer_id: str,
    new_features: np.ndarray,
    original_features: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.25,
) -> str:
    """
    Return a one-paragraph human-readable explanations of detected drift.
    """
    drifted, max_delta, top_feat, per_feat = detect_drift(
        new_features, original_features,
        feature_names=feature_names, threshold=threshold
    )

    if not drifted:
        return ""

    new_arr  = np.asarray(new_features).ravel()
    orig_arr = np.asarray(original_features).ravel()
    deltas   = np.abs(new_arr - orig_arr)

    # Collect all features that drifted above threshold
    drifted_feats = [
        (feature_names[i] if i < len(feature_names) else f"feature_{i}", float(deltas[i]))
        for i in np.where(deltas > threshold)[0]
    ]
    drifted_feats.sort(key=lambda x: x[1], reverse=True)

    feat_list = ", ".join(f"'{n}' (+{d:.2f})" for n, d in drifted_feats[:3])
    return (
        f"⚠ Customer {customer_id}: risk profile may be stale. "
        f"Significant changes detected in: {feat_list}. "
        f"Largest shift: '{top_feat}' moved {max_delta:.0%} of feature range. "
        f"Re-run FL inference to update risk label."
    )
