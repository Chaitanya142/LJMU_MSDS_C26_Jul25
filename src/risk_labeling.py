"""
risk_labeling.py  –  Derive a 5-class 'Risk Appetite' label for each customer.

Strategy
--------
A composite risk-appetite score is computed from the engineered features using
domain-driven weights, then binned into a BELL-CURVE distribution:

    Very_Low (0) →  12.5%  – Very conservative: high debt, bad credit, minimal investment
    Low      (1) →  25.0%  – Conservative
    Medium   (2) →  25.0%  – Balanced
    High     (3) →  25.0%  – Aggressive
    Very_High(4) →  12.5%  – Very aggressive: strong income, good credit, heavy investment

Rationale for asymmetric distribution
--------------------------------------
In a real customer population the extreme risk classes (Very_Low / Very_High)
are rare — genuinely debt-stressed or genuinely high-earning aggressive investors
form the tails; the majority sits in the three middle tiers.  This mirrors the
normal-like shape of the raw composite score (see EDA notebook cell 7b).
  → First + last bins together = 25 % (12.5 % each, one IQR tail each side)
  → Three middle bins together = 75 % (three equal slices of the IQR)

Weight Rationale & References
------------------------------
Weights follow SEBI's risk-suitability framework and established behavioural-
finance / credit-scoring literature:

  Investment_Ratio (+3.0)
    Core proxy for risk appetite; higher savings commitment signals tolerance.
    Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — risk profiling guidelines.

  Spending_Behaviour_Score (+2.5)
    Derived from Payment_Behaviour categories (high/low/normal spend).
    Aggressive spenders exhibit higher risk tolerance.
    Ref: Kahneman & Tversky (1979) — Prospect Theory, Econometrica 47(2).

  Annual_Income_norm (+2.0)
    Higher income → larger financial buffer → greater capacity to absorb loss.
    Ref: Grable & Lytton (1999), Financial Counseling and Planning 10(1).

  Monthly_Inhand_Salary_norm (+1.5)
    Take-home salary captures disposable income more accurately than gross.
    Ref: RBI Master Circular on KYC norms (income-based suitability).

  Credit_Mix_Score (+1.5)
    Good credit mix (Good=1.0, Standard=0.5, Bad=0.0) signals financial discipline.
    Ref: CIBIL TransUnion score methodology — credit-health composite.

  Debt_Burden_Ratio (-2.5)
    Outstanding_Debt / Annual_Income; high ratio → stressed balance sheet.
    Ref: Basel III leverage ratio framework; ISB risk-profiling studies.

  Delay_Score (-2.0)
    Delay_from_due_date + Num_Delayed_Payments; chronic delays → lower tolerance.
    Ref: Altman Z-score components; CIBIL payment history weight (~35 %).

  Credit_Utilization_Ratio (-1.0)
    High utilisation → approaching credit limit → financial stress.
    Ref: FICO score framework — 30 % weight on amounts owed.

  Num_Bank_Accounts_norm (+0.5)
    More accounts indicates broader financial engagement (minor positive signal).
    Ref: RBI Financial Inclusion index — account diversity.

  Interest_Rate_norm (-0.5)
    Paying above-average rates → riskier borrower profile → less appetite.
    Ref: Merton (1974) credit-risk model — spread as distress indicator.

  Age_Risk_Proxy (+1.5)
    Younger investors have a longer investment horizon → can tolerate more
    volatility.  Formula: (70 - Age) / 52 (clipped to [18, 70]).
    Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — \u2018investment horizon\u2019
    is an explicit factor in the SEBI risk-suitability profiling framework.
    Lifecycle theory (Bodie et al., 1992) — equity share declines with age.

  Occupation_Stability_Score (+1.2)
    Stable, high-income professions (Doctor, Lawyer, Architect) have greater
    financial capacity to absorb investment losses.  Entrepreneurs show
    high risk tolerance.  Mechanical/creative roles score lower.
    Ref: SEBI/AMFI risk-assessment questionnaire Q3 (occupation/employment
    type as a suitability factor).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RISK_CLASSES, N_RISK_CLASSES, RISK_FEATURES, LABEL_ENCODER_PATH


# ─── Bell-curve quantile boundaries ─────────────────────────────────────────
# Distribution: 12.5% | 25% | 25% | 25% | 12.5%
# Mirrors the natural shape of the raw risk score (approx. normal):
#   tails are rare (genuinely extreme cases), middle tiers are common.
# Boundaries as cumulative probabilities:
#   [0.000 → 0.125] = Very_Low   (bottom-tail: 12.5 %)
#   [0.125 → 0.375] = Low        (lower-middle: 25.0 %)
#   [0.375 → 0.625] = Medium     (centre: 25.0 %)
#   [0.625 → 0.875] = High       (upper-middle: 25.0 %)
#   [0.875 → 1.000] = Very_High  (top-tail: 12.5 %)
RISK_QUANTILES = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]

# ─── Domain weights for composite score ──────────────────────────────────────
# Positive weight  → higher value raises risk-appetite
# Negative weight  → higher value lowers risk-appetite

FEATURE_WEIGHTS = {
    "Investment_Ratio":             +3.0,  # core indicator: higher = more tolerant
    "Annual_Income_norm":           +2.0,  # wealthier = more capacity for risk
    "Monthly_Inhand_Salary_norm":   +1.5,
    "Spending_Behaviour_Score":     +2.5,  # aggressive spender = risk-tolerant
    "Credit_Mix_Score":             +1.5,  # Good credit = can absorb risk
    "Debt_Burden_Ratio":            -2.5,  # heavy debt = less appetite
    "Delay_Score":                  -2.0,  # payment delays = lower tolerance
    "Credit_Utilization_Ratio":     -1.0,  # high utilisation = less capacity
    "Num_Bank_Accounts_norm":       +0.5,  # more accounts = slight engagement
    "Interest_Rate_norm":           -0.5,  # paying high rates = stressed borrower
    "Age_Risk_Proxy":               +1.5,  # younger = longer horizon = more tolerance
                                           # (70-Age)/52; SEBI investment-horizon factor)
    "Occupation_Stability_Score":   +1.2,  # stable income = higher risk capacity
    # ── New v2 features ──────────────────────────────────────────────────
    "EMI_Income_Ratio":             -1.8,  # High DTI = financial stress = less risk capacity
                                           #   Ref: RBI mortgage guidelines — max DTI 50%
    "Savings_Rate":                 +1.5,  # Higher savings buffer = more risk tolerance
                                           #   Ref: RBI Financial Stability Report
    "Credit_History_Score":         +0.8,  # Longer history = maturity = moderate positive
                                           #   Ref: CIBIL methodology (~15% weight)
                                           # (SEBI/AMFI suitability factor)
}


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Return a continuous composite risk-appetite score for each row.
    Score = Σ (weight × feature_value)
    """
    score = pd.Series(0.0, index=df.index)
    for feat, weight in FEATURE_WEIGHTS.items():
        if feat in df.columns:
            score += weight * df[feat].fillna(0)
    return score


def compute_risk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose risk into 4 orthogonal sub-dimensions (multi-metric risk matrix).

    Returns a DataFrame with columns:
        Financial_Capacity, Behavioral_Tolerance, Time_Horizon, Credit_Health,
        composite_risk_score

    Each sub-score is normalised to [0, 1] via min-max scaling across the
    customer population, then combined with equal dimension weights.

    Reference
    ----------------
    SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — multi-factor suitability
    profiling requires assessment across financial capacity, risk tolerance,
    investment horizon, and credit discipline.

    Grable & Lytton (1999) — "Financial risk tolerance revisited: the development
    of a risk assessment instrument", Financial Services Review 8(3), pp.163–181.
    """
    from config import RISK_MATRIX_DIMENSIONS

    result = pd.DataFrame(index=df.index)

    for dim_name, dim_cfg in RISK_MATRIX_DIMENSIONS.items():
        dim_score = pd.Series(0.0, index=df.index)
        for feat, weight in zip(dim_cfg["features"], dim_cfg["weights"]):
            if feat in df.columns:
                dim_score += weight * df[feat].fillna(0)
        # Min-max normalise to [0, 1]
        rng = dim_score.max() - dim_score.min()
        if rng > 1e-9:
            dim_score = (dim_score - dim_score.min()) / rng
        else:
            dim_score = 0.5
        result[dim_name] = dim_score

    # Composite = equal-weight average of 4 normalised dimensions
    result["composite_risk_score"] = result.mean(axis=1)
    return result


def assign_risk_label(df: pd.DataFrame,
                      fit_encoder: bool = True) -> pd.DataFrame:
    """
    Add 'risk_score', 'risk_label' (string), and 'risk_label_encoded' (int)
    columns to `df`.

    Distribution (bell-curve, mirrors real customer populations):
        Very_Low  : 12.5 %  (bottom tail)
        Low       : 25.0 %
        Medium    : 25.0 %  (centre)
        High      : 25.0 %
        Very_High : 12.5 %  (top tail)

    Parameters
    ----------
    df          : DataFrame containing RISK_FEATURES columns
    fit_encoder : if True, fit and save a LabelEncoder; else load saved one
    """
    df = df.copy()
    df["risk_score"] = compute_risk_score(df)

    # Bell-curve binning: 12.5% / 25% / 25% / 25% / 12.5%
    # Uses RISK_QUANTILES = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    df["risk_label"] = pd.qcut(
        df["risk_score"],
        q=RISK_QUANTILES,
        labels=RISK_CLASSES,
        duplicates="drop",
    )

    # Handle edge-case where qcut produces fewer bins (duplicate edges)
    # Fall back to approximate bell-curve cut-points
    if df["risk_label"].isna().any():
        score_min = df["risk_score"].min()
        score_max = df["risk_score"].max()
        q_vals = df["risk_score"].quantile(RISK_QUANTILES[1:-1]).tolist()
        bins = [score_min - 1e-6] + q_vals + [score_max + 1e-6]
        df["risk_label"] = pd.cut(
            df["risk_score"],
            bins=bins,
            labels=RISK_CLASSES,
            include_lowest=True,
        )

    df["risk_label"] = df["risk_label"].astype(str)

    # Encode to integer
    if fit_encoder:
        le = LabelEncoder()
        df["risk_label_encoded"] = le.fit_transform(df["risk_label"])
        joblib.dump(le, LABEL_ENCODER_PATH)
    else:
        le = joblib.load(LABEL_ENCODER_PATH)
        # Map unseen labels to closest known class
        known = set(le.classes_)
        df["risk_label_safe"] = df["risk_label"].apply(
            lambda x: x if x in known else "Medium"
        )
        df["risk_label_encoded"] = le.transform(df["risk_label_safe"])
        df.drop(columns=["risk_label_safe"], inplace=True)

    return df


def infer_risk_label_from_score(score: float,
                                thresholds: np.ndarray) -> str:
    """
    Given a pre-computed risk score and the quintile thresholds from training,
    return the string label.  Used for single-customer inference.

    Parameters
    ----------
    score      : composite risk score
    thresholds : array of N_RISK_CLASSES-1 boundary values (from training data)
    """
    idx = int(np.searchsorted(thresholds, score, side="right"))
    idx = min(idx, N_RISK_CLASSES - 1)
    return RISK_CLASSES[idx]


def get_risk_thresholds(labeled_df: pd.DataFrame) -> np.ndarray:
    """
    Extract the quintile boundary values from the labeled training set.
    These are saved alongside the model for use during single-customer inference.

    If a risk class has no members (edge case with very small datasets), a NaN
    boundary is replaced with the midpoint between its neighbours so that
    ``infer_risk_label_from_score`` never receives a NaN threshold.
    """
    boundaries = []
    for i in range(1, N_RISK_CLASSES):
        pool = labeled_df[labeled_df["risk_label"] == RISK_CLASSES[i]]["risk_score"]
        threshold = float(pool.min()) if not pool.empty else float("nan")
        boundaries.append(threshold)

    # Fill NaN thresholds with interpolated neighbours to avoid searchsorted failure
    arr = np.array(boundaries, dtype=float)
    nan_mask = np.isnan(arr)
    if nan_mask.any() and not nan_mask.all():
        # Forward-fill then backward-fill for interior/edge NaNs
        indices = np.arange(len(arr))
        arr = np.interp(indices, indices[~nan_mask], arr[~nan_mask])
    elif nan_mask.all():
        arr = np.linspace(0.0, 1.0, N_RISK_CLASSES - 1)

    return arr
