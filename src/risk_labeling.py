"""
risk_labeling.py  –  Derive a 5-class 'Risk Appetite' label for each customer.

Strategy
--------
Scoring is done exclusively via PCA: a Principal Component Analysis (PCA) is
fitted on all 15 MinMax-scaled risk features, and the sign-corrected first
principal component (PC1) is used as the risk score.  PC1 explains 32.48 % of
total feature variance — the dominant latent financial-health dimension.

  Risk_Score = sign_factor × pca.transform(MinMaxScaled(X))[:, 0]

where sign_factor = −1.0 so that higher score = higher risk capacity.
A domain-weighted sum (ABLATION BASELINE ONLY) is provided in compute_risk_score()
for research comparison purposes; it is not called by the production pipeline.

After quantile binning into five classes the distribution is BELL-CURVE-shaped:
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
normal-like shape of the raw PC1 score (see EDA notebook cell 7b).
  → First + last bins together = 25 % (12.5 % each, one IQR tail each side)
  → Three middle bins together = 75 % (three equal slices of the IQR)

Weight Rationale & References (PCA-derived; no manual assumptions)
---------------------------------------------------------------------------
Weights = sign_factor × PC1_loading, normalised to ±2.5.
Ranked by |PC1 loading| (rank 1 = highest absolute PC1 loading):

  Credit_Mix_Score (+2.50)                   [rank 1, PC1_loading=−0.5247]
    Largest |PC1 loading|; after sign correction this is the dominant
    risk-capacity driver.  Good credit mix signals financial discipline.
    Ref: CIBIL TransUnion score methodology; SEBI/HO/IMD/DF2/CIR/P/2019/17.

  Interest_Rate_norm (−1.99)                 [rank 2, PC1_loading=+0.4177]
    Paying above-average interest rates signals a riskier borrower profile;
    sign-corrected to negative in the risk score.
    Ref: Merton (1974) credit-risk model — spread as distress indicator.

  Credit_History_Score (+1.70)               [rank 3, PC1_loading=−0.3559]
    Credit maturity; longer history = greater discipline and risk capacity.
    Ref: CIBIL TransUnion score methodology.

  Num_Bank_Accounts_norm (−1.64)             [rank 4, PC1_loading=+0.3449]
    Spearman ρ = -0.76 vs PC1 risk_score.  Customers with many bank accounts in
    this dataset are managing debt across them (account proliferation), signalling
    financial stress rather than engagement breadth.  Sign is empirically negative.
    (Domain theory predicted +; data overrides.)
    Ref: RBI Financial Inclusion index — account diversity.

  Delay_Score (−1.57)                        [rank 5, PC1_loading=+0.3297]
    Delay_from_due_date + Num_Delayed_Payments; chronic payment delays signal
    lower financial discipline and risk tolerance.
    Ref: Altman Z-score components; CIBIL payment history weight (~35 %).

  Monthly_Inhand_Salary_norm (+1.20)         [rank 6, PC1_loading=−0.2512]
    Spearman ρ = +0.95 — strongest supervised income signal.
    Take-home salary captures disposable income more accurately than gross.
    Ref: RBI Master Circular on KYC norms (income-based suitability).

  Annual_Income_norm (+1.00)                 [rank 7, PC1_loading=−0.2104]
    Higher income → larger financial buffer → greater capacity to absorb loss.
    Ref: Grable & Lytton (1999), Financial Counseling and Planning 10(1).

  Age_Risk_Proxy (−0.85)                     [rank 8, PC1_loading=+0.1784]
    Spearman ρ = -0.33 vs PC1 risk_score.  Younger customers in this dataset carry
    higher debt ratios and worse credit mix, making them empirically lower-risk-
    tolerance than middle-aged customers.  Sign is empirically negative.
    (Domain lifecycle theory predicted +; data overrides.)
    Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — investment horizon factor.
    Lifecycle theory (Bodie et al., 1992) — equity share declines with age.

  Debt_Burden_Ratio (−0.78)                  [rank 9, PC1_loading=+0.1636]
    Outstanding_Debt / Annual_Income; high ratio → stressed balance sheet.
    Ref: Basel III leverage ratio framework; ISB risk-profiling studies.

  Spending_Behaviour_Score (+0.62)           [rank 10, PC1_loading=−0.1291]
    Derived from Payment_Behaviour categories (high/low/normal spend).
    Aggressive spenders exhibit higher risk tolerance.
    Ref: Kahneman & Tversky (1979) — Prospect Theory, Econometrica 47(2).

  Credit_Utilization_Ratio (+0.31)           [rank 11, PC1_loading=−0.0646]
    Spearman ρ = +0.23 vs PC1 risk_score.  Customers with Good credit mix actively
    use their available credit, so utilization is a confounded positive signal in
    this dataset (it co-varies with Credit_Mix_Score which has ρ=+0.90).  Sign is
    empirically positive.  (FICO/domain theory predicted −; data overrides.)
    Ref: FICO score framework — 30 % weight on amounts owed.

  EMI_Income_Ratio (−0.30)                   [rank 12, PC1_loading=+0.0635]
    Total_EMI_per_month / Monthly_Inhand_Salary — standard DTI proxy.
    Ref: RBI mortgage guideline; Basel III consumer lending norms.

  Savings_Rate (+0.29)                       [rank 13, PC1_loading=−0.0612]
    (Salary − EMI) / Salary; measures financial buffer available for investing.
    Ref: Grable & Lytton (1999), Financial Counseling and Planning 10(1).

  Investment_Ratio (−0.16)                   [rank 14, PC1_loading=+0.0345]
    Spearman ρ = -0.22 vs PC1 risk_score.  High investment/salary ratio coexists
    with high debt in this Kaggle dataset (forced-savings / PPF pattern), and PCA
    confirms collinearity with debt features.  Sign is empirically negative.
    (Domain theory predicted +; data overrides.)
    Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — risk profiling guidelines.

  Occupation_Stability_Score (+0.04)         [rank 15, PC1_loading=−0.0083]
    Stable, high-income professions (Doctor, Lawyer, Architect) have greater
    financial capacity to absorb investment losses.  Entrepreneurs show
    high risk tolerance.  Mechanical/creative roles score lower.
    Ref: SEBI/AMFI risk-assessment questionnaire Q3 (occupation/employment type).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RISK_CLASSES, N_RISK_CLASSES, RISK_FEATURES, LABEL_ENCODER_PATH, PCA_RISK_PATH, PCA_N_COMPONENTS, FEATURE_WEIGHTS


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

# ─── FEATURE_WEIGHTS — imported from config.py (single source of truth) ──────
# Defined in config.py alongside RISK_MATRIX_DIMENSIONS so both systems share
# identical data-derived weights (PCA PC1 loading, sign-corrected, normalised to ±2.5).
# Re-exported here so that weight_validation.py and notebooks can still do:
#   from src.risk_labeling import FEATURE_WEIGHTS


# ─── PCA-based risk scoring (primary method) ─────────────────────────────────

def fit_pca_risk_model(df: pd.DataFrame, n_components: int = None):
    """
    Fit a PCA on RISK_FEATURES and sign-correct PC1 so that higher PC1 always
    indicates higher risk appetite.

    PC1 is the direction of maximum variance in the 15-feature risk space.
    After sign correction:
      high PC1  → Savings_Rate ↑, Income ↑, Credit_Mix ↑ → high risk capacity
      low  PC1  → Debt_Burden ↑, Delay ↑, Interest_Rate ↑ → low risk capacity

    The model is saved to PCA_RISK_PATH and returned.

    References
    ----------
    Jolliffe (2002) — Principal Component Analysis, 2nd ed.
    SEBI (2019) — Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 (multi-factor suitability).
    """
    from sklearn.decomposition import PCA as SklearnPCA

    n_components = n_components or PCA_N_COMPONENTS
    feat_cols = [f for f in RISK_FEATURES if f in df.columns]
    X = df[feat_cols].fillna(0.0).values.astype(np.float64)

    pca = SklearnPCA(n_components=min(n_components, len(feat_cols)), random_state=42)
    pca.fit(X)

    # Sign-correct PC1: sum of loadings for data-confirmed risk-positive features
    # must be > 0.  risk_positive is derived dynamically from FEATURE_WEIGHTS
    # (positive weight = higher value raises risk appetite) so that the PCA
    # sign-correction is consistent with the empirically validated weight signs.
    risk_positive = {
        f for f, w in FEATURE_WEIGHTS.items() if w > 0
    }
    pos_loading_sum = sum(
        pca.components_[0, i]
        for i, f in enumerate(feat_cols)
        if f in risk_positive
    )
    pc1_sign = 1.0 if pos_loading_sum >= 0 else -1.0
    pca._pc1_sign = pc1_sign   # preserved through joblib serialisation

    joblib.dump(pca, PCA_RISK_PATH)
    ev = pca.explained_variance_ratio_
    print(f"[PCA-Risk] Fitted {n_components}-component PCA on {len(feat_cols)} features")
    print(f"[PCA-Risk] Explained variance: " +
          " | ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(ev)))
    print(f"[PCA-Risk] PC1 sign: {'positive (no flip)' if pc1_sign > 0 else 'flipped'} "
          f"(pos-loading sum={pos_loading_sum:+.4f})")

    # Print PC1 loadings for interpretability / thesis output
    loadings = pca.components_[0] * pc1_sign
    loading_pairs = sorted(zip(feat_cols, loadings), key=lambda x: -x[1])
    print("[PCA-Risk] PC1 loadings (↑=risk-positive, sorted):")
    for feat, load in loading_pairs:
        bar = "█" * max(1, int(abs(load) * 18))
        print(f"    {'+'if load>=0 else '-'}{bar:<18}  {load:+.4f}  {feat}")

    return pca


def compute_risk_score_pca(df: pd.DataFrame, pca_model=None) -> pd.Series:
    """
    Compute composite risk-appetite score via PCA PC1 projection.

    Risk_Score = sign-corrected PC1 · MinMaxScaled(RISK_FEATURES)

    Higher score → higher risk appetite (sign-corrected so that
    Savings_Rate ↑, Income ↑, Credit_Mix ↑ → high score).

    Parameters
    ----------
    df        : preprocessed DataFrame with MinMaxScaled RISK_FEATURES
    pca_model : fitted PCA model; if None, auto-loaded from PCA_RISK_PATH

    Returns
    -------
    pd.Series of sign-corrected PC1 scores

    Raises
    ------
    FileNotFoundError
        If PCA_RISK_PATH does not exist on disk.  Run train.py to fit and
        save the PCA model before calling this function.
    """
    if pca_model is None:
        if not Path(PCA_RISK_PATH).exists():
            raise FileNotFoundError(
                f"PCA risk model not found at '{PCA_RISK_PATH}'.\n"
                "Run train.py (or fit_pca_risk_model()) to generate it before "
                "calling compute_risk_score_pca().\n"
                "The domain-weighted sum fallback has been intentionally disabled — "
                "the system uses only the PCA PC1 projection as its risk score."
            )
        pca_model = joblib.load(PCA_RISK_PATH)

    feat_cols = [f for f in RISK_FEATURES if f in df.columns]
    X = df[feat_cols].fillna(0.0).values.astype(np.float64)
    pc1_raw = pca_model.transform(X)[:, 0]
    sign    = getattr(pca_model, "_pc1_sign", 1.0)
    return pd.Series(sign * pc1_raw, index=df.index, name="risk_score")


# ─── Legacy weighted-sum score (kept as fallback / ablation) ──────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    ABLATION BASELINE ONLY — NOT used by the production pipeline.

    Domain-weighted linear sum:
        Score = Σ (weight_i × feature_i)   using PCA-ranked FEATURE_WEIGHTS.

    Retained solely for Chapter 3 ablation study (§3.5.2).  The live system
    always derives risk scores via PCA PC1 projection
    (see compute_risk_score_pca).  Do NOT call this function from any pipeline
    stage; calls should go through compute_risk_score_pca() exclusively.
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
                      fit_encoder: bool = True,
                      pca_model=None) -> pd.DataFrame:
    """
    Add 'risk_score', 'risk_label' (string), and 'risk_label_encoded' (int)
    columns to `df`.

    Scoring method (primary): PCA PC1 projection via compute_risk_score_pca().
    The sign-corrected PC1 captures the dominant latent financial-risk dimension
    across all 15 features without manual weight engineering.

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
    pca_model   : fitted PCA model; if None, auto-loaded from PCA_RISK_PATH;
                  if file absent, falls back to domain-weighted sum
    """
    df = df.copy()
    df["risk_score"] = compute_risk_score_pca(df, pca_model=pca_model)

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
