"""
preprocessing.py  –  Data loading, cleaning, and feature engineering
for Smart Fund Advisor.

Pipeline
--------
1.  Load raw bank CSV (50 k rows, 27 cols).
2.  Clean / coerce noisy numeric columns (trailing _, leading spaces, etc.).
3.  Aggregate per customer → one representative record per user.
4.  Feature-engineer the 10 risk input features defined in config.RISK_FEATURES.
5.  Return a clean DataFrame ready for risk labelling & model training.
"""

import re
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BANK_CSV, SCALER_PATH, RISK_FEATURES, RANDOM_SEED
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric garbage, coerce to float, fill NaN with median."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", np.nan)
    )
    cleaned = pd.to_numeric(cleaned, errors="coerce")
    median_val = cleaned.median()
    return cleaned.fillna(median_val)


def _encode_credit_mix(series: pd.Series) -> pd.Series:
    mapping = {"good": 1.0, "standard": 0.5, "bad": 0.0}
    return series.str.strip().str.lower().map(mapping).fillna(0.5)


def _encode_occupation(series: pd.Series) -> pd.Series:
    """
    Map Occupation to a financial risk-capacity score [0, 1].

    Rationale
    ---------
    Two dimensions drive this score:
      1. Income stability  — how predictable is the earnings stream?
      2. Risk tolerance    — profession-level propensity to take financial risk.

    High capacity  : Doctor, Lawyer, Architect — stable + high income.
    High tolerance : Entrepreneur — business owner, accustomed to uncertainty.
    Mid-high       : Engineer, Scientist, Developer — specialised, in-demand.
    Mid            : Manager, Accountant, Media_Manager — moderate stability.
    Moderate       : Teacher, Journalist, Writer, Musician — lower variance income.
    Manual/Other   : Mechanic — lower disposable surplus for risk.
    Unknown noise  : '______' fallback — neutral 0.55.

    References
    ----------
    SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — suitability includes
    \'employment type\' as an explicit risk-profiling factor.
    AMFI risk-assessment questionnaire, Q3 (occupation/employment).
    """
    mapping = {
        # High risk capacity (stable high income)
        "doctor":        0.90,
        "lawyer":        0.88,
        "architect":     0.85,
        # High risk tolerance (entrepreneurial / business owner)
        "entrepreneur":  0.88,
        # Mid-high (specialised, in-demand)
        "engineer":      0.78,
        "scientist":     0.75,
        "developer":     0.73,
        # Moderate-high
        "manager":       0.68,
        "accountant":    0.65,
        "media_manager": 0.62,
        # Moderate
        "teacher":       0.58,
        "journalist":    0.55,
        "writer":        0.52,
        "musician":      0.50,
        # Manual / lower surplus
        "mechanic":      0.48,
    }
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(mapping).fillna(0.55)   # 0.55 for noise / unknown


def _encode_payment_behaviour(series: pd.Series) -> pd.Series:
    """
    Payment_Behaviour values like:
      High_spent_Large_value_payments   → 1.0  (aggressive spender)
      High_spent_Medium_value_payments  → 0.8
      High_spent_Small_value_payments   → 0.6
      Low_spent_Large_value_payments    → 0.5
      Low_spent_Medium_value_payments   → 0.3
      Low_spent_Small_value_payments    → 0.1  (very conservative)
    """
    mapping = {
        "high_spent_large_value_payments":  1.00,
        "high_spent_medium_value_payments": 0.80,
        "high_spent_small_value_payments":  0.60,
        "low_spent_large_value_payments":   0.50,
        "low_spent_medium_value_payments":  0.30,
        "low_spent_small_value_payments":   0.10,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0.3)
    )


def _parse_credit_history_months(series: pd.Series) -> pd.Series:
    """'22 Years and 9 Months' → 22*12 + 9 = 273"""
    def parse_one(val):
        val = str(val)
        y_match = re.search(r"(\d+)\s*[Yy]ear", val)
        m_match = re.search(r"(\d+)\s*[Mm]onth", val)
        years   = int(y_match.group(1)) if y_match else 0
        months  = int(m_match.group(1)) if m_match else 0
        total   = years * 12 + months
        return total if total > 0 else np.nan
    return series.apply(parse_one)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def load_and_clean(csv_path=None) -> pd.DataFrame:
    """
    Load raw bank CSV, clean all noisy columns, return a per-row DataFrame.
    Does NOT yet aggregate per customer.
    """
    csv_path = csv_path or BANK_CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]

    # ── Numeric columns that may contain noise (underscores, spaces, etc.) ──
    num_cols = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    # ── Credit_History_Age  →  total months (numeric) ──
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Months"] = _parse_credit_history_months(
            df["Credit_History_Age"]
        )
        median_hist = df["Credit_History_Months"].median()
        df["Credit_History_Months"] = df["Credit_History_Months"].fillna(median_hist)

    # ── Categorical encodings (used in feature engineering) ──
    df["Credit_Mix_Score"]          = _encode_credit_mix(df.get("Credit_Mix", pd.Series(["standard"]*len(df))))
    df["Spending_Behaviour_Score"]  = _encode_payment_behaviour(df.get("Payment_Behaviour", pd.Series(["low_spent_medium_value_payments"]*len(df))))
    df["Occupation_Stability_Score"] = _encode_occupation(df.get("Occupation", pd.Series(["unknown"]*len(df))))

    # ── Clip outliers using 1st–99th percentile ──
    for col in ["Annual_Income", "Monthly_Inhand_Salary",
                "Outstanding_Debt", "Amount_invested_monthly",
                "Credit_Utilization_Ratio"]:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    return df


def aggregate_per_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one record per Customer_ID by averaging numeric columns
    and taking the mode for categoricals.
    This mirrors what a mobile device would compute from its ≤4 local records.
    """
    agg_dict = {
        "Age":                       "mean",
        "Annual_Income":             "mean",
        "Monthly_Inhand_Salary":     "mean",
        "Num_Bank_Accounts":         "mean",
        "Num_Credit_Card":           "mean",
        "Interest_Rate":             "mean",
        "Num_of_Loan":               "mean",
        "Delay_from_due_date":       "mean",
        "Num_of_Delayed_Payment":    "mean",
        "Changed_Credit_Limit":      "mean",
        "Num_Credit_Inquiries":      "mean",
        "Outstanding_Debt":          "mean",
        "Credit_Utilization_Ratio":  "mean",
        "Total_EMI_per_month":       "mean",
        "Amount_invested_monthly":   "mean",
        "Monthly_Balance":           "mean",
        "Credit_Mix_Score":          "mean",
        "Spending_Behaviour_Score":  "mean",
        "Occupation_Stability_Score":"mean",  # stable across months; mean = mode-equivalent
        "Credit_History_Months":     "mean",  # credit maturity (v2 feature)
    }
    # Only keep columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    agg_df = df.groupby("Customer_ID").agg(agg_dict).reset_index()
    return agg_df


def engineer_features(agg_df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
    """
    Build the 10 engineered feature columns listed in config.RISK_FEATURES.

    Parameters
    ----------
    agg_df      : per-customer aggregated DataFrame
    fit_scaler  : if True fit a new MinMaxScaler and save it;
                  if False load the saved scaler (inference mode)
    """
    df = agg_df.copy()

    # ── Ratios ──
    eps = 1e-6
    df["Investment_Ratio"] = (
        df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + eps)
    ).clip(0, 5)

    df["Debt_Burden_Ratio"] = (
        df["Outstanding_Debt"] / (df["Annual_Income"] + eps)
    ).clip(0, 5)

    df["Delay_Score"] = (
        df["Delay_from_due_date"].fillna(0) +
        df["Num_of_Delayed_Payment"].fillna(0)
    )

    # ── Age risk proxy: younger = longer investment horizon = higher tolerance ──
    # Age is cleaned in load_and_clean (noisy values like -500 / 914 coerced);
    # clip to [18, 70] to handle any residual outliers before computing proxy.
    # Formula: (70 - Age) / 52  →  age 22 ≈ 0.923 (high tolerance),
    #                             age 70 =  0.000 (pre-retirement, low tolerance).
    # Ref: SEBI CIRCULAR SEBI/HO/IMD/DF2/CIR/P/2019/17 — investment horizon
    #      is an explicit factor in risk-suitability classification.
    age_col = df.get("Age", pd.Series(35.0, index=df.index)).fillna(35.0)
    age_clean = pd.to_numeric(age_col, errors="coerce").fillna(35.0).clip(18, 70)
    df["Age_Risk_Proxy"] = ((70.0 - age_clean) / 52.0).clip(0.0, 1.0)

    # ── NEW v2 features: production-realistic financial ratios ─────────────

    # EMI_Income_Ratio: Total_EMI / Monthly_Inhand_Salary
    # Standard DTI metric used in credit underwriting (RBI / Basel III).
    # High DTI → less capacity for risk; low DTI → more disposable income for investing.
    df["EMI_Income_Ratio"] = (
        df["Total_EMI_per_month"].fillna(0) / (df["Monthly_Inhand_Salary"] + eps)
    ).clip(0, 3)

    # Savings_Rate: (Salary - EMI) / Salary — financial buffer indicator
    # Higher savings rate → more capacity to absorb investment volatility.
    # Ref: RBI Financial Stability Report — household savings rate as risk indicator.
    salary = df["Monthly_Inhand_Salary"].fillna(0)
    emi    = df["Total_EMI_per_month"].fillna(0)
    df["Savings_Rate"] = ((salary - emi) / (salary + eps)).clip(-1, 1)

    # Credit_History_Score: normalised credit history months
    # Longer credit history → more mature/reliable borrower.
    # Ref: CIBIL score methodology — credit age carries ~15% weight.
    if "Credit_History_Months" in df.columns:
        df["Credit_History_Score"] = df["Credit_History_Months"].fillna(0)
    else:
        df["Credit_History_Score"] = 0.0

    # ── Pass-through columns ──
    passthrough = [
        "Annual_Income", "Monthly_Inhand_Salary",
        "Credit_Utilization_Ratio", "Num_Bank_Accounts", "Interest_Rate",
        "Credit_Mix_Score", "Spending_Behaviour_Score",
        "Occupation_Stability_Score",
    ]
    for col in passthrough:
        if col not in df.columns:
            df[col] = 0.0

    # ── Scale to [0, 1] ──
    raw_cols = [
        "Annual_Income", "Monthly_Inhand_Salary",
        "Investment_Ratio", "Debt_Burden_Ratio",
        "Credit_Utilization_Ratio", "Delay_Score",
        "Credit_Mix_Score", "Spending_Behaviour_Score",
        "Num_Bank_Accounts", "Interest_Rate",
        "Age_Risk_Proxy",
        "Occupation_Stability_Score",
        "EMI_Income_Ratio",
        "Savings_Rate",
        "Credit_History_Score",
    ]
    target_names = [
        "Annual_Income_norm", "Monthly_Inhand_Salary_norm",
        "Investment_Ratio", "Debt_Burden_Ratio",
        "Credit_Utilization_Ratio", "Delay_Score",
        "Credit_Mix_Score", "Spending_Behaviour_Score",
        "Num_Bank_Accounts_norm", "Interest_Rate_norm",
        "Age_Risk_Proxy",
        "Occupation_Stability_Score",
        "EMI_Income_Ratio",
        "Savings_Rate",
        "Credit_History_Score",
    ]

    if fit_scaler:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[raw_cols].fillna(0))
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        scaled = scaler.transform(df[raw_cols].fillna(0))

    for i, name in enumerate(target_names):
        df[name] = scaled[:, i]

    return df


def get_clean_customer_data(fit_scaler: bool = True) -> pd.DataFrame:
    """
    Full pipeline:  raw CSV  →  cleaned rows  →  per-customer aggregate  →  features.
    Returns a DataFrame with Customer_ID + RISK_FEATURES columns.
    """
    df_raw  = load_and_clean()
    df_agg  = aggregate_per_customer(df_raw)
    df_feat = engineer_features(df_agg, fit_scaler=fit_scaler)
    return df_feat


# ─── Device-side helper ───────────────────────────────────────────────────────

def get_device_records(df_raw: pd.DataFrame, customer_id: str,
                       max_records: int = 4) -> pd.DataFrame:
    """
    Simulate what a mobile device holds for a given customer:
    the most-recent `max_records` monthly rows, cleaned but NOT aggregated.
    Used during federated training to mimic on-device data.
    """
    cdf = df_raw[df_raw["Customer_ID"] == customer_id].tail(max_records).copy()
    cdf["Credit_Mix_Score"]           = _encode_credit_mix(cdf.get("Credit_Mix", pd.Series(["standard"]*len(cdf))))
    cdf["Spending_Behaviour_Score"]   = _encode_payment_behaviour(cdf.get("Payment_Behaviour", pd.Series(["low_spent_medium_value_payments"]*len(cdf))))
    cdf["Occupation_Stability_Score"] = _encode_occupation(cdf.get("Occupation", pd.Series(["other"]*len(cdf))))
    return cdf
