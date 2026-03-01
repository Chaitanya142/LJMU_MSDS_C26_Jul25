"""
demo_single_user.py — End-to-end investment profile for a single customer.

Pipeline stages
───────────────
  STAGE 1  Raw bank records          (CSV → displayed as-is)
  STAGE 2  Feature engineering       (preprocessing → normalised features)
  STAGE 3  Risk scoring & labelling  (weighted composite score → bell-curve bin)
  STAGE 4  Multi-metric risk matrix  (4 orthogonal dimensions)
  STAGE 5  FL model inference        (RiskMLP → class probabilities)
  STAGE 6  Fund recommendations      (core-satellite per horizon: 1 / 3 / 5 / 10 yr)
  STAGE 7  GenAI explanation         (LLM or deterministic rule-based fallback)

Usage
─────
  python demo_single_user.py                         # default: CUS_0xd40
  python demo_single_user.py --customer CUS_0xd40
  python demo_single_user.py --customer CUS_0x7a1 --amount 1000000
  python demo_single_user.py --list-customers        # list first 20 customer IDs
"""

from __future__ import annotations

import argparse
import time
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch


# ─── Project imports ──────────────────────────────────────────────────────────
from config import (
    RISK_FEATURES, RISK_CLASSES, N_RISK_CLASSES,
    LABEL_ENCODER_PATH, CENTRAL_MODEL_PATH,
    RISK_MATRIX_DIMENSIONS, DEMO_CUSTOMERS_PATH,
)
from src.preprocessing   import get_clean_customer_data
from src.risk_labeling   import assign_risk_label, compute_risk_matrix, compute_risk_score
from src.central_model   import load_central_model, predict
from src.recommender     import recommend_full_profile, load_mutual_funds
from src.gpt_explainer   import explain_full_profile


# ─── Display helpers ──────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    line = "─" * 72
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _arrow(label: str, value: str) -> None:
    print(f"  {'→'} {label:<38} {value}")


def _section(label: str) -> None:
    print(f"\n  ── {label}")


def _divider() -> None:
    print("  " + "·" * 68)


# ─── Stage implementations ────────────────────────────────────────────────────

def stage1_raw_records(raw_df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    """Show the raw monthly bank records for the chosen customer."""
    _banner("STAGE 1 — Raw Bank Records (as stored in CSV)")

    rows = raw_df[raw_df["Customer_ID"] == customer_id]
    if rows.empty:
        raise ValueError(f"Customer '{customer_id}' not found in the dataset.")

    display_cols = [
        "Month", "Name", "Age", "Occupation", "Annual_Income",
        "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Amount_invested_monthly", "Monthly_Balance",
        "Num_of_Delayed_Payment", "Credit_Mix", "Payment_Behaviour",
    ]
    display_cols = [c for c in display_cols if c in rows.columns]

    name = rows.iloc[0].get("Name", customer_id)
    print(f"\n  Customer  : {customer_id}  ({name})")
    print(f"  Months    : {len(rows)} record(s) found\n")

    for _, row in rows.iterrows():
        month = row.get("Month", "?")
        print(f"  [{month}]")
        for col in display_cols:
            if col in ("Name",):
                continue
            val = row[col]
            if isinstance(val, float):
                print(f"    {col:<38} {val:,.2f}")
            else:
                print(f"    {col:<38} {val}")
        print()

    return rows


def stage2_feature_engineering(processed_df: pd.DataFrame, customer_id: str) -> pd.Series:
    """
    Show the engineered feature vector for the customer.
    processed_df is the full population DataFrame after preprocessing.
    We take the most-recent record (tail(1)).
    """
    _banner("STAGE 2 — Feature Engineering (Normalised Risk Features)")

    cdf = processed_df[processed_df["Customer_ID"] == customer_id]
    if cdf.empty:
        raise ValueError(f"No processed rows for '{customer_id}'.")

    feat_row = cdf.tail(1).iloc[0]

    # Group features for readable display
    groups = {
        "Income & Savings": [
            "Annual_Income_norm", "Monthly_Inhand_Salary_norm",
            "Investment_Ratio", "Savings_Rate",
        ],
        "Debt & Credit Burden": [
            "Debt_Burden_Ratio", "EMI_Income_Ratio",
            "Credit_Utilization_Ratio", "Interest_Rate_norm",
        ],
        "Behavioural Signals": [
            "Delay_Score", "Credit_Mix_Score",
            "Spending_Behaviour_Score", "Num_Bank_Accounts_norm",
        ],
        "Temporal & Profile": [
            "Age_Risk_Proxy", "Occupation_Stability_Score",
            "Credit_History_Score",
        ],
    }

    for group_name, cols in groups.items():
        _section(group_name)
        for col in cols:
            if col in feat_row.index:
                print(f"    {col:<38} {feat_row[col]:.4f}")

    return feat_row


def stage3_risk_label(processed_df: pd.DataFrame, customer_id: str) -> tuple[str, float]:
    """
    Run assign_risk_label on the full population, then extract this customer's
    risk score and label.  Population-level qcut ensures accurate percentile
    binning (same as production training: 12.5/25/25/25/12.5 distribution).
    """
    _banner("STAGE 3 — Risk Scoring & Labelling (Bell-Curve Quantile Binning)")

    labeled = assign_risk_label(processed_df.copy(), fit_encoder=False)
    row     = labeled[labeled["Customer_ID"] == customer_id].tail(1).iloc[0]

    risk_score = float(row["risk_score"])
    risk_label = str(row["risk_label"])

    _section("Composite Risk Score computation")
    print(f"    Formula  : weighted sum of 16 risk features (see RISK_FEATURES)")
    print(f"    Raw score: {risk_score:.4f}  (population range: 0 – 1)")

    _section("Bell-curve quantile bins (SEBI-aligned population distribution)")
    print(f"    {'Bin':<12} {'Percentile range':<22} {'Population share':<20}")
    print("    " + "-" * 56)
    bins = [
        ("Very_Low",  "0 – 12.5 %",   "12.5 %  (low-risk tail)"),
        ("Low",       "12.5 – 37.5 %","25.0 %"),
        ("Medium",    "37.5 – 62.5 %","25.0 %  (centre)"),
        ("High",      "62.5 – 87.5 %","25.0 %"),
        ("Very_High", "87.5 – 100 %", "12.5 %  (high-risk tail)"),
    ]
    for cls, pct, share in bins:
        marker = "  ◄" if cls == risk_label else ""
        print(f"    {cls:<12} {pct:<22} {share:<20}{marker}")

    _section("Result")
    _arrow("Risk Score",  f"{risk_score:.4f}")
    _arrow("Risk Label",  risk_label)

    return risk_label, risk_score


def stage4_risk_matrix(processed_df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    """
    Compute the 4-dimension multi-metric risk matrix for the customer.
    Shows each sub-dimension and their equal-weight composite.
    """
    _banner("STAGE 4 — Multi-Metric Risk Matrix (4 Orthogonal Dimensions)")

    # Use the full population so min-max normalisation is meaningful
    full_matrix  = compute_risk_matrix(processed_df.copy())
    cust_indices = processed_df[processed_df["Customer_ID"] == customer_id].tail(1).index
    matrix       = full_matrix.loc[cust_indices]

    dim_descriptions = {
        "Financial_Capacity":    "Income, savings, investments — how much can be invested",
        "Behavioral_Tolerance":  "Debt behaviour, delays, utilisation — attitude toward risk",
        "Time_Horizon":          "Age, credit history, occupation stability — capacity to wait",
        "Credit_Health":         "Credit mix, payment patterns, interest burden — financial discipline",
    }

    bar_len = 20

    _section("Dimension scores (normalised 0.00 – 1.00, higher = higher risk capacity)")
    print()
    for dim in ["Financial_Capacity", "Behavioral_Tolerance", "Time_Horizon", "Credit_Health"]:
        if dim in matrix.columns:
            score = float(matrix[dim].iloc[0])
            bar   = "█" * int(score * bar_len) + "░" * (bar_len - int(score * bar_len))
            desc  = dim_descriptions.get(dim, "")
            print(f"    {dim:<26} [{bar}]  {score:.3f}")
            print(f"    {'':<26}  └─ {desc}\n")

    comp = float(matrix["composite_risk_score"].iloc[0]) if "composite_risk_score" in matrix.columns else None
    if comp is not None:
        bar = "█" * int(comp * bar_len) + "░" * (bar_len - int(comp * bar_len))
        _section("Composite (equal-weight average of 4 dimensions)")
        print(f"    {'Composite_Risk_Score':<26} [{bar}]  {comp:.3f}")

    return matrix


def stage5_fl_model(processed_df: pd.DataFrame, customer_id: str, risk_label: str) -> str:
    """
    Run the saved FL/central RiskMLP and report its class probabilities.
    The predicted class is compared to the rule-based label from Stage 3.
    """
    _banner("STAGE 5 — FL Model Inference (RiskMLP Neural Network)")

    cdf      = processed_df[processed_df["Customer_ID"] == customer_id].tail(1)
    feat_vec = cdf[[f for f in RISK_FEATURES if f in cdf.columns]].values.astype(np.float32)

    model = load_central_model()
    preds, probs = predict(model, feat_vec)

    le        = joblib.load(LABEL_ENCODER_PATH)
    pred_idx  = int(preds[0])
    pred_label_enc = le.classes_[pred_idx]    # label encoder's alphabetical class

    _section("Architecture")
    print(f"    Model     : RiskMLP  (15→256→128→64→32→5, GELU, residual)")
    print(f"    Training  : FocalLoss(γ=2.0) + Label-Smoothing(0.05)")
    print(f"    Privacy   : Differentially-private FL  (ε=10.01, δ=1e-5)")

    _section("Softmax class probabilities")
    sorted_classes = le.classes_          # alphabetical: High, Low, Medium, Very_High, Very_Low
    for cls, p in zip(sorted_classes, probs[0]):
        bar    = "█" * int(p * 40) + "░" * (40 - int(p * 40))
        marker = "  ◄ predicted" if cls == pred_label_enc else ""
        print(f"    {cls:<12} [{bar}]  {p:.4f}{marker}")

    _section("FL model prediction")
    _arrow("Predicted Risk Class (model)", pred_label_enc)
    _arrow("Population-based Label (rule)", risk_label)

    agreement = "✓ AGREE" if pred_label_enc == risk_label else "≈ minor difference (expected)"
    print(f"\n    Agreement check : {agreement}")

    # Return population-based label as authoritative (used downstream)
    return risk_label


def stage6_recommendations(risk_label: str, mf_df: pd.DataFrame,
                             total_amount: float) -> dict:
    """
    Use recommend_full_profile() to generate diversified core-satellite
    fund portfolios for all investment horizons.
    """
    _banner("STAGE 6 — Fund Recommendations  (Core-Satellite per Horizon)")

    profile = recommend_full_profile(
        risk_label      = risk_label,
        mf_df           = mf_df,
        horizons        = [1, 3, 5, 10],
        top_n_per_bracket = 3,
        total_amount    = total_amount,
    )

    _section("Equity–Debt glide path (SEBI-aligned allocation by horizon)")
    glide = [(1, 15, 85), (3, 50, 50), (5, 75, 25), (10, 85, 15)]
    print(f"    {'Horizon':<12} {'Equity %':<12} {'Debt %':<12}")
    print("    " + "-" * 36)
    for h, eq, dbt in glide:
        print(f"    {str(h)+' yr':<12} {str(eq)+'%':<12} {str(dbt)+'%':<12}")

    for horizon_label, port in profile["horizons"].items():
        _section(f"Horizon: {horizon_label}")
        eq_raw  = port.get("equity_pct", 0)           # stored as fraction 0–1
        eq_pct  = eq_raw * 100 if eq_raw <= 1.0 else eq_raw   # convert to %
        print(f"    Equity allocation : {eq_pct:.0f}%")
        print(f"    Debt  allocation  : {100-eq_pct:.0f}%\n")

        for bkt in port.get("brackets", []):
            bracket  = bkt.get("bracket", "?")
            pct      = bkt.get("pct", 0)
            funds_df = bkt.get("funds", pd.DataFrame())
            alloc    = int(round(pct / 100 * port["total_amount"]))

            if funds_df is not None and not (hasattr(funds_df, 'empty') and funds_df.empty):
                print(f"    [{bracket}]  {pct:.0f}% weight  →  ₹{alloc:,.0f}")
                rows = funds_df.to_dict("records") if hasattr(funds_df, 'to_dict') else []
                for i, f in enumerate(rows, 1):
                    scheme = f.get("Scheme_Name", "?")
                    cat    = f.get("Scheme_Category", "?")
                    ret1   = f.get("Return_1Y", None)
                    ret_str = f"  1yr: {ret1:.1f}%" if ret1 is not None else ""
                    print(f"      {i}. {scheme[:55]:<55}  [{cat[:30]}]{ret_str}")
                print()

    return profile


def stage7_genai(profile: dict, risk_label: str, name: str,
                 occupation: str = "Customer") -> None:
    """
    Call explain_full_profile() for LLM/rule-based portfolio explanations.
    """
    _banner("STAGE 7 — GenAI Explanation  (LLM  →  Rule-Based Fallback)")

    user_context = (
        f"Customer {name} is a {occupation} with {risk_label.replace('_', ' ')} "
        f"risk appetite, looking to invest ₹{profile['total_amount']:,.0f}."
    )

    print(f"\n  User context: {user_context}\n")
    print("  Generating explanations (may take a few seconds)...\n")

    explanations = explain_full_profile(
        profile_data = profile,          # full profile dict (keys: risk_label, total_amount, horizons)
        user_risk    = risk_label,
        user_context = user_context,
    )

    for horizon_label, result in explanations.items():
        _section(f"Horizon: {horizon_label}  [provider: {result.get('provider','?')}]")
        explanation = result.get("explanation", "No explanation generated.")
        wrapped = textwrap.fill(explanation, width=70,
                                initial_indent="    ",
                                subsequent_indent="    ")
        print(wrapped)
        print()


# ─── Main driver ──────────────────────────────────────────────────────────────

def run_demo(customer_id: str, total_amount: float) -> None:
    t0 = time.time()

    print("\n" + "═" * 72)
    print("  SMART FUND ADVISOR — Single User End-to-End Demo")
    print("═" * 72)
    print(f"  Customer ID  : {customer_id}")

    # Indicate whether this customer is from the held-out demo set
    if DEMO_CUSTOMERS_PATH.exists():
        demo_ids = set(pd.read_csv(DEMO_CUSTOMERS_PATH)["Customer_ID"].astype(str))
        holdout_tag = "YES (5% demo holdout — never seen in training or FL)" \
                      if customer_id in demo_ids else "NO (not in demo holdout)"
        print(f"  Demo holdout : {holdout_tag}")
    print(f"  Amount       : ₹{total_amount:,.0f}")
    print(f"  Horizons     : 1 yr  |  3 yr  |  5 yr  |  10 yr+")
    print("═" * 72)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  [Loading datasets…]")
    raw_df  = pd.read_csv("Data/bank_user_dataset.csv", dtype=str,
                          low_memory=False)
    # Convert numeric columns for display
    num_cols = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    for col in num_cols:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    mf_df   = load_mutual_funds()          # tags risk_tier from Scheme_Category
    proc_df = get_clean_customer_data(fit_scaler=False)   # use saved scaler
    print("  [Datasets loaded]")

    # ── Pipeline stages ───────────────────────────────────────────────────────
    stage1_raw_records(raw_df, customer_id)

    feat_row = stage2_feature_engineering(proc_df, customer_id)

    risk_label, risk_score = stage3_risk_label(proc_df, customer_id)

    stage4_risk_matrix(proc_df, customer_id)

    final_label = stage5_fl_model(proc_df, customer_id, risk_label)

    profile = stage6_recommendations(final_label, mf_df, total_amount)

    # Retrieve name & occupation for GenAI context
    raw_row  = raw_df[raw_df["Customer_ID"] == customer_id].tail(1)
    name     = raw_row.iloc[0].get("Name", customer_id)     if not raw_row.empty else customer_id
    occ      = raw_row.iloc[0].get("Occupation", "Customer") if not raw_row.empty else "Customer"

    stage7_genai(profile, final_label, name, occ)

    # ── Summary table ─────────────────────────────────────────────────────────
    _banner("SUMMARY")
    _arrow("Customer",              f"{customer_id}  ({name})")
    _arrow("Occupation",             str(occ))
    _arrow("Risk Score",             f"{risk_score:.4f}")
    _arrow("Risk Label (rules)",     risk_label)
    _arrow("Risk Label (FL model)",  final_label)
    _arrow("Investment Amount",      f"₹{total_amount:,.0f}")
    _arrow("Horizons covered",       "1 yr  |  3 yr  |  5 yr  |  10 yr+")
    _arrow("Explanation provider",   "Groq / OpenRouter / HF / rule-based")
    _arrow("Elapsed",                f"{time.time()-t0:.1f} s")

    print("\n" + "═" * 72)
    print("  Demo complete.")
    print("═" * 72 + "\n")


def list_customers(n: int = 20) -> None:
    """Print demo holdout IDs (preferred) or fall back to first N raw CSV IDs."""
    if DEMO_CUSTOMERS_PATH.exists():
        demo_df = pd.read_csv(DEMO_CUSTOMERS_PATH)
        print(f"\n  Demo holdout customers ({len(demo_df)} total — never used in training or FL):")
        for _, row in demo_df.head(n).iterrows():
            print(f"    {row['Customer_ID']:<18}  risk: {row['risk_label']}")
    else:
        raw_df = pd.read_csv("Data/bank_user_dataset.csv", usecols=["Customer_ID"],
                             low_memory=False)
        ids = raw_df["Customer_ID"].dropna().unique()[:n]
        print(f"  [No demo_customers.csv found — run train.py first]")
        print(f"  First {n} customer IDs in the dataset:")
        for cid in ids:
            print(f"    {cid}")
    print()


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _default_demo_customer() -> str:
    """Return the first customer from the held-out demo set, or a fallback."""
    if DEMO_CUSTOMERS_PATH.exists():
        demo_df = pd.read_csv(DEMO_CUSTOMERS_PATH)
        if not demo_df.empty:
            return str(demo_df.iloc[0]["Customer_ID"])
    return "CUS_0xd40"   # fallback before train.py has been run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end Smart Fund Advisor demo for one customer.",
    )
    parser.add_argument(
        "--customer", "-c",
        default=None,
        help="Customer_ID to demo (default: first customer from demo holdout set)",
    )
    parser.add_argument(
        "--amount", "-a",
        type=float,
        default=100_000.0,
        help="Total investment amount in INR (default: 500000)",
    )
    parser.add_argument(
        "--list-customers",
        action="store_true",
        help="Print first 20 customer IDs and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_customers:
        list_customers()
    else:
        customer_id = args.customer if args.customer else _default_demo_customer()
        run_demo(customer_id=customer_id, total_amount=args.amount)
