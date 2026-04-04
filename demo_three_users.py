"""
demo_three_users.py — End-to-end proof that the Smart Fund Advisor pipeline
works correctly across the full risk spectrum.

Three demo customers are drawn from the held-out validation set (5% of the
12,500-customer population — none of these customers were seen during central
model training or any FL round):

  ┌────────────────┬──────────────────────────┬────────┬───────────┬──────────────┐
  │ Customer ID    │ Name                     │ Age    │ Bracket   │ Amount       │
  ├────────────────┼──────────────────────────┼────────┼───────────┼──────────────┤
  │ CUS_0x11ae     │ Xiaoyi Shaou             │ 22     │ Very_Low  │ ₹   50,000   │
  │ CUS_0x1096     │ Edward Krudyk            │ 43     │ Medium    │ ₹  500,000   │
  │ CUS_0x102d     │ Neil Chatterjeex         │ 31     │ Very_High │ ₹2,000,000   │
  └────────────────┴──────────────────────────┴────────┴───────────┴──────────────┘

Each user is processed through all 7 pipeline stages:
  STAGE 1  Raw bank records
  STAGE 2  Feature engineering (normalised 15-feature vector)
  STAGE 3  Risk scoring & labelling (bell-curve quantile binning)
  STAGE 4  Multi-metric risk matrix (4 orthogonal dimensions)
  STAGE 5  FL model inference (RiskMLP neural network)
  STAGE 6  Fund recommendations (core-satellite, 3-yr horizon)
  STAGE 7  GenAI explanation (LLM → rule-based fallback)

At the end a consolidated proof table summarises all key results side-by-side
and performs 7 automated correctness checks to confirm system integrity.

Usage
─────
  python demo_three_users.py           # run all three users sequentially
  python demo_three_users.py --no-genai  # skip Stage 7 (faster)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ─── Reuse all stage functions and helpers from demo_single_user ──────────────
from demo_single_user import (
    stage1_raw_records,
    stage2_feature_engineering,
    stage3_risk_label,
    stage4_risk_matrix,
    stage5_fl_model,
    stage6_recommendations,
    stage7_genai,
    _banner,
    _arrow,
    _section,
    _divider,
)

from config import RISK_FEATURES, LABEL_ENCODER_PATH, DEMO_CUSTOMERS_PATH
from src.preprocessing import get_clean_customer_data
from src.recommender   import load_mutual_funds


# ─── Holdout demo users ───────────────────────────────────────────────────────
# Each tuple: (customer_id, expected_bracket, investment_amount_INR)
THREE_USERS: list[tuple[str, str, float]] = [
    ("CUS_0x11ae", "Very_Low",   50_000.0),
    ("CUS_0x1096", "Medium",    500_000.0),
    ("CUS_0x102d", "Very_High", 2_000_000.0),
]

# ─── Display helpers (local) ──────────────────────────────────────────────────

def _double_banner(title: str) -> None:
    line = "═" * 72
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _proof_check(label: str, passed: bool) -> None:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"    {label:<52} {status}")


# ─── Single-user driver (wraps demo_single_user stages) ──────────────────────

def run_one_user(
    customer_id: str,
    expected_bracket: str,
    total_amount: float,
    raw_df: pd.DataFrame,
    proc_df: pd.DataFrame,
    mf_df: pd.DataFrame,
    include_genai: bool,
) -> dict:
    """
    Run the full 7-stage pipeline for one customer and return a results dict
    for the final proof table.
    """
    _double_banner(
        f"USER: {customer_id}  |  Expected bracket: {expected_bracket}"
        f"  |  Amount: ₹{total_amount:,.0f}"
    )

    # Confirm holdout membership
    if DEMO_CUSTOMERS_PATH.exists():
        demo_ids = set(pd.read_csv(DEMO_CUSTOMERS_PATH)["Customer_ID"].astype(str))
        in_holdout = customer_id in demo_ids
        print(f"  Demo holdout : {'YES — never seen in training or FL' if in_holdout else 'NO'}")

    print()
    t_start = time.time()

    # Stage 1 – raw records
    stage1_raw_records(raw_df, customer_id)

    # Stage 2 – feature engineering
    feat_row = stage2_feature_engineering(proc_df, customer_id)

    # Stage 3 – risk scoring & labelling
    rule_label, risk_score = stage3_risk_label(proc_df, customer_id)

    # Stage 4 – multi-metric risk matrix
    stage4_risk_matrix(proc_df, customer_id)

    # Stage 5 – FL model inference
    fl_label = stage5_fl_model(proc_df, customer_id, rule_label)

    # Stage 6 – fund recommendations
    profile = stage6_recommendations(fl_label, mf_df, total_amount)

    # Stage 7 – GenAI explanation (optional)
    if include_genai:
        raw_row = raw_df[raw_df["Customer_ID"] == customer_id].tail(1)
        occ = raw_row.iloc[0].get("Occupation", "Customer") if not raw_row.empty else "Customer"
        stage7_genai(profile, fl_label, occ)

    elapsed = time.time() - t_start

    # Collect per-user softmax confidence for the proof table
    le = joblib.load(LABEL_ENCODER_PATH)
    from src.central_model import load_central_model, predict
    model    = load_central_model()
    cdf      = proc_df[proc_df["Customer_ID"] == customer_id].tail(1)
    feat_vec = cdf[[f for f in RISK_FEATURES if f in cdf.columns]].values.astype(np.float32)
    _, probs = predict(model, feat_vec)
    top_prob = float(probs[0].max())

    # Per-user mini summary
    _banner(f"USER SUMMARY — {customer_id}")
    _arrow("Expected bracket",     expected_bracket)
    _arrow("Rule-based label",     rule_label)
    _arrow("FL model prediction",  fl_label)
    _arrow("Risk score",           f"{risk_score:.4f}")
    _arrow("Top-class confidence", f"{top_prob:.4f}")
    _arrow("Investment amount",    f"₹{total_amount:,.0f}")
    _arrow("Elapsed (this user)",  f"{elapsed:.1f} s")

    label_match   = (rule_label == expected_bracket)
    fl_match      = (fl_label   == expected_bracket)
    both_agree    = (rule_label == fl_label)
    high_conf     = top_prob >= 0.70

    print()
    _proof_check("Rule label matches expected bracket",          label_match)
    _proof_check("FL model matches expected bracket",            fl_match)
    _proof_check("Rule and FL agree with each other",            both_agree)
    _proof_check("FL confidence ≥ 0.70 (clear prediction)",      high_conf)
    print()

    return {
        "customer_id":       customer_id,
        "expected":          expected_bracket,
        "rule_label":        rule_label,
        "fl_label":          fl_label,
        "risk_score":        risk_score,
        "top_prob":          top_prob,
        "total_amount":      total_amount,
        "elapsed":           elapsed,
        "label_match":       label_match,
        "fl_match":          fl_match,
        "both_agree":        both_agree,
        "high_conf":         high_conf,
    }


# ─── Consolidated proof table ──────────────────────────────────────────────────

def print_proof_table(results: list[dict], total_elapsed: float) -> None:
    _double_banner("CONSOLIDATED PROOF TABLE — All 3 Demo Users")

    # ── Per-user comparison ────────────────────────────────────────────────────
    header  = f"  {'Customer':<14} {'Expected':<12} {'Rule':<12} {'FL Model':<12} {'Score':>7}  {'Conf':>6}  {'Amount':>13}  {'Rule✓':<6} {'FL✓':<6} {'Agree':<6}"
    divider = "  " + "─" * (len(header) - 2)
    print(header)
    print(divider)
    for r in results:
        rule_ok  = "✓" if r["label_match"] else "✗"
        fl_ok    = "✓" if r["fl_match"]    else "✗"
        agree_ok = "✓" if r["both_agree"]  else "✗"
        print(
            f"  {r['customer_id']:<14} {r['expected']:<12} {r['rule_label']:<12}"
            f" {r['fl_label']:<12} {r['risk_score']:>7.4f}  {r['top_prob']:>6.4f}"
            f"  ₹{r['total_amount']:>12,.0f}  {rule_ok:<6} {fl_ok:<6} {agree_ok}"
        )
    print(divider)

    # ── Automated correctness checks ──────────────────────────────────────────
    _section("Automated Correctness Checks")
    print()

    all_rule_match   = all(r["label_match"] for r in results)
    all_fl_match     = all(r["fl_match"]    for r in results)
    all_agree        = all(r["both_agree"]  for r in results)
    all_high_conf    = all(r["high_conf"]   for r in results)
    vl_is_lowest     = results[0]["risk_score"] < results[1]["risk_score"] < results[2]["risk_score"]
    vl_under_medium  = results[0]["risk_score"]  < 0.0
    vh_over_medium   = results[2]["risk_score"]  > 0.0
    brackets_differ  = len({r["fl_label"] for r in results}) == 3

    _proof_check("All 3 rule-based labels match expected brackets",     all_rule_match)
    _proof_check("All 3 FL model predictions match expected brackets",  all_fl_match)
    _proof_check("Rule and FL agree for every user",                    all_agree)
    _proof_check("All 3 FL predictions have confidence ≥ 0.70",        all_high_conf)
    _proof_check("Risk scores are strictly ordered (VL < Med < VH)",   vl_is_lowest)
    _proof_check("Very_Low user has negative risk score (low-risk)",    vl_under_medium)
    _proof_check("Very_High user has positive risk score (high-risk)",  vh_over_medium)
    _proof_check("All 3 users receive distinct risk tier labels",       brackets_differ)

    total_checks = 8
    passed_checks = sum([
        all_rule_match, all_fl_match, all_agree, all_high_conf,
        vl_is_lowest, vl_under_medium, vh_over_medium, brackets_differ,
    ])

    print()
    print(f"  Result : {passed_checks}/{total_checks} checks PASSED")
    overall = "ALL CHECKS PASSED — System correctly classifies users across the full risk spectrum." \
              if passed_checks == total_checks else \
              f"WARNING: {total_checks - passed_checks} check(s) failed — review output above."
    sep = "═" * 68
    print(f"  {sep}")
    print(f"  {overall}")
    print(f"  {sep}")

    # ── Timing summary ──────────────────────────────────────────────────────
    _section("Timing")
    for r in results:
        print(f"    {r['customer_id']:<14}  {r['elapsed']:.1f} s")
    print(f"    {'Total':<14}  {total_elapsed:.1f} s")
    print()


# ─── Main driver ──────────────────────────────────────────────────────────────

def run_three_users_demo(include_genai: bool = True) -> None:
    t_total = time.time()

    _double_banner("SMART FUND ADVISOR — Three-User End-to-End Proof Demo")
    print("  This demo runs the complete 7-stage investment advisory pipeline for")
    print("  three customers from the held-out validation set, covering Very_Low,")
    print("  Medium, and Very_High risk brackets.")
    print()
    print("  Holdout set: 625 customers (5% of 12,500) — zero training/FL exposure")
    print()
    print(f"  {'#':<3} {'Customer ID':<14} {'Bracket':<12} {'Name':<24} {'Age':>4}  {'Amount':>13}")
    print("  " + "─" * 72)
    names_ages = [
        ("Xiaoyi Shaou",    22),
        ("Edward Krudyk",   43),
        ("Neil Chatterjeex", 31),
    ]
    for i, ((cid, bracket, amount), (name, age)) in enumerate(zip(THREE_USERS, names_ages), 1):
        print(f"  {i:<3} {cid:<14} {bracket:<12} {name:<24} {age:>4}  ₹{amount:>12,.0f}")
    print()

    # ── Load shared datasets once ──────────────────────────────────────────────
    print("  [Loading datasets…]")
    raw_df = pd.read_csv("Data/bank_user_dataset.csv", dtype=str, low_memory=False)
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

    mf_df   = load_mutual_funds()
    proc_df = get_clean_customer_data(fit_scaler=False)
    print("  [Datasets loaded — processing each user now]")

    # ── Run pipeline for each user ─────────────────────────────────────────────
    all_results = []
    for customer_id, bracket, amount in THREE_USERS:
        result = run_one_user(
            customer_id      = customer_id,
            expected_bracket = bracket,
            total_amount     = amount,
            raw_df           = raw_df,
            proc_df          = proc_df,
            mf_df            = mf_df,
            include_genai    = include_genai,
        )
        all_results.append(result)

    # ── Print consolidated proof table ─────────────────────────────────────────
    print_proof_table(all_results, time.time() - t_total)

    print("═" * 72)
    print("  Demo complete.")
    print("═" * 72 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run end-to-end Smart Fund Advisor demo for 3 representative users.",
    )
    p.add_argument(
        "--no-genai",
        action="store_true",
        help="Skip Stage 7 (GenAI explanation) for faster execution.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_three_users_demo(include_genai=not args.no_genai)
