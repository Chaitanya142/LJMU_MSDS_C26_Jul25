"""
prepare_clean_data.py
─────────────────────
Creates two production-ready clean datasets from the raw CSVs based on EDA findings.

BANK (50,000 rows, 27 cols → 22 cols)
  Dropped (PII / unused in model):  ID, SSN, Name, Month, Type_of_Loan
  Added:  Credit_History_Months (from Credit_History_Age)

MUTUAL FUND (16,376 rows, 16 cols → 12 cols)
  Dropped (identifier codes / fully-redundant):
    ISIN_Div_Payout/Growth, ISIN_Div_Reinvestment,
    ISIN_Div_Payout/Growth/Div_Reinvestment, AAUM_Quarter
  Kept:  Launch_Date, Closure_Date, Scheme_NAV_Name (all used by downstream code)
"""
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/Users/chaitanya/Downloads/Submission/Code/20Feb26')
DATA = ROOT / 'Data'

# ──────────────────────────────────────────────────────────────────────────────
# BANK DATASET
# ──────────────────────────────────────────────────────────────────────────────
df_bank = pd.read_csv(DATA / 'bank_user_dataset.csv', low_memory=False)
df_bank.columns = [c.strip() for c in df_bank.columns]

print(f"Bank raw shape: {df_bank.shape}")

# Parse Credit_History_Age → months BEFORE dropping the column
def _parse_credit_age(val):
    val = str(val)
    y = re.search(r'(\d+)\s*[Yy]ear', val)
    m = re.search(r'(\d+)\s*[Mm]onth', val)
    years  = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    total  = years * 12 + months
    return total if total > 0 else np.nan

if 'Credit_History_Age' in df_bank.columns:
    df_bank['Credit_History_Months'] = df_bank['Credit_History_Age'].apply(_parse_credit_age)
    # forward/back fill within customer, then global median
    df_bank['Credit_History_Months'] = (
        df_bank.groupby('Customer_ID')['Credit_History_Months']
        .transform(lambda x: x.ffill().bfill())
    )
    df_bank['Credit_History_Months'] = df_bank['Credit_History_Months'].fillna(
        df_bank['Credit_History_Months'].median()
    )

# Drop PII and unused columns
bank_drop = ['ID', 'SSN', 'Name', 'Month', 'Type_of_Loan', 'Credit_History_Age']
df_bank_clean = df_bank.drop(columns=[c for c in bank_drop if c in df_bank.columns])

print(f"Bank clean shape: {df_bank_clean.shape}")
print(f"Bank clean columns ({len(df_bank_clean.columns)}): {list(df_bank_clean.columns)}")

out_bank = DATA / 'bank_user_dataset_clean.csv'
df_bank_clean.to_csv(out_bank, index=False)
print(f"Saved: {out_bank}\n")

# ──────────────────────────────────────────────────────────────────────────────
# MUTUAL FUND DATASET
# ──────────────────────────────────────────────────────────────────────────────
df_fund = pd.read_csv(DATA / 'mutual_fund_data.csv', low_memory=False)
df_fund.columns = [c.strip() for c in df_fund.columns]

print(f"Fund raw shape: {df_fund.shape}")

# Drop only ISIN identifier columns and AAUM_Quarter (truly unused / redundant)
fund_drop = [
    'ISIN_Div_Payout/Growth',
    'ISIN_Div_Reinvestment',
    'ISIN_Div_Payout/Growth/Div_Reinvestment',
    'AAUM_Quarter',
]
df_fund_clean = df_fund.drop(columns=[c for c in fund_drop if c in df_fund.columns])

print(f"Fund clean shape: {df_fund_clean.shape}")
print(f"Fund clean columns ({len(df_fund_clean.columns)}): {list(df_fund_clean.columns)}")

out_fund = DATA / 'mutual_fund_data_clean.csv'
df_fund_clean.to_csv(out_fund, index=False)
print(f"Saved: {out_fund}\n")

print("='*60")
print("Summary of changes:")
print(f"  Bank:  {df_bank.shape} → {df_bank_clean.shape}  (removed {len(bank_drop)} cols)")
print(f"  Fund:  {df_fund.shape} → {df_fund_clean.shape}  (removed {len(fund_drop)} cols)")
print("Clean datasets saved. Ready to update config.py.")
