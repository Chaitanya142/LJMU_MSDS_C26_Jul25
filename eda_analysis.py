"""EDA script - runs full analysis on both datasets, identifies columns to drop,
plots correlation matrix, saves results."""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

ROOT = '/Users/chaitanya/Downloads/Submission/Code/20Feb26'
OUT  = os.path.join(ROOT, 'eda_outputs')
os.makedirs(OUT, exist_ok=True)

# ──────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────
df_bank = pd.read_csv(os.path.join(ROOT, 'Data', 'bank_user_dataset.csv'))
df_fund = pd.read_csv(os.path.join(ROOT, 'Data', 'mutual_fund_data.csv'))

print(f"Bank dataset : {df_bank.shape}")
print(f"Fund dataset : {df_fund.shape}")

# ──────────────────────────────────────────────
# 2. BANK DATASET EDA
# ──────────────────────────────────────────────
print("\n===== BANK DATASET EDA =====")

# --- clean dirty numeric columns ---
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True),
        errors='coerce'
    )

df_bank['Age_clean']              = clean_numeric(df_bank['Age'])
df_bank['Annual_Income_clean']    = clean_numeric(df_bank['Annual_Income'])
df_bank['Num_of_Loan_clean']      = clean_numeric(df_bank['Num_of_Loan'])
df_bank['Outstanding_Debt_clean'] = clean_numeric(df_bank['Outstanding_Debt'])
df_bank['Monthly_Balance_clean']  = clean_numeric(df_bank['Monthly_Balance'])
df_bank['Amount_invested_monthly_clean'] = clean_numeric(df_bank['Amount_invested_monthly'])
df_bank['Num_of_Delayed_Payment_clean']  = clean_numeric(df_bank['Num_of_Delayed_Payment'])
df_bank['Changed_Credit_Limit_clean']    = clean_numeric(df_bank['Changed_Credit_Limit'])

# Parse Credit_History_Age → months
def parse_credit_age(s):
    if pd.isna(s): return np.nan
    m = re.match(r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?', str(s))
    if m: return int(m.group(1))*12 + int(m.group(2))
    m2 = re.match(r'(\d+)\s*Years?', str(s))
    if m2: return int(m2.group(1))*12
    return np.nan
df_bank['Credit_History_Months'] = df_bank['Credit_History_Age'].apply(parse_credit_age)

# Encode categoricals for correlation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_bank_enc = df_bank.copy()
for col in ['Occupation','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour']:
    df_bank_enc[col+'_enc'] = le.fit_transform(df_bank_enc[col].fillna('Unknown'))

# Numeric columns for correlation
bank_num_cols = [
    'Age_clean','Annual_Income_clean','Monthly_Inhand_Salary','Num_Bank_Accounts',
    'Num_Credit_Card','Interest_Rate','Num_of_Loan_clean','Delay_from_due_date',
    'Num_of_Delayed_Payment_clean','Changed_Credit_Limit_clean','Num_Credit_Inquiries',
    'Outstanding_Debt_clean','Credit_Utilization_Ratio','Credit_History_Months',
    'Total_EMI_per_month','Amount_invested_monthly_clean','Monthly_Balance_clean',
    'Occupation_enc','Credit_Mix_enc','Payment_of_Min_Amount_enc','Payment_Behaviour_enc'
]
bank_num_cols = [c for c in bank_num_cols if c in df_bank_enc.columns]
df_bank_corr = df_bank_enc[bank_num_cols].dropna(how='all')

corr_bank = df_bank_corr.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_bank, dtype=bool))
sns.heatmap(corr_bank, ax=ax, mask=mask, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size':7},
            linewidths=0.3, square=True)
ax.set_title('Bank User Dataset – Feature Correlation Matrix', fontsize=14, pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'bank_correlation_matrix.png'), dpi=150)
plt.close()
print("Saved bank_correlation_matrix.png")

# --- missing rate table ---
bank_missing = (df_bank.isnull().sum() / len(df_bank) * 100).sort_values(ascending=False)
print("\nBank missing % :\n", bank_missing[bank_missing > 0])

# --- Columns to DROP from bank dataset ---
# ID / SSN / Name → PII, no predictive value
# Month → ordering artefact, we aggregate per customer
# ISIN columns → not needed for risk labeling
# Type_of_Loan → free text, hard to use; high null; Num_of_Loan captures count
bank_drop_cols = ['ID', 'SSN', 'Name', 'Month', 'Type_of_Loan', 'Credit_History_Age',
                  'Amount_invested_monthly',   # → replaced by _clean
                  'Outstanding_Debt',          # → replaced by _clean
                  'Monthly_Balance',           # → replaced by _clean
                  'Age',                       # → replaced by _clean
                  'Annual_Income',             # → replaced by _clean
                  'Num_of_Loan',               # → replaced by _clean
                  'Num_of_Delayed_Payment',    # → replaced by _clean
                  'Changed_Credit_Limit',      # → replaced by _clean
                  ]

print("\nBank columns TO DROP:", bank_drop_cols)

# High-corr pairs (|r|>0.8)
print("\nHighly correlated bank pairs (|r| > 0.80):")
for i in range(len(corr_bank.columns)):
    for j in range(i+1, len(corr_bank.columns)):
        v = corr_bank.iloc[i,j]
        if abs(v) > 0.80:
            print(f"  {corr_bank.columns[i]} ↔ {corr_bank.columns[j]}  r={v:.3f}")

# ──────────────────────────────────────────────
# 3. MUTUAL FUND DATASET EDA
# ──────────────────────────────────────────────
print("\n===== MUTUAL FUND DATASET EDA =====")
fund_missing = (df_fund.isnull().sum() / len(df_fund) * 100).sort_values(ascending=False)
print("Fund missing % :\n", fund_missing[fund_missing > 0])

print("\nScheme_Type distribution:\n", df_fund['Scheme_Type'].value_counts())
print("\nTop 15 Scheme_Category:\n", df_fund['Scheme_Category'].value_counts().head(15))
print("\nNAV stats:\n", df_fund['NAV'].describe())
print("\nAverage_AUM_Cr stats:\n", df_fund['Average_AUM_Cr'].describe())

# Fund numeric columns for correlation
df_fund['Scheme_Min_Amt_clean'] = pd.to_numeric(
    df_fund['Scheme_Min_Amt'].astype(str).str.replace(',','').str.extract(r'([\d.]+)')[0],
    errors='coerce'
)
df_fund['Launch_Year'] = pd.to_datetime(df_fund['Launch_Date'], errors='coerce').dt.year
df_fund['Is_Closed']   = df_fund['Closure_Date'].notna().astype(int)
df_fund['Scheme_Type_enc'] = le.fit_transform(df_fund['Scheme_Type'].fillna('Unknown'))
df_fund['Scheme_Cat_enc']  = le.fit_transform(df_fund['Scheme_Category'].fillna('Unknown'))

fund_num_cols = ['NAV', 'Average_AUM_Cr', 'Scheme_Min_Amt_clean',
                 'Launch_Year', 'Is_Closed', 'Scheme_Type_enc', 'Scheme_Cat_enc']
fund_num_cols = [c for c in fund_num_cols if c in df_fund.columns]
df_fund_corr = df_fund[fund_num_cols].dropna(how='all')

corr_fund = df_fund_corr.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_fund, dtype=bool))
sns.heatmap(corr_fund, ax=ax, mask=mask, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size':9},
            linewidths=0.5, square=True)
ax.set_title('Mutual Fund Dataset – Feature Correlation Matrix', fontsize=13, pad=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fund_correlation_matrix.png'), dpi=150)
plt.close()
print("Saved fund_correlation_matrix.png")

# --- Columns to DROP from fund dataset ---
# ISIN_Div_Payout/Growth, ISIN_Div_Reinvestment, ISIN_Div_Payout/Growth/Div_Reinvestment
#   → identifier columns, >12% null, no predictive value for scoring
# AAUM_Quarter → redundant with Average_AUM_Cr (same quarter label)
# Closure_Date → 76% null (irrelevant for active funds); Is_Closed flag replaces it
# Scheme_NAV_Name → near-duplicate of Scheme_Name, verbose text
fund_drop_cols = ['ISIN_Div_Payout/Growth',
                  'ISIN_Div_Reinvestment',
                  'ISIN_Div_Payout/Growth/Div_Reinvestment',
                  'AAUM_Quarter',
                  'Closure_Date',
                  'Scheme_NAV_Name']
print("\nFund columns TO DROP:", fund_drop_cols)

# Unique counts for categorical columns
print("\nAMC count:", df_fund['AMC'].nunique())
print("Scheme_Category count:", df_fund['Scheme_Category'].nunique())

# ──────────────────────────────────────────────
# 4. CLEAN & SAVE DATASETS
# ──────────────────────────────────────────────

# --- BANK: aggregate to one row per customer (keep last month) ---
df_bank_clean = df_bank.copy()
# parse all dirty numerics first (already done into _clean cols)
# keep raw cleaned columns, drop originals + identifiers
df_bank_clean = df_bank_clean.drop(columns=[c for c in bank_drop_cols if c in df_bank_clean.columns])

# Rename _clean columns back to original names
rename_map = {
    'Age_clean': 'Age',
    'Annual_Income_clean': 'Annual_Income',
    'Num_of_Loan_clean': 'Num_of_Loan',
    'Outstanding_Debt_clean': 'Outstanding_Debt',
    'Monthly_Balance_clean': 'Monthly_Balance',
    'Amount_invested_monthly_clean': 'Amount_invested_monthly',
    'Num_of_Delayed_Payment_clean': 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit_clean': 'Changed_Credit_Limit',
    'Credit_History_Months': 'Credit_History_Months',
}
df_bank_clean = df_bank_clean.rename(columns=rename_map)

# Remove remaining dirty characters from numeric-string columns (Occupation etc already str)
print("\nBank clean shape:", df_bank_clean.shape)
print("Bank clean columns:", list(df_bank_clean.columns))

# --- FUND: drop columns ---
df_fund_clean = df_fund.drop(columns=[c for c in fund_drop_cols if c in df_fund.columns])
df_fund_clean['Fund_Age_Years'] = (
    pd.Timestamp('2026-03-13') - pd.to_datetime(df_fund_clean['Launch_Date'], errors='coerce')
).dt.days / 365.25
df_fund_clean = df_fund_clean.drop(columns=['Launch_Date'], errors='ignore')
print("\nFund clean shape:", df_fund_clean.shape)
print("Fund clean columns:", list(df_fund_clean.columns))

# Save cleaned datasets
bank_clean_path = os.path.join(ROOT, 'Data', 'bank_user_dataset_clean.csv')
fund_clean_path = os.path.join(ROOT, 'Data', 'mutual_fund_data_clean.csv')
df_bank_clean.to_csv(bank_clean_path, index=False)
df_fund_clean.to_csv(fund_clean_path, index=False)
print(f"\nSaved: {bank_clean_path}")
print(f"Saved: {fund_clean_path}")

# ──────────────────────────────────────────────
# 5. DISTRIBUTION PLOTS
# ──────────────────────────────────────────────

# Bank numeric distributions
num_plot_bank = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
                 'Num_Credit_Card','Interest_Rate','Num_of_Loan','Outstanding_Debt',
                 'Credit_Utilization_Ratio','Total_EMI_per_month',
                 'Amount_invested_monthly','Monthly_Balance']
num_plot_bank = [c for c in num_plot_bank if c in df_bank_clean.columns]
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(num_plot_bank):
    data = pd.to_numeric(df_bank_clean[col], errors='coerce').dropna()
    data = data[data.between(data.quantile(0.01), data.quantile(0.99))]
    axes[i].hist(data, bins=40, color='steelblue', edgecolor='white', linewidth=0.3)
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].tick_params(labelsize=7)
    axes[i].set_ylabel('Count', fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Bank Dataset – Feature Distributions (1–99 pct)', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'bank_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved bank_distributions.png")

# Fund distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ['NAV','Average_AUM_Cr','Fund_Age_Years']):
    data = pd.to_numeric(df_fund_clean[col], errors='coerce').dropna()
    data = data[data > 0]
    if col in ('NAV','Average_AUM_Cr'):
        data = np.log1p(data)
        lbl = f'log1p({col})'
    else:
        lbl = col
    ax.hist(data, bins=40, color='darkorange', edgecolor='white', linewidth=0.3)
    ax.set_title(lbl, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
plt.suptitle('Mutual Fund Dataset – Key Feature Distributions', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fund_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fund_distributions.png")

print("\n\n========== SUMMARY ==========")
print(f"Bank:  {df_bank.shape} → {df_bank_clean.shape}  (dropped {len(bank_drop_cols)} columns)")
print(f"Fund:  {df_fund.shape} → {df_fund_clean.shape}  (dropped {len(fund_drop_cols)} columns)")
print(f"All plots saved to: {OUT}")
print("EDA complete.")
