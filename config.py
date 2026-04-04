"""
config.py  –  Central configuration for Smart Fund Advisor
All paths, hyper-parameters, and constants live here.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FUNDPERF_DIR = DATA_DIR / "FundPerf"
NIFTY500_DIR = DATA_DIR / "Nifty500"

BANK_CSV            = DATA_DIR  / "bank_user_dataset_clean.csv"   # EDA-cleaned: PII + unused cols removed
MF_CSV              = DATA_DIR  / "mutual_fund_data_clean.csv"     # EDA-cleaned: ISIN identifiers + AAUM_Quarter removed
NAV_HISTORY_PARQUET = DATA_DIR  / "mutual_fund_nav_history.parquet"
TER_CSV             = DATA_DIR  / "ter-of-mf-schemes.csv"   # SEBI TER disclosures (real data)
NAV_METRICS_CSV     = MODELS_DIR / "nav_metrics.csv"   # cached per-scheme metrics
FUND_BENCHMARK_FEATURES_CSV = MODELS_DIR / "fund_benchmark_features.csv"

CENTRAL_MODEL_PATH    = MODELS_DIR / "central_risk_model.pt"
FL_GLOBAL_MODEL       = MODELS_DIR / "fl_global_risk_model.pt"
SCALER_PATH           = MODELS_DIR / "feature_scaler.joblib"
LABEL_ENCODER_PATH    = MODELS_DIR / "label_encoder.joblib"
PCA_RISK_PATH         = MODELS_DIR / "pca_risk.joblib"        # PCA model for risk scoring
PCA_N_COMPONENTS      = 5                                      # components to retain (PC1 is used as risk score)
DEMO_CUSTOMERS_PATH   = MODELS_DIR / "demo_customers.csv"   # 5 % unseen holdout

# ─── Data split ───────────────────────────────────────────────────────────────
# Population of unique customers is partitioned into three non-overlapping sets:
#   65 % → Central model training (80/20 internal train/val)
#   30 % → Federated Learning simulation (each device = one customer)
#    5 % → Demo holdout (never seen during training or FL; used only for demos)
# The demo holdout ensures evaluation/demo users are genuinely unseen.
CENTRAL_SPLIT   = 0.65   # fraction of *unique customers* for central model
DEMO_SPLIT      = 0.05   # fraction reserved as unseen demo holdout
RANDOM_SEED     = 42

# ─── Risk label (5 classes, ordered by risk appetite) ─────────────────────────
RISK_CLASSES = ["Very_Low", "Low", "Medium", "High", "Very_High"]
N_RISK_CLASSES = len(RISK_CLASSES)

# ─── Feature columns used for risk modelling ──────────────────────────────────
RISK_FEATURES = [
    "Annual_Income_norm",
    "Monthly_Inhand_Salary_norm",
    "Investment_Ratio",          # Amount_invested_monthly / Monthly_Inhand_Salary
    "Debt_Burden_Ratio",         # Outstanding_Debt / Annual_Income
    "Credit_Utilization_Ratio",
    "Delay_Score",               # normalised delay-from-due + delayed-payments
    "Credit_Mix_Score",          # Good=1.0, Standard=0.5, Bad=0.0
    "Spending_Behaviour_Score",  # encoded from Payment_Behaviour
    "Num_Bank_Accounts_norm",
    "Interest_Rate_norm",
    "Age_Risk_Proxy",            # (70-Age)/52 — younger = longer horizon = more tolerant
    "Occupation_Stability_Score",# job financial risk-capacity score [0-1]
    # ── New production features (v2) ──────────────────────────────────────────
    "EMI_Income_Ratio",          # Total_EMI / Monthly_Inhand_Salary (DTI proxy)
    "Savings_Rate",              # (Salary - EMI) / Salary — financial buffer metric
    "Credit_History_Score",      # normalised credit history months — maturity proxy
]

# ─── Neural-network hyper-params ──────────────────────────────────────────────
HIDDEN_DIMS  = [256, 128, 64, 32]   # deeper network for 15 features
DROPOUT      = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE   = 64
EPOCHS       = 40          # central training epochs (early stopping active)
LABEL_SMOOTHING = 0.05     # label smoothing to regularise confident predictions
FOCAL_LOSS_GAMMA = 2.0     # focal loss gamma for hard-example mining on tail classes
EARLY_STOP_PATIENCE = 8    # stop if val_acc doesn't improve for N epochs

# ─── Federated learning hyper-params ─────────────────────────────────────────
#
# Architecture: one virtual mobile device per customer in the 30 % FL cohort.
# Each device holds ONLY that customer's own bank records (≤ 4 monthly rows).
# Raw data never leaves the device — only DP-noised gradient updates are sent
# to the aggregation server.
#
FL_ROUNDS          = 15    # number of federated communication rounds
FL_MIN_CLIENTS     = 10    # min devices to sample per round
FL_FRACTION_FIT    = 0.3   # fraction of devices sampled per round  (≈ 30%)
FL_LOCAL_EPOCHS    = 5     # local epochs per device per round
FL_BATCH_SIZE      = 4     # mini-batch size per device (devices have ≤ 4 records)
DP_NOISE_MULTIPLIER = 1.1  # Gaussian DP noise σ — raised from 1.0 to tighten ε via amplification
DP_MAX_GRAD_NORM   = 1.0   # gradient clipping norm for DP sensitivity bound
FEDPROX_MU         = 0.01  # FedProx proximal coefficient (0 = pure FedAvg)

# ── Non-IID simulation: Dirichlet participation bias ─────────────────────────
# Real FL devices do NOT participate uniformly across rounds — conservative
# investors (older, rural) tend to be active in different rounds than
# aggressive investors (young, urban).  We model this via per-round class
# distribution drawn from Dirichlet(α):
#   α = 100  → near-IID (uniform class mix each round)
#   α = 0.5  → moderate non-IID (realistic geographic/demographic clustering)
#   α = 0.1  → extreme non-IID (almost single-class rounds)
# Reference: Li et al. (2022) "Federated Learning on Non-IID Data Silos"
DIRICHLET_ALPHA    = 0.5   # participation skew parameter

# ── Client drift monitoring ───────────────────────────────────────────────────
# After each round, compute cosine similarity between each client's parameter
# update (Δw = w_local − w_global) and the aggregated global update direction.
# Clients with similarity < DRIFT_COSINE_THRESHOLD flag as "high-drift" devices.
# These are logged per round; their proportional weight in FedAvg is halved.
# This simulates a server-side Byzantine-resilience lite mechanism.
DRIFT_COSINE_THRESHOLD = 0.0   # cosine < 0: update opposes global direction
DRIFT_WEIGHT_PENALTY   = 0.5   # multiply n_samples weight by this for drifting clients

# ─── Recommendation engine ────────────────────────────────────────────────────
# Maps risk label → allowed Scheme_Category substrings (case-insensitive match).
# Rules:
#   1. Matching is first-hit: scan Very_Low → Low → Medium → High → Very_High.
#   2. Within each tier, more-specific strings are listed before shorter ones
#      to prevent premature substring matches (e.g. "Balanced Hybrid" before
#      "Balanced", "Gold ETF" before "ETF").
#   3. Legacy pre-SEBI-2018 labels mapped to their modern equivalent tier:
#        "Income"  → Low  (was debt income funds),  "Growth" → High  (was equity growth),
#        "Liquid"  → Very_Low,                        "Balanced" → Medium.
RISK_TO_FUND_CATEGORIES = {
    "Very_Low": [
        "Liquid",               # covers "Liquid Fund" AND the legacy "Liquid" label
        "Overnight Fund",
        "Ultra Short Duration",
        "Money Market",
        "Low Duration",
        "Assured Return",       # legacy capital-protected / assured-return schemes
    ],
    "Low": [
        "Short Duration",
        "Banking and PSU",
        "Corporate Bond",
        "Floater",
        "Gilt",
        "Dynamic Bond",         # Debt Scheme - Dynamic Bond
        "Credit Risk",          # Debt Scheme - Credit Risk Fund (higher yield debt)
        "Medium Duration",      # catches both "Medium Duration Fund" and
                                # "Medium to Long Duration Fund"
        "Long Duration",        # Debt Scheme - Long Duration Fund
        "Income",               # legacy pre-2018 SEBI label for debt income funds
    ],
    "Medium": [
        "Balanced Advantage",           # must come before bare "Balanced"
        "Dynamic Asset Allocation",
        "Aggressive Hybrid",
        "Conservative Hybrid",
        "Balanced Hybrid",              # must come before bare "Balanced"
        "Balanced",                     # legacy "Balanced" closed-end funds
        "Multi Asset Allocation",
        "Arbitrage",
        "Equity Savings",               # Hybrid Scheme - Equity Savings (low equity %)
        "Solution Oriented",            # covers Retirement Fund & Children's Fund
        "FoF Domestic",                 # Fund-of-Funds domestic (diversified multi-scheme)
        "Gold ETF",                     # alternative asset — moderate volatility
    ],
    "High": [
        "Large & Mid Cap",              # must come before bare "Large Cap"
        "Large Cap",
        "Flexi Cap",
        "Multi Cap",
        "Value",
        "Dividend Yield",
        "Focused Fund",                 # concentrated equity, large-cap oriented
        "Contra",                       # contrarian equity, large-cap equivalent
        "Growth",                       # legacy pre-2018 SEBI label for equity growth funds
    ],
    "Very_High": [
        "Mid Cap",
        "Small Cap",
        "Micro Cap",
        "ELSS",
        "Sectoral",
        "Thematic",
        "Index",                        # includes Nifty Next 50, small-cap index
        "FoF Overseas",                 # foreign equity + currency risk
        "ETF",                          # other ETFs (sectoral, factor); Gold ETF
                                        # is already caught by "Gold ETF" in Medium
    ],
}
TOP_N_RECOMMENDATIONS = 5

# ─── Fund risk band floor/ceiling overrides (SEBI-aligned) ───────────────────
# Used by nav_history.compute_fund_risk_bands() step 3.
# After assigning percentile bands purely from realized vol/drawdown data, each
# band is clamped to [floor, ceiling] so that a liquid fund statistically at
# "Medium" within its tiny peer group cannot be rated worse than "Low", and a
# small-cap fund at statistical "Low" is floored at "High".
#
# Key: lowercase substring of Scheme_Category (first match wins; order matters).
# Value: (floor_index, ceiling_index) into RISK_CLASSES list.
#   Very_Low = 0 | Low = 1 | Medium = 2 | High = 3 | Very_High = 4
# Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — product risk-o-meter.
CATEGORY_RISK_BOUNDS: dict = {
    # ── Near-cash / overnight / liquid ───────────────────────────────────────
    "overnight":           (0, 0),   # always Very_Low — zero credit/duration risk
    "liquid":              (0, 1),   # Very_Low – Low   (t+1 redemption)
    "money market":        (0, 1),
    "ultra short":         (0, 1),
    "low duration":        (0, 1),
    "assured return":      (0, 0),   # legacy Capital-Protected schemes
    "arbitrage":           (0, 1),   # exploits cash-futures spread; near-zero net risk
    # ── Conservative debt / defensive hybrid ─────────────────────────────────
    "equity savings":      (1, 2),   # Low – Medium  (small equity hedge component)
    "conservative hybrid": (1, 2),
    "short duration":      (1, 2),
    "banking and psu":     (1, 2),
    "corporate bond":      (1, 2),
    "floater":             (1, 2),
    "gilt":                (1, 3),   # Low – High  (long duration = rate-risk can spike)
    # ── Moderate: dynamic debt, balanced, solution-oriented ──────────────────
    "dynamic bond":        (2, 3),   # Medium floor (manager can take long duration bets)
    "credit risk":         (2, 3),   # credit spread adds downside vs sovereign
    "medium duration":     (2, 3),   # catches "Medium Duration" and "Medium to Long"
    "long duration":       (2, 3),
    "balanced advantage":  (2, 3),
    "dynamic asset":       (2, 3),
    "multi asset":         (2, 3),
    "aggressive hybrid":   (2, 3),
    "balanced hybrid":     (2, 3),
    "solution oriented":   (2, 3),   # retirement / children — medium floor
    "fof domestic":        (2, 3),
    "gold etf":            (2, 3),   # alternative asset — checked before bare "etf"
    # ── Diversified equity (actively managed) ─────────────────────────────────
    "large & mid":         (2, 3),   # checked before bare "large"
    "large cap":           (2, 3),
    "flexi cap":           (2, 3),
    "multi cap":           (2, 3),
    "dividend yield":      (2, 3),
    "value":               (2, 3),
    "focused":             (2, 3),
    "contra":              (2, 3),
    # ── High-volatility equity ────────────────────────────────────────────────
    "mid cap":             (3, 4),   # High – Very_High floor
    "small cap":           (3, 4),
    "micro cap":           (3, 4),
    "elss":                (3, 4),
    "sectoral":            (3, 4),
    "thematic":            (3, 4),
    "fof overseas":        (3, 4),   # currency + foreign equity risk
    # ── Broad (type-dependent) ────────────────────────────────────────────────
    "index":               (2, 4),   # large-cap index ≈ Medium; small-cap ≈ Very_High
    "etf":                 (2, 4),   # generic ETFs span Medium–Very_High
}

# ─── PCA feature weights (single source of truth) ───────────────────────────
# Weights derived EXCLUSIVELY from sign-corrected PCA PC1 loadings.
# No domain assumptions, no manual overrides, no hybrid scoring.
#
#   Step 1  raw_loading  = pca.components_[0]         (sklearn PC1 eigenvector)
#   Step 2  sign_loading = sign_factor × raw_loading
#           sign_factor  = -1  (Credit_Mix_Score raw loading is negative;
#                               flip so higher score = more risk-tolerant)
#   Step 3  weight       = sign_loading × 2.5 / max(|sign_loading|)
#           clamped to [-2.5, +2.5]
#
# Source: models/pca_risk.joblib  (fit_pca_risk_model() in src/risk_labeling.py)
#   PC1 explains 32.48 % of variance;  PC1+PC2 cumulative = 49.99 %
#
# This dict is the SINGLE source of truth — imported by risk_labeling.py and
# referenced directly by RISK_MATRIX_DIMENSIONS below, ensuring the global risk
# scorer and the 4-dimension explainability view use IDENTICAL PCA-derived weights.
#
# Rank  Feature                    Raw_PC1   Sign×PC1   Weight
#  1    Credit_Mix_Score           -0.5247   +0.5247    +2.50
#  2    Interest_Rate_norm         +0.4177   -0.4177    -1.99
#  3    Credit_History_Score       -0.3559   +0.3559    +1.70
#  4    Num_Bank_Accounts_norm     +0.3449   -0.3449    -1.64
#  5    Delay_Score                +0.3297   -0.3297    -1.57
#  6    Monthly_Inhand_Salary_norm -0.2512   +0.2512    +1.20
#  7    Annual_Income_norm         -0.2104   +0.2104    +1.00
#  8    Age_Risk_Proxy             +0.1784   -0.1784    -0.85
#  9    Debt_Burden_Ratio          +0.1636   -0.1636    -0.78
# 10    Spending_Behaviour_Score   -0.1291   +0.1291    +0.62
# 11    Credit_Utilization_Ratio   -0.0646   +0.0646    +0.31
# 12    EMI_Income_Ratio           +0.0635   -0.0635    -0.30
# 13    Savings_Rate               -0.0612   +0.0612    +0.29
# 14    Investment_Ratio           +0.0345   -0.0345    -0.16
# 15    Occupation_Stability_Score -0.0083   +0.0083    +0.04
FEATURE_WEIGHTS: dict = {
    "Credit_Mix_Score":             +2.50,
    "Interest_Rate_norm":           -1.99,
    "Credit_History_Score":         +1.70,
    "Num_Bank_Accounts_norm":       -1.64,
    "Delay_Score":                  -1.57,
    "Monthly_Inhand_Salary_norm":   +1.20,
    "Annual_Income_norm":           +1.00,
    "Age_Risk_Proxy":               -0.85,
    "Debt_Burden_Ratio":            -0.78,
    "Spending_Behaviour_Score":     +0.62,
    "Credit_Utilization_Ratio":     +0.31,
    "EMI_Income_Ratio":             -0.30,
    "Savings_Rate":                 +0.29,
    "Investment_Ratio":             -0.16,
    "Occupation_Stability_Score":   +0.04,
}

# ─── Multi-Metric Risk Matrix dimensions ─────────────────────────────────────
# Each customer's risk is decomposed into 4 orthogonal sub-scores.
# Weights are taken DIRECTLY from FEATURE_WEIGHTS above (the PCA PC1 loadings)
# so that the 4-dimension explainability view is methodologically consistent with
# and traceable to the PCA analysis — no separate hand-crafted values.
# Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — multi-factor suitability
RISK_MATRIX_DIMENSIONS = {
    "Financial_Capacity": {
        "features": ["Annual_Income_norm", "Monthly_Inhand_Salary_norm",
                      "Savings_Rate", "EMI_Income_Ratio"],
        "weights":  [FEATURE_WEIGHTS["Annual_Income_norm"],
                     FEATURE_WEIGHTS["Monthly_Inhand_Salary_norm"],
                     FEATURE_WEIGHTS["Savings_Rate"],
                     FEATURE_WEIGHTS["EMI_Income_Ratio"]],
        "description": "Ability to absorb financial loss (income, savings buffer, debt load)",
    },
    "Behavioral_Tolerance": {
        "features": ["Investment_Ratio", "Spending_Behaviour_Score",
                      "Credit_Mix_Score"],
        "weights":  [FEATURE_WEIGHTS["Investment_Ratio"],
                     FEATURE_WEIGHTS["Spending_Behaviour_Score"],
                     FEATURE_WEIGHTS["Credit_Mix_Score"]],
        "description": "Willingness to take risk (investment habits, spending pattern)",
    },
    "Time_Horizon": {
        "features": ["Age_Risk_Proxy"],
        "weights":  [FEATURE_WEIGHTS["Age_Risk_Proxy"]],
        "description": "Investment time horizon (younger = longer = more tolerant)",
    },
    "Credit_Health": {
        "features": ["Debt_Burden_Ratio", "Delay_Score",
                      "Credit_Utilization_Ratio", "Interest_Rate_norm",
                      "Credit_History_Score", "Num_Bank_Accounts_norm",
                      "Occupation_Stability_Score"],
        "weights":  [FEATURE_WEIGHTS["Debt_Burden_Ratio"],
                     FEATURE_WEIGHTS["Delay_Score"],
                     FEATURE_WEIGHTS["Credit_Utilization_Ratio"],
                     FEATURE_WEIGHTS["Interest_Rate_norm"],
                     FEATURE_WEIGHTS["Credit_History_Score"],
                     FEATURE_WEIGHTS["Num_Bank_Accounts_norm"],
                     FEATURE_WEIGHTS["Occupation_Stability_Score"]],
        "description": "Credit discipline and financial stability",
    },
}

# ─── Investment horizon-based allocation (SEBI glide-path) ───────────────────
# Maps (risk_label, horizon_years) → target equity % (rest = debt/hybrid).
# Short horizons push towards capital preservation; long horizons allow full
# risk-tier equity exposure.
# Ref: AMFI Investor Education — "Goal-Based Asset Allocation"; SEBI 2019.
HORIZON_EQUITY_GLIDE = {
    #              1yr    3yr    5yr    10yr+
    "Very_Low":  [0.00,  0.00,  0.05,  0.10],
    "Low":       [0.00,  0.10,  0.20,  0.30],
    "Medium":    [0.10,  0.30,  0.50,  0.60],
    "High":      [0.15,  0.50,  0.75,  0.85],
    "Very_High": [0.20,  0.60,  0.90,  0.95],
}
HORIZON_LABELS = ["1yr", "3yr", "5yr", "10yr+"]

# ─── Core-Satellite diversified portfolio strategy ───────────────────────────
# Core: user's exact risk tier.  Satellite: adjacent tiers for diversification.
# Ref: BlackRock "Core-Satellite Investing" white paper; Vanguard 2022.
CORE_SATELLITE_SPLIT = {
    "core_pct":     0.60,   # 60% in user's own risk tier
    "stability_pct": 0.20,  # 20% one tier below (stability anchor)
    "growth_pct":    0.20,  # 20% one tier above (growth kicker)
}

# ─── Horizon-specific fund category overrides ────────────────────────────────
# For very short horizons, certain categories are always preferred regardless of
# risk tier — SEBI suitability: "high risk for short duration is unsuitable".
HORIZON_CATEGORY_OVERRIDE = {
    "1yr": [
        "Liquid Fund", "Overnight Fund", "Ultra Short Duration",
        "Money Market", "Low Duration", "Arbitrage",
    ],
    "3yr": [
        "Short Duration", "Banking and PSU", "Corporate Bond",
        "Balanced Advantage", "Conservative Hybrid", "Dynamic Asset Allocation",
        "Floater", "Gilt",
    ],
}
