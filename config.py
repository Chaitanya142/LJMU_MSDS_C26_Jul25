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

BANK_CSV            = DATA_DIR  / "bank_user_dataset.csv"
MF_CSV              = DATA_DIR  / "mutual_fund_data.csv"
NAV_HISTORY_PARQUET = DATA_DIR  / "mutual_fund_nav_history.parquet"
NAV_METRICS_CSV     = MODELS_DIR / "nav_metrics.csv"   # cached per-scheme metrics

CENTRAL_MODEL_PATH    = MODELS_DIR / "central_risk_model.pt"
FL_GLOBAL_MODEL       = MODELS_DIR / "fl_global_risk_model.pt"
SCALER_PATH           = MODELS_DIR / "feature_scaler.joblib"
LABEL_ENCODER_PATH    = MODELS_DIR / "label_encoder.joblib"
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
DP_NOISE_MULTIPLIER = 1.0  # Gaussian DP noise σ (higher σ → lower ε → stronger privacy)
DP_MAX_GRAD_NORM   = 1.0   # gradient clipping norm for DP sensitivity bound
FEDPROX_MU         = 0.01  # FedProx proximal coefficient (0 = pure FedAvg)

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

# ─── Multi-Metric Risk Matrix dimensions ─────────────────────────────────────
# Each customer's risk is decomposed into 4 orthogonal sub-scores.
# Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — multi-factor suitability
RISK_MATRIX_DIMENSIONS = {
    "Financial_Capacity": {
        "features": ["Annual_Income_norm", "Monthly_Inhand_Salary_norm",
                      "Savings_Rate", "EMI_Income_Ratio"],
        "weights":  [+2.0, +1.5, +1.5, -1.8],
        "description": "Ability to absorb financial loss (income, savings buffer, debt load)",
    },
    "Behavioral_Tolerance": {
        "features": ["Investment_Ratio", "Spending_Behaviour_Score",
                      "Credit_Mix_Score"],
        "weights":  [+3.0, +2.5, +1.5],
        "description": "Willingness to take risk (investment habits, spending pattern)",
    },
    "Time_Horizon": {
        "features": ["Age_Risk_Proxy"],
        "weights":  [+1.5],
        "description": "Investment time horizon (younger = longer = more tolerant)",
    },
    "Credit_Health": {
        "features": ["Debt_Burden_Ratio", "Delay_Score",
                      "Credit_Utilization_Ratio", "Interest_Rate_norm",
                      "Credit_History_Score", "Num_Bank_Accounts_norm",
                      "Occupation_Stability_Score"],
        "weights":  [-2.5, -2.0, -1.0, -0.5, +0.8, +0.5, +1.2],
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
