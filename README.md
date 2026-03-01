# Smart Fund Advisor

> **Federated-Learning-based Risk Appetite Assessment + Mutual Fund Recommender**  
> *Privacy-Preserving Personalized Financial Advisory (v3)*

---

## What's New in v3

| Feature | Description |
|---------|-------------|
| **Multi-Metric Risk Matrix** | 4-dimension risk decomposition (Financial Capacity, Behavioral Tolerance, Time Horizon, Credit Health) — SEBI suitability compliant |
| **Horizon-Based Recommendations** | Different fund allocations for 1yr / 3yr / 5yr / 10yr+ investment horizons following SEBI glide-path |
| **Core-Satellite Portfolio** | 60% core (user's tier) + 20% stability (tier below) + 20% growth (tier above) — BlackRock-inspired diversification |
| **LLM Upgrade** | Groq Llama-3.3-70B, OpenRouter Gemma-2-9B, HuggingFace Qwen-2.5-3B — all free tier |
| **3-Model Ensemble** | XGBoost (0.40) + RandomForest (0.35) + LightGBM (0.25) fund scoring |
| **15-Feature Engineering** | 3 new v2 features (EMI_Income_Ratio, Savings_Rate, Credit_History_Score) |
| **13 System Tests** | Up from 10 — new tests for risk matrix, horizon recs, core-satellite |
| **Two-Step Fund Risk Classification** | `compute_fund_risk_bands()`: per-category z-score → P20/P40/P60/P80 bands → SEBI floor/ceiling clamp (`CATEGORY_RISK_BOUNDS`, 43 entries); 14,270 funds via NAV history, 2,076 via keyword fallback |
| **100% Fund Category Coverage** | `RISK_TO_FUND_CATEGORIES` rewritten with 52 keywords; all pre-SEBI-2018 labels, ETF, FoF, Solution-Oriented, Credit Risk, Contra, Focused mapped — 0 unmapped funds |
| **AMC Diversity Enforcement** | `recommend_funds()` pre-caps each AMC at `ceil(top_n × 0.40)` slots; `allocate_portfolio()` uses iterative convergent loop for weight-cap compliance |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CENTRAL SERVER                                  │
│                                                                              │
│  ┌──────────────────────┐    ┌────────────────────────────────────────────┐ │
│  │  RiskMLP              │    │  Mutual Fund Recommendation Engine (v3)    │ │
│  │  (global weights)     │◄FL─│  ┌──────────────────────────────────────┐ │ │
│  │  15→256→128→64→32→5   │    │  │ XGB + RF + LightGBM Ensemble (v2)   │ │ │
│  │  GELU + Residual      │    │  │ 19 features (12 base + 7 from NAV)   │ │ │
│  │  FocalLoss            │    │  │ Real-return target (CAGR/Sharpe/DD)  │ │ │
│  └──────────────────────┘    │  └──────────────────────────────────────┘ │ │
│         ↑ FedProx-FedAvg     │  ┌──────────────────────────────────────┐ │ │
│         │ (DP-noised Δw only) │  │ NAV History Engine (21M rows parquet)│ │ │
│         │                    │  └──────────────────────────────────────┘ │ │
│         │  ┌──────────────┐  │  ┌──────────────────────────────────────┐ │ │
│         │  │ Risk Matrix  │  │  │ Horizon-Based Allocation (v3)       │ │ │
│         │  │ 4 dimensions │  │  │ SEBI glide-path: 1yr/3yr/5yr/10yr+  │ │ │
│         │  └──────────────┘  │  └──────────────────────────────────────┘ │ │
│         │                    │  ┌──────────────────────────────────────┐ │ │
│         │                    │  │ Core-Satellite Portfolio (v3)        │ │ │
│         │                    │  │ 60% core + 20% stability + 20% growth│ │ │
│         │                    │  └──────────────────────────────────────┘ │ │
│         │                    │  + LLM Explanation (Llama 70B / Gemma 9B) │ │
│         │                    └────────────────────────────────────────────┘ │
│         │                                    ↓ Top-N scored fund list       │
└─────────┼──────────────────────────────────────────────────────────────────-┘
          │  DP-noised weight deltas (raw bank data NEVER leaves device)
          │
┌─────────┴───────────────────────────────────────────────────────────────────┐
│              USERS' MOBILE DEVICES  (FL cohort — 30% of users)              │
│                                                                              │
│  📱 CUS_0x1000        📱 CUS_0x1009   …   📱 CUS_0xNNNN                     │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐                   │
│  │ own bank   │       │ own bank   │       │ own bank   │  ← raw data       │
│  │ records    │       │ records    │       │ records    │    stays here      │
│  │ only       │       │ only       │       │ only       │                   │
│  │ (≤ 4 rows) │       │ (≤ 4 rows) │       │ (≤ 4 rows) │                   │
│  └────────────┘       └────────────┘       └────────────┘                   │
│                                                                              │
│  Each device trains locally → sends DP-noised Δw → raw data NEVER shared   │
│  Data isolation: ✓ GUARANTEED  (groupby Customer_ID, 0 overlaps)            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## System Design Alignment

The system is designed to:
> *"Use federated learning to assess users' risk appetite directly on their device, ensuring that sensitive bank details remain private. Only non-identifying attributes are shared for broader analysis."*

Implementation matches:
- Each user's mobile device runs local model training on **that user's own bank records only** (≤ 4 monthly rows)
- Only DP-noised gradient weight updates are sent to the server — **no raw data shared**
- Central server aggregates updates via FedProx-FedAvg to improve the global risk model
- Mutual fund recommendation engine + GenAI explanation runs on the central server

---

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Risk model | MLP 15→256→128→64→32→5 (GELU, residual, FocalLoss) | Deeper network for 15 features; GELU smoother gradients; residual for training stability |
| Risk matrix | 4-dimension decomposition (Financial, Behavioral, Horizon, Credit) | SEBI multi-factor suitability profiling (SEBI/HO/IMD/DF2/CIR/P/2019/17) |
| FL clients | 1 device per customer (3,750 devices) | One virtual device per FL user; each device holds only that user's own records |
| Per-device data | Only that customer's own records (≤ 4 rows) | Sensitive bank data stays private |
| FL strategy | **FedProx** proximal term + FedAvg | Prevents client drift across heterogeneous devices |
| Privacy | Manual gradient clipping C=1.0 + Gaussian DP noise σ=1.0 | (ε,δ)-DP; ε=10.01, δ=1e-5 |
| Fund scoring | **XGBoost + RF + LightGBM** (3-model ensemble, 0.40+0.35+0.25) | Objective fund quality ranking via ensemble predictions |
| Horizon allocation | SEBI glide-path (1yr→10yr+, 5×4 matrix) | Short horizons → capital preservation; long → growth |
| Portfolio strategy | Core-Satellite (60/20/20) | BlackRock + Vanguard research: diversification across adjacent risk tiers |
| LLM stack | Llama-3.3-70B → Gemma-2-9B → Qwen-2.5-3B → rule-based | Best free models: 70B for reasoning, 9B for instruction-following, 3B for no-API fallback |
| Clustering | KMeans on **temperature-softmax vectors** (T=0.2) | Silhouette 0.24 → **0.9472** using learned risk-space representations |
| NAV history | **21M-row parquet** chunked row-group reader | ~500 MB; scanned in ~14s; ref date auto-resolved from parquet max-date |

---

## Evaluation Results

| # | Metric | Threshold | Result | Status |
|---|--------|-----------|--------|--------|
| 1 | Cluster Silhouette (temperature-softmax embeddings) | ≥ 0.80 | **0.9472** | ✅ PASS |
| 2 | Macro F1 (FL global model, pseudo-label mode) | > 0.80 | **0.9611** | ✅ PASS |
| 3 | DP Privacy (ε, δ) | formal guarantee | **ε=10.0087, δ=1e-5** | ✅ PASS |
| 4 | FL Loss Stability (prediction change) | < 10% | **0.19%** | ✅ PASS |
| 5 | GenAI Fund Explanation Correctness | ≥ 75% | **100%** | ✅ PASS |

**5/5 evaluation metrics pass (incremental pseudo-label FL mode). System tests: 13/13 pass (v3).**

---

## Project Structure

```
20Feb26/
├── environment.yml                   # conda environment spec
├── config.py                         # all hyper-parameters & paths
├── train.py                          # full 8-step pipeline CLI runner
├── test_system.py                    # 13-test system validation suite (v3)
├── ARCHITECTURE.md                   # detailed architecture notes (v3)
│
├── Data/
│   ├── bank_user_dataset.csv         # 12,500 customers, synthetic
│   ├── mutual_fund_data.csv          # SEBI-registered schemes (16,346 total; 14,270 with NAV history)
│   └── mutual_fund_nav_history.parquet  # 21M+ rows, 14,427 schemes, 2006–2026
│
├── src/
│   ├── preprocessing.py              # data cleaning, feature engineering
│   ├── risk_labeling.py              # 5-class risk label derivation
│   ├── central_model.py              # RiskMLP definition + central training
│   ├── fl_simulation.py              # FedProx mobile-device FL simulation
│   │                                 #   _local_train()                  — FedProx + Adam + DP per device
│   │                                 #   _fedavg()                       — weighted FedAvg aggregation
│   │                                 #   run_fl_simulation()             — batch mode: all 3,750 devices from round 1
│   │                                 #   run_incremental_fl_simulation() — production-realistic: 5 waves of 750 users
│   │                                 #                                     new users onboard via on-device pseudo-labelling
│   │                                 #                                     global_model predicts label; server never assigns labels
│   ├── fl_client.py                  # per-device FL client logic (local train + DP noise)
│   ├── cluster_recommender.py        # KMeans on model probability embeddings
│   │                                 #   fit_cluster_model()           — raw-feature baseline
│   │                                 #   fit_embedding_cluster_model() — temperature-softmax T=0.2
│   ├── recommender.py                # mutual fund recommender (tier filter + horizon + core-satellite)
│   ├── ensemble_recommender.py       # XGBoost + RF + LightGBM fund-scoring ensemble (3-model)
│   │                                 #   build_fund_features()   — 11 or 18 feature matrix
│   │                                 #   fit_fund_ensemble()     — trains RF + XGB, saves artefacts
│   │                                 #   score_funds_ensemble()  — inference-time scoring
│   ├── nav_history.py                # historical NAV parquet reader + metric engine
│   │                                 #   iter_parquet_chunks()       — row-group chunked reader
│   │                                 #   compute_all_metrics()       — 10 metrics for 14,427 schemes
│   │                                 #   load_nav_metrics()          — cache-aware loader
│   │                                 #   nav_history_quick_stats()   — fast metadata summary
│   │                                 #   compute_fund_risk_bands()   — two-step realized-risk classifier
│   ├── privacy_analysis.py           # (ε,δ)-DP accounting & σ→ε trade-off table
│   ├── gpt_explainer.py              # LLM fund explanation (Llama-3.3-70B / Gemma-2-9B / Qwen-2.5-3B)
│   ├── evaluation.py                 # 5 evaluation metrics (cluster, F1, privacy, FL stability, GPT)
    ├── utils.py                      # plot helpers (training curves, FL history, risk dist.)
    └── drift_detector.py             # concept drift detection: PSI, KS test, distribution shift alerts
│
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb   # EDA + features + NAV history quick stats
│   ├── 02_Central_Model_Training.ipynb
│   ├── 03_Federated_Learning_Simulation.ipynb
│   ├── 04_Recommendation_Engine.ipynb         # ensemble training, NAV charts, top performers
│   ├── 05_End_to_End_Demo.ipynb
│   └── 06_Evaluation_Metrics.ipynb
│
└── models/                           # auto-created on first run
    ├── central_risk_model.pt         # RiskMLP weights (central training)
    ├── fl_global_risk_model.pt       # RiskMLP weights (after FL)
    ├── feature_scaler.joblib         # MinMaxScaler fit on training data
    ├── label_encoder.joblib          # LabelEncoder for risk classes
    ├── cluster_kmeans.joblib         # KMeans (raw features)
    ├── cluster_kmeans_embed.joblib   # KMeans (temperature-softmax embeddings)
    ├── cluster_metadata.json
    ├── cluster_metadata_embed.json
    ├── rf_fund_model.joblib          # Random Forest fund scorer
    ├── xgb_fund_model.joblib         # XGBoost fund scorer
    ├── fund_feature_cols.joblib      # feature column list for ensemble
    ├── ensemble_fund_meta.json       # RF R², XGB R², feature list, uses_history flag
    ├── nav_metrics.csv               # cached NAV metrics (14,427 schemes)
    ├── central_training_history.json
    ├── fl_training_history.json
    ├── evaluation_results.json
    └── plot_*.png                    # training curves, confusion matrices, dashboards
```

---

## Quick Start

### 1. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate smart_fund_advisor
```

### 2. Run the full pipeline

```bash
python train.py                         # batch FL (all 3,750 devices from round 1)
python train.py --incremental           # production-realistic FL (users join in 5 waves of 750)
python train.py --incremental --waves 5 --rounds-per-wave 3  # explicit (same defaults)
```

8-step pipeline (~3–4 minutes):
1. Data preprocessing & feature engineering (12,500 customers)
2. Risk label derivation (5 balanced classes, bell-curve bins) — **central users only**
3. Central model training — RiskMLP, 70% customers, 30 epochs
4. FL simulation — 3,750 virtual mobile devices, 15 rounds, FedProx + DP  
   *Incremental mode*: users join in 5 waves of 750; each new user's risk label is **predicted on-device** by the current global model (pseudo-label) — server never assigns labels to FL users
5. Differential Privacy accounting (ε=10.0087, δ=1e-5)
6. KMeans cluster analysis on temperature-softmax embeddings (silhouette = 0.9472)
7. Load historical NAV metrics (21M rows, cached) → train XGB+RF ensemble → recommendations + GenAI explanation
8. Evaluation — **5/5 PASS**

### 3. Run system tests

```bash
python test_system.py
```

Expected: **13/13 tests pass** (v3 — includes risk matrix, horizon, core-satellite tests)

---

## Feature Engineering

| Feature | Formula | Direction |
|---------|---------|-----------|
| `Investment_Ratio` | `Amount_invested_monthly / Monthly_Inhand_Salary` | ↑ = more risk tolerant |
| `Debt_Burden_Ratio` | `Outstanding_Debt / Annual_Income` | ↑ = less risk tolerant |
| `Annual_Income_norm` | MinMax scaled | ↑ = more capacity |
| `Monthly_Inhand_Salary_norm` | MinMax scaled | ↑ = more capacity |
| `Credit_Utilization_Ratio` | Raw, scaled | ↑ = more stressed |
| `Delay_Score` | `Delay_from_due + Num_Delayed_Payment` | ↑ = less disciplined |
| `Credit_Mix_Score` | Good=1.0, Standard=0.5, Bad=0.0 | ↑ = better credit |
| `Spending_Behaviour_Score` | Encoded Payment_Behaviour | ↑ = more aggressive |
| `Num_Bank_Accounts_norm` | MinMax scaled | ↑ = more financial engagement |
| `Interest_Rate_norm` | MinMax scaled | ↑ = paying more = stressed |
| `Age_Risk_Proxy` | `clip((70−Age)/52, 0, 1)` — Age clipped to [18, 70] | ↑ = younger = longer horizon = more tolerant (SEBI suitability) |
| `Occupation_Stability_Score` | Profession→[0,1] lookup (e.g. Doctor=0.90, Mechanic=0.48) | ↑ = more stable income = greater risk capacity (SEBI/AMFI) |

---

## Risk Label Derivation

No explicit label exists in the dataset — engineered via composite score:

| Feature | Weight |
|---------|--------|
| `Investment_Ratio` | +3.0 |
| `Spending_Behaviour_Score` | +2.5 |
| `Annual_Income_norm` | +2.0 |
| `Debt_Burden_Ratio` | -2.5 |
| `Delay_Score` | -2.0 |
| `Monthly_Inhand_Salary_norm` | +1.5 |
| `Credit_Mix_Score` | +1.5 |
| `Age_Risk_Proxy` | +1.5 |
| `Occupation_Stability_Score` | +1.2 |
| `Credit_Utilization_Ratio` | -1.0 |
| `Num_Bank_Accounts_norm` | +0.5 |
| `Interest_Rate_norm` | -0.5 |

Score binned at equal-frequency quintiles:  
`Very_Low (0) → Low (1) → Medium (2) → High (3) → Very_High (4)`

---

## Multi-Metric Risk Matrix (v3)

Beyond the single composite score, each customer's risk is decomposed into 4 orthogonal sub-dimensions (all normalised [0, 1]):

| Dimension | Features | Weights | Meaning |
|-----------|----------|---------|---------|
| **Financial Capacity** | Income, Salary, Savings_Rate, EMI_Ratio | +2.0, +1.5, +1.5, −1.8 | Ability to absorb financial loss |
| **Behavioral Tolerance** | Investment_Ratio, Spending, Credit_Mix | +3.0, +2.5, +1.5 | Willingness to take risk |
| **Time Horizon** | Age_Risk_Proxy | +1.5 | Years available for compounding |
| **Credit Health** | Debt_Burden, Delay, CU_Ratio, Interest, Credit_History, Accounts, Occupation | −2.5, −2.0, −1.0, −0.5, +0.8, +0.5, +1.2 | Credit discipline |

**Composite** = equal-weight average of all 4 normalised dimensions.

This enables richer LLM explanations and allows advisors to see *why* a user is classified at a certain risk level.

Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17; Grable & Lytton (1999)

---

## Horizon-Based Fund Recommendations (v3)

Different investment horizons warrant different asset allocations. The system implements a SEBI-aligned glide-path matrix:

| Risk \ Horizon | 1yr | 3yr | 5yr | 10yr+ |
|----------------|-----|-----|-----|-------|
| Very_Low | 0% equity | 0% | 5% | 10% |
| Low | 0% | 10% | 20% | 30% |
| Medium | 10% | 30% | 50% | 60% |
| High | 15% | 50% | 75% | 85% |
| Very_High | 20% | 60% | 90% | 95% |

**Short-horizon overrides**: For 1yr investments, the system forces safe categories (Liquid, Overnight, Ultra Short Duration) regardless of risk appetite — SEBI suitability mandates capital preservation for short durations.

```bash
# Example: High risk user with different horizons
# 1yr → 15% equity (Arbitrage) + 85% debt (Liquid/Overnight) — capital preservation
# 5yr → 75% equity (Large Cap, Flexi Cap) + 25% debt — balanced growth
# 10yr → 85% equity (full risk-tier exposure) + 15% debt — maximum growth
```

Ref: AMFI "Goal-Based Asset Allocation"; SEBI 2019 suitability circular

---

## Core-Satellite Diversified Portfolio (v3)

Every portfolio is constructed across multiple risk brackets:

| Bracket | Allocation | Purpose |
|---------|-----------|---------|
| **Core** | 60% | User's own risk tier — primary return driver |
| **Stability** | 20% | One tier below — downside buffer |
| **Growth** | 20% | One tier above — upside kicker |

Example for a "High" risk user with ₹5,00,000 corpus (5yr horizon):
- Core (60%): Large Cap, Flexi Cap, Multi Cap — ₹3,00,000
- Stability (20%): Balanced Hybrid, Conservative Hybrid — ₹1,00,000
- Growth (20%): Mid Cap, Small Cap, ELSS — ₹1,00,000

**Edge cases**: Very_Low → no tier below → 80% core + 20% growth. Very_High → no tier above → 80% core + 20% stability.

Each bracket enforces AMC concentration cap (≤40%) and computes HHI diversification score.

Ref: BlackRock "Core-Satellite Investing" (2018); Vanguard "Diversification" (2022); Markowitz (1952)

---

## LLM Provider Stack (v3)

| Priority | Provider | Model | Params | Benchmark | Cost |
|----------|----------|-------|--------|-----------|------|
| 1 | Groq | llama-3.3-70b-versatile | 70B | MT-Bench 8.6 | Free |
| 2 | OpenRouter | google/gemma-2-9b-it:free | 9B | MMLU 71.3% | Free |
| 3 | HuggingFace | Qwen/Qwen2.5-3B-Instruct | 3B | IFEval 76.1% | Free |
| 4 | Rule-based | — | — | 100% factual | Free |

All use temperature=0.3, SEBI-advisor system prompt, max 200 words output.

---

## Federated Learning Details

| Parameter | Value |
|-----------|-------|
| Architecture | Manual FedProx + FedAvg (no Flower/Opacus) |
| FL mode | **Batch** (default): all 3,750 devices from round 1  ·  **Incremental** (`--incremental`): users join in 5 waves of 750; each new user's label generated on-device by the current global model (pseudo-label — no central labelling) |
| FL virtual devices | 3,750 (one per customer in the 30% FL cohort) |
| Records per device | ≤ 4 (most-recent monthly bank records) |
| Data on device | Only that customer's own bank records |
| Data shared with server | DP-noised gradient weight deltas **only** |
| Rounds | 15 (5 waves × 3 rounds in incremental mode) |
| Devices sampled per round | ~1,125 (30% of active pool) |
| Straggler dropout | 15% of sampled devices dropped per round |
| Local epochs | Uniform(2–8) (variable per device per round) |
| Local batch size | 4 (matches per-device record count) |
| Optimizer | Adam (lr=0.001) |
| FedProx µ | Adaptive: 0.01 × 0.9^(round−1) |
| BatchNorm | Eval mode (running stats from central model) |
| DP clip norm C | 1.0 |
| DP noise σ | 1.0 |
| Privacy budget ε | 10.01 (δ=1e-5) |

### Incremental FL — Pseudo-Label Wave Results

In incremental mode the global model labels each new wave of users on-device. Both metrics are reported per wave:

| Wave | New Users | Pool Size | Rounds | PseudoAcc | EndAcc (vs oracle) |
|------|-----------|-----------|--------|-----------|-------------------|
| 1 | 750 | 750 | 1–3 | 0.9693 | 0.9644 |
| 2 | 750 | 1,500 | 4–6 | 0.9533 | 0.9733 |
| 3 | 750 | 2,250 | 7–9 | 0.9707 | 0.9615 |
| 4 | 750 | 3,000 | 10–12 | 0.9560 | 0.9656 |
| 5 | 750 | 3,750 | 13–15 | 0.9600 | 0.9573 |

- **PseudoAcc** = fraction of new-arrivals where the current global model's predicted label matches the rule-based oracle label (measures label quality; stays above 0.95 throughout)
- **EndAcc (vs oracle)** = global model prediction accuracy on oracle labels at end of wave (Evaluation Metric 2)

### FedProx Objective (per device)

$$L_\text{local}(w) = L_\text{CE}(w) + \frac{\mu}{2} \| w - w_\text{global} \|^2$$

The proximal term pulls local device weights back toward the global model, preventing individual devices from over-specialising on their 1–4 private records.

---

## Embedding-Based Clustering

| Method | Silhouette | DB Index |
|--------|-----------|----------|
| Raw 10-dim features | 0.24 | ~1.4 |
| Model penultimate layer (32-dim) | 0.48 | 0.67 |
| **Temperature-softmax (5-dim, T=0.2)** | **0.9472** | **0.09** |

Temperature-scaled softmax turns model confidence scores into near-one-hot probability vectors. KMeans on these vectors naturally recovers the 5 risk classes.

---

## Ensemble Fund-Scoring Model

Two-model ensemble (soft vote: 0.5 × RF + 0.5 × XGB) scores every mutual fund on a continuous quality scale.

### Features (19 when NAV history available, 12 otherwise)

**Base features (12):** `log_aum`, `nav_recency_days`, `fund_age_years`, `risk_tier_num`, `expense_ratio`, `return_1yr`, `return_3yr`, `return_5yr`, `is_direct`, `has_nav`, `is_large_aum`, `expense_ratio_norm`

**Historical NAV features (7, added when parquet is present):**

| Feature | Metric |
|---------|--------|
| `cagr_1yr_hist` | CAGR last 1 year (ref: 2026-02-15) |
| `cagr_3yr_hist` | CAGR last 3 years |
| `cagr_5yr_hist` | CAGR last 5 years |
| `vol_1yr_hist` | Annualised volatility (1yr, floor 0.2%) |
| `sharpe_1yr_hist` | Sharpe ratio (rf=6.5%, clipped ±10) |
| `max_drawdown_hist` | Maximum peak-to-trough decline |
| `momentum_6m_hist` | 6-month price momentum |

### Quality Target

$$Q = 0.35 \times \text{CAGR}_{3\text{yr}} + 0.25 \times \text{Sharpe}_{1\text{yr}} + 0.25 \times (-\text{MaxDD}) + 0.15 \times \text{Momentum}_{6\text{m}} - 0.10 \times \text{TER}_{\text{norm}}$$

### Ensemble Performance

| Model | Train R² | CV R² |
|-------|---------|-------|
| Random Forest | 0.9985 | 0.9608 ± 0.063 |
| XGBoost | 0.9977 | 0.9750 ± 0.041 |

---

## Historical NAV Analysis (`src/nav_history.py`)

| Property | Value |
|----------|-------|
| Source file | `Data/mutual_fund_nav_history.parquet` |
| Total rows | 21,357,943 |
| Unique schemes | 14,427 |
| Date range | 2006-04-01 → 2026-02-15 |
| Reference date | **Auto-detected** from parquet `max(Date)` = **2026-02-15** |
| Row groups | 21 (~1 M rows each) |
| Read strategy | Chunked row-group iteration (~14s full scan) |
| Cache | `models/nav_metrics.csv` — recomputed on demand via `load_nav_metrics()` |

**10 metrics computed per scheme:** CAGR 1/3/5yr · Annualised volatility · Sharpe · Sortino · Max drawdown · 6m momentum · NAV recency · Record count

---

## Mutual Fund Category → Risk Tier Mapping

### Step 1 — Primary: Data-Driven (NAV History)

`compute_fund_risk_bands()` in `src/nav_history.py` assigns tiers from realized volatility and drawdown for all funds with ≥5 history records (14,270 of 16,346):

1. For each `Scheme_Category` peer group: compute `risk_score = 0.5 × z_vol + 0.5 × z_drawdown` (within-category z-scores)
2. Bucket at P20 / P40 / P60 / P80 percentiles within each category
3. Clamp result to SEBI-aligned floor/ceiling from `CATEGORY_RISK_BOUNDS` (43 keyword rules)

### Step 2 — Fallback: Keyword Mapping

`RISK_TO_FUND_CATEGORIES` (52 keywords, all SEBI categories covered, 0% unmapped) is used for the 2,076 funds without NAV history:

| Risk Tier | Matched Scheme_Category Keywords |
|-----------|----------------------------------|
| Very_Low | Liquid, Overnight Fund, Ultra Short Duration, Money Market, Low Duration, Assured Return |
| Low | Short Duration, Banking and PSU, Corporate Bond, Floater, Gilt, Dynamic Bond, Credit Risk, Medium Duration, Long Duration, Income |
| Medium | Balanced Advantage, Dynamic Asset Allocation, Aggressive Hybrid, Conservative Hybrid, Balanced Hybrid, Balanced, Multi Asset Allocation, Arbitrage, Equity Savings, Solution Oriented, FoF Domestic, Gold ETF |
| High | Large & Mid Cap, Large Cap, Flexi Cap, Multi Cap, Value, Dividend Yield, Focused Fund, Contra, Growth |
| Very_High | Mid Cap, Small Cap, Micro Cap, ELSS, Sectoral, Thematic, Index, FoF Overseas, ETF |

### Coverage Summary

| Source | Funds | % of 16,346 |
|--------|-------|-------------|
| NAV history (data-driven + SEBI clamp) | 14,270 | 87.3% |
| Keyword fallback | 2,076 | 12.7% |
| Unmapped / excluded | 0 | 0.0% |

Funds ranked by **ensemble score** (XGB+RF+LGBM, 19 features); fallback: `0.7 × AUM score + 0.3 × NAV recency score`.

---

## Configuration (`config.py`)

| Parameter | Value | Description |
|-----------|-------|
| `CENTRAL_SPLIT` | 0.65 | Fraction of customers for central model (80/20 internal train/val split) |
| `FL_ROUNDS` | 15 | Federated communication rounds |
| `FL_MIN_CLIENTS` | 10 | Min devices sampled per round |
| `FL_FRACTION_FIT` | 0.30 | Fraction of devices per round |
| `FL_LOCAL_EPOCHS` | Uniform(2–8) | Variable local epochs per device per round |
| `FL_BATCH_SIZE` | 4 | Matches max per-device record count |
| `FEDPROX_MU` | 0.01 × 0.9^(round−1) | Adaptive FedProx proximal coefficient |
| `DP_NOISE_MULTIPLIER` | 1.0 | Gaussian DP noise σ |
| `DP_MAX_GRAD_NORM` | 1.0 | Gradient clipping norm |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `RANDOM_SEED` | 42 | NumPy / PyTorch seed |
| `NAV_HISTORY_PARQUET` | `Data/mutual_fund_nav_history.parquet` | Raw NAV history |
| `NAV_METRICS_CSV` | `models/nav_metrics.csv` | Cached per-scheme metrics |
| `CATEGORY_RISK_BOUNDS` | dict (43 entries) | SEBI floor/ceiling clamp per category keyword; used in `compute_fund_risk_bands()` step 3 |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.1 | Risk MLP model |
| `scikit-learn` | 1.4.1 | Preprocessing, KMeans, ensemble base |
| `xgboost` | 3.2.0 | XGBoost fund-scoring model |
| `lightgbm` | 4.6.0 | LightGBM fund-scoring model (3rd ensemble member) |
| `pandas` | 2.2.1 | Data wrangling |
| `pyarrow` | ≥ 14.0 | Parquet row-group reading (NAV history) |
| `numpy` | 1.26.4 | Numerics |
| `matplotlib` / `seaborn` | 3.8 / 0.13 | Visualisation |
| `joblib` | — | Model serialisation |
| `scipy` | — | KDE, statistical tests |

---

## Project Reference

> *Smart Fund Advisor: A Privacy-Preserving Federated Learning System for  
> Risk-Aware Mutual Fund Recommendations*  
> February 2026

## Dataset
https://www.kaggle.com/datasets/khanmdsaifullahanjar/bank-user-dataset
https://github.com/InertExpert2911/Mutual_Fund_Data/tree/main

## Sample Run Command
python train.py --skip-cluster --incremental > output.txt  
python demo_single_user.py --customer CUS_0x102d > demo.txt