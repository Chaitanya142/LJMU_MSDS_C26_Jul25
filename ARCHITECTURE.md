# Smart Fund Advisor — System Architecture (v3)

## Overview

A privacy-preserving mutual fund recommendation system that:
1. Scores user **risk appetite** from banking transaction history using **15 engineered features** and a **multi-metric risk matrix** (4 sub-dimensions: Financial Capacity, Behavioral Tolerance, Time Horizon, Credit Health)
2. Trains a **neural network** (RiskMLP 15→256→128→64→32→5, GELU activations, residual connections, FocalLoss) on 65% of customers centrally
3. Fine-tunes with **Federated Learning** (FedProx + FedAvg, 15 rounds) across 3,750 virtual mobile devices with **Differential Privacy** (Gaussian σ=1.0)
4. Classifies 16,346 mutual funds into risk tiers via a **two-step data-driven mechanism** (per-category z-score → percentile bands → SEBI floor/ceiling clamp) then ranks with a **3-model ensemble** (XGBoost 0.40 + Random Forest 0.35 + LightGBM 0.25); 14,270 funds have full NAV history
5. Generates **horizon-based recommendations** (1yr/3yr/5yr/10yr+) following SEBI glide-path allocation
6. Builds **Core-Satellite diversified portfolios** (60% core + 20% stability + 20% growth) across multiple risk brackets
7. Sources **21M-row NAV history** from parquet (2006–2026-02-15) to compute CAGR, Sharpe, drawdown, momentum per scheme
8. **Explains** recommendations using free LLMs: Groq (Llama-3.3 70B) → OpenRouter (Gemma 2 9B) → HuggingFace (Qwen 2.5 3B) → rule-based fallback

---

## Pipeline Flow

```
bank_user_dataset.csv ──► Preprocessing (15 features) ──► Risk Labelling (bell-curve)
                                    │                              │
                                    ▼                              ▼
                         Multi-Metric Risk Matrix          Central Model Training
                         (4 sub-dimensions)                (RiskMLP + FocalLoss)
                                                                   │
mutual_fund_data.csv ──────────────────────────────────┐           │
                                                        │           ▼
                                         ┌──────────────┤   FL Simulation (FedProx + DP)
                                         │              │           │
                                         │              │           ▼
                                         │       RiskMLP → Risk Tier Prediction
                                         │              │
                                         │       Two-Step Risk Tier Assignment (16,346 funds)
                                         │       → NAV z-score bands (14,270) + keyword fallback (2,076)
                                         │       → SEBI floor/ceiling clamp (CATEGORY_RISK_BOUNDS)
                                         │              │
                                         │       Horizon Filter → Fund Pool
                                         │              │
                                         │       XGB + RF + LGBM Ensemble Scoring (19 features)
                                         │              │
                                         ├──────────────► Horizon-Based Recommendations
                                         │              │  (1yr/3yr/5yr/10yr+, SEBI glide-path)
                                         │              │
                                         └──────────────► Core-Satellite Portfolio
                                                        │  (60% core + 20% stability + 20% growth)
                                                        │
                                                        ▼
                                              GPT Explanation (Llama 70B / Gemma 9B / Qwen 3B)
                                                        │
                                                        ▼
                                              Correctness Validation

  ┄ ┄ ┄ Evaluation-only (no impact on recommendations) ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄
  KMeans(k=5) on temperature-softmax embeddings ──► Silhouette Score (Evaluation Metric 1)
```

---

## v3 Architecture: Multi-Metric Risk Matrix

Each customer's risk is decomposed into 4 orthogonal normalised sub-dimensions:

| Dimension | Features Used | Description |
|-----------|--------------|-------------|
| **Financial Capacity** | Annual_Income, Salary, Savings_Rate, EMI_Income_Ratio | Ability to absorb financial loss |
| **Behavioral Tolerance** | Investment_Ratio, Spending_Behaviour, Credit_Mix | Willingness to accept volatility |
| **Time Horizon** | Age_Risk_Proxy | Years available for compounding |
| **Credit Health** | Debt_Burden, Delay_Score, Credit_Utilisation, Interest_Rate, Credit_History, Accounts, Occupation | Financial discipline and stability |

```
  15 Features ──► RISK_MATRIX_DIMENSIONS (4 dim, weighted) ──► Min-Max Normalise [0,1]
                                                                         │
                  ┌─ Financial_Capacity   ──────────────────────┐        │
                  ├─ Behavioral_Tolerance ──────────────────────┤        │
                  ├─ Time_Horizon         ──────────────────────┤► Composite = mean(4 dims)
                  └─ Credit_Health        ──────────────────────┘
```

Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17; Grable & Lytton (1999)

---

## v3 Architecture: Horizon-Based Recommendations

Investment horizon determines equity/debt allocation via SEBI-aligned glide-path:

| Risk \ Horizon | 1yr | 3yr | 5yr | 10yr+ |
|----------------|-----|-----|-----|-------|
| Very_Low | 0% | 0% | 5% | 10% |
| Low | 0% | 10% | 20% | 30% |
| Medium | 10% | 30% | 50% | 60% |
| High | 15% | 50% | 75% | 85% |
| Very_High | 20% | 60% | 90% | 95% |

**Short horizon overrides**: For 1yr, regardless of risk tier, only safe categories are allowed (Liquid, Overnight, Ultra Short Duration, Money Market, Arbitrage). For 3yr, category is limited to Short Duration, Banking & PSU, Balanced, Gilt.

Ref: AMFI "Goal-Based Asset Allocation"; SEBI 2019 suitability circular

---

## v3 Architecture: Core-Satellite Portfolio Diversification

```
  User Risk: "High"    Horizon: 5yr
       │
       ├─ Core (60%) ────────► "High" tier funds (Large Cap, Flexi Cap, Multi Cap)
       │                        → primary return driver
       │
       ├─ Stability (20%) ──► "Medium" tier funds (Balanced, Hybrid, Dynamic)
       │                        → downside buffer anchor
       │
       └─ Growth (20%) ─────► "Very_High" tier funds (Mid Cap, Small Cap, ELSS)
                                → upside kicker satellite
```

**Edge case handling**:
- Very_Low risk → no tier below → stability 20% folds into core (80% core + 0% stability + 20% growth)
- Very_High risk → no tier above → growth 20% folds into core (80% core + 20% stability + 0% growth)

Within each bracket: AMC concentration cap (≤40% per AMC, enforced by iterative convergent loop), equal-weight distribution, HHI diversification scoring.

Ref: BlackRock "Core-Satellite Investing" (2018); Vanguard "Diversification" (2022); Markowitz (1952)

---

## v4 Architecture: Two-Step Data-Driven Fund Risk Classification

Fund risk tiers are assigned empirically from realized NAV history, not purely by keyword lookup.

```
  nav_metrics.csv (vol_1yr, max_drawdown per scheme)
       │
       ▼
  ── Step 1: Per-category z-score ──────────────────────────────────────────
  For each Scheme_Category peer group (≥ 5 funds with history):
    z_vol  = (vol_1yr  − μ_cat) / σ_cat
    z_dd   = (|max_drawdown| − μ_cat) / σ_cat
    risk_score = 0.5 × z_vol  +  0.5 × z_dd

  ── Step 2: Percentile band assignment ────────────────────────────────────
    ≤ P20 → Very_Low  |  ≤ P40 → Low  |  ≤ P60 → Medium
    ≤ P80 → High      |  > P80 → Very_High
  (thresholds computed within each category — not globally)

  ── Step 3: SEBI floor/ceiling clamp (CATEGORY_RISK_BOUNDS) ─────────────
  Example overrides:
    "overnight" → always Very_Low         (0, 0)
    "small cap" → floor High, ceil VH     (3, 4)
    "gilt"      → floor Low, ceil High    (1, 3)
    "index"     → floor Med, ceil VH      (2, 4)
  First keyword match in CATEGORY_RISK_BOUNDS wins (ordered dict, 43 entries).

  ── Fallback ─────────────────────────────────────────────────────────────
  Funds in categories with < 5 NAV history records → keyword tier from
  RISK_TO_FUND_CATEGORIES (52 keywords, 100% category coverage)

  ── Result ───────────────────────────────────────────────────────────────
  14,270 funds  →  risk_tier_source = "nav_history"
   2,076 funds  →  risk_tier_source = "keyword"   (no NAV history)
       0 funds  →  unmapped  (full coverage across all SEBI categories)
```

Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — product risk-o-meter.

---

## Module Map

| File | Responsibility |
|------|---------------|
| `src/preprocessing.py` | Load, clean, aggregate, engineer 15 risk features (12 financial + Age_Risk_Proxy + Occupation_Stability_Score + 3 v2 features) |
| `src/risk_labeling.py` | Weighted rule-based scoring → 5-class bell-curve label; **v3: `compute_risk_matrix()` — 4-dimension risk decomposition** |
| `src/central_model.py` | RiskMLP (15→256→128→64→32→5, GELU, residual, FocalLoss), train/predict/save/load |
| `src/fl_simulation.py` | Manual FedProx + FedAvg loop, 3,750 virtual devices; incremental: on-device pseudo-labelling |
| `src/fl_client.py` | Per-device FL client: local FedProx training + Gaussian DP noise |
| `src/privacy_analysis.py` | Rényi DP ε accounting, σ→ε trade-off table |
| `src/cluster_recommender.py` | KMeans(k=5) on temperature-softmax(T=0.2) embeddings — evaluation only |
| `src/recommender.py` | Fund filter by risk tier + ensemble ranking + `allocate_portfolio()` + **v3: `recommend_funds_by_horizon()`, `recommend_diversified_portfolio()`**; **v4: two-step tier assignment, AMC pre-cap (≤40%), iterative allocation cap** |
| `src/ensemble_recommender.py` | XGBoost + RF + LightGBM fund scorer (3-model ensemble, 19 features) |
| `src/nav_history.py` | 21M-row parquet reader; 10 per-scheme NAV metrics; cache layer; **v4: `compute_fund_risk_bands()` — two-step realized-risk classification with SEBI floor/ceiling clamp** |
| `src/gpt_explainer.py` | Free LLM: **v3: Groq Llama-3.3-70B / OpenRouter Gemma-2-9B / HF Qwen-2.5-3B** / rule-based |
| `src/evaluation.py` | All 5 evaluation metrics + dashboard plots |
| `src/utils.py` | Plot helpers (training curves, FL history, risk distribution) |
| `src/drift_detector.py` | Concept drift detection: PSI, KS test, distribution shift alerts |

---

## Model Architecture: RiskMLP (v2)

```
Input  (15 features: income, savings, debt, EMI ratio, credit history, age proxy, occupation, ...)
  │
  ├─ Linear(15 → 256) → BatchNorm1d → GELU → Dropout(0.3)
  │   ↓ + residual projection (15 → 256)
  ├─ Linear(256 → 128) → BatchNorm1d → GELU → Dropout(0.3)
  ├─ Linear(128 → 64)  → BatchNorm1d → GELU → Dropout(0.3)
  ├─ Linear(64  → 32)  → BatchNorm1d → GELU
  └─ Linear(32  →  5)  → LogSoftmax
         │
  Output: [Very_Low, Low, Medium, High, Very_High]

  Loss:  FocalLoss(γ=2.0, label_smoothing=0.05)
  Total params: ~83,000
```

---

## LLM Provider Stack (v3)

| Priority | Provider | Model | Params | Why |
|----------|----------|-------|--------|-----|
| 1 | Groq | llama-3.3-70b-versatile | 70B | MT-Bench 8.6, superior financial reasoning, free tier |
| 2 | OpenRouter | google/gemma-2-9b-it:free | 9B | MMLU 71.3%, strong instruction-following, free |
| 3 | HuggingFace | Qwen/Qwen2.5-3B-Instruct | 3B | IFEval 76.1%, best <5B model, no API key needed |
| 4 | Rule-based | — | — | Deterministic, always available |

All providers use temperature=0.3, max_tokens=400, with SEBI-advisor system prompt.

---

## Ensemble Fund-Scoring Architecture (v2)

```
  mutual_fund_data.csv ──► build_fund_features() ──────────────────────────────┐
                           (12 base features incl. expense_ratio_norm)          │
  nav_metrics.csv ────────► +7 historical features ──► Feature Matrix (19)     │
  (cached from parquet)     CAGR/Sharpe/Drawdown/                   ┌──────────┘
                            Momentum                                 │
                                                                     ▼
                         Quality Target Q = 0.35×CAGR_3yr + 0.25×Sharpe
                                        + 0.25×(−MaxDD) + 0.15×Momentum_6m
                                        − 0.10×TER_norm
                                                                     │
                                                     ┌───────────────┼───────────────┐
                                                     ▼               ▼               ▼
                                            RandomForest(200)   XGBoost(200)   LightGBM(200)
                                              weight: 0.35       weight: 0.40   weight: 0.25
                                                     │               │               │
                                                     └───── weighted avg ────────────┘
                                                               │
                                                               ▼
                                                   ensemble_score per fund
```

---

## Evaluation Results

| # | Metric | Target | Result | Status |
|---|--------|--------|--------|--------|
| 1 | Cluster Silhouette (temperature-softmax) | ≥ 0.80 | **0.9472** | ✅ PASS |
| 2 | F1 Score (FL model, pseudo-label mode) | > 0.80 | **0.9611** | ✅ PASS |
| 3 | Differential Privacy (ε, δ) | formal guarantee | **ε=10.0087, δ=1e-5** | ✅ PASS |
| 4 | Federated Loss Stability | Δ < 10% | **0.19%** | ✅ PASS |
| 5 | GPT Correctness | ≥ 75% | **100%** | ✅ PASS |

System tests: **13/13 pass** (v3)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML Framework | PyTorch 2.x (ARM64 native) |
| Federated Learning | Manual FedProx + FedAvg (no Flower dependency) |
| Differential Privacy | Manual Gaussian noise injection (no Opacus) |
| Clustering | scikit-learn KMeans (temperature-softmax embeddings) |
| Fund Scoring | XGBoost 3.2.0 + RandomForest + LightGBM 4.6.0 (3-model ensemble) |
| NAV History | pyarrow parquet (row-group chunked reads, 21M rows) |
| LLM | Groq Llama-3.3-70B / OpenRouter Gemma-2-9B / HuggingFace Qwen-2.5-3B |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Notebooks | Jupyter (6 notebooks) |
| Environment | conda `smart_fund_advisor` |
