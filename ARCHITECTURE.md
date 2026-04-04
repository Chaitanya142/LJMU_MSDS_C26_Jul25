# Smart Fund Advisor — System Architecture (v3.3)

## Overview

A privacy-preserving mutual fund recommendation system that:
1. Scores user **risk appetite** from banking transaction history using **15 engineered features** and a **multi-metric risk matrix** (4 sub-dimensions: Financial Capacity, Behavioral Tolerance, Time Horizon, Credit Health)
2. Trains a **neural network** (RiskMLP 15→256→128→64→32→5, GELU activations, residual connections, CrossEntropyLoss(label_smoothing=0.05)) on 65% of customers centrally — **test accuracy 94.51%**, val_acc=0.9545 at epoch 18 (early stop at epoch 26)
3. Fine-tunes with **Federated Learning** (FedProx + FedAvg, adaptive Dirichlet α 1.5→0.3, 15 rounds) across 3,750 virtual mobile devices with **Differential Privacy** (Gaussian σ=1.1, **ε=0.4862**) — **FL macro F1=0.9439**, accuracy=0.9435
4. Classifies 16,346 mutual funds into risk tiers via a **two-step data-driven mechanism** (per-category z-score → percentile bands → SEBI floor/ceiling clamp) then ranks with an **enhanced 5-model ensemble** (Random Forest, XGBoost, LightGBM, ExtraTreesRegressor, CatBoost) **combined via Ridge stacking meta-learner** achieving **R²=0.8680** (+54.8% improvement over baseline) trained on **42 features** including alpha target (excess NIFTY return - TER) and regime_beta from a **154,060-row forward-looking panel**; 14,330 funds scored (14,427 with NAV history)
5. Generates **horizon-based recommendations** (1yr/3yr/5yr/10yr+) following SEBI glide-path allocation
6. Builds **Core-Satellite diversified portfolios** (60% core + 20% stability + 20% growth) across multiple risk brackets
7. Sources **21M-row NAV history** from parquet (2006–2026-02-15) to compute CAGR, Sharpe, drawdown, momentum, Nifty500-relative consistency, and recovery-time per scheme
8. Integrates **FundPerf + Nifty500 benchmark data** to compute real benchmark-relative features (benchmark status, benchmark returns, excess returns, benchmark-relative consistency)
9. **Integrates real TER data** (SEBI ter-of-mf-schemes.csv): 62.8% real coverage (9,004/14,330 funds); unmatched funds use category/global-median imputation with `ter_missing_flag` — no synthetic TER anywhere in the pipeline
10. **Explains** recommendations using free LLMs: Groq (Llama-3.3 70B) → OpenRouter (Gemma 2 9B) → HuggingFace (Qwen 2.5 3B) → rule-based fallback

---

## v3.3 Hardening Updates (TER + Consistency + Benchmark)

1. **Real TER Integration**: TER matching runs in strict priority (scheme code → name+category → name-only). Unmatched funds use category/global-median imputation for modelling — marked with `ter_missing_flag` and `ter_source`. No synthetic TER is generated anywhere in the pipeline.
2. **TER Penalty Down-weighting**: When TER is imputed (missing), the scoring penalty applied to expense ratio is halved to avoid over-penalising funds for which no TER data is available.
3. **Forward-Looking Ensemble Target**: Training labels are built from a forward-looking panel `(scheme, as_of_date)` snapshots targeting next-3Y top-20% fund quality. 154,060 panel rows, positive-rate=22.1%.
4. **Asset-Segment Ensembles**: Separate sub-models for equity (47,613 panel rows) vs non-equity (106,447 panel rows). Benchmark columns zeroed for non-benchmarked/non-equity funds.
5. **Benchmark-Relative Consistency**: `consistency_1y` is computed as percentage of rolling 1Y windows beating Nifty500 TRI (where Nifty history is available), with positive-return fallback otherwise.
6. **Data-Quality Mode Tracking**: Funds explicitly tracked as Mode A (full NAV history), B (FundPerf-only), C (low-information). Case C hotspot count logged per training run.
7. **Recommendation Hardening**: Segment-aware shortlist balancing (Equity/Hybrid/Debt), `Not_Eligible` vs `Benchmarked_With_Data` vs `Eligible_No_Data` benchmark status handling, per-category AMC caps.

---

## v3.4 Enhancement — Advanced Ensemble with Stacking Meta-Learner & Alpha Target

**Performance Improvement: +54.8% R² (0.5606 → 0.8680 on test set)**

1. **5-Model Base Ensemble**:
   - **Random Forest**: 200 estimators, min_leaf_samples=5, min_split_samples=10
   - **XGBoost**: 400 estimators, learning_rate=0.02, max_depth=6, alpha=0.05, lambda=0.5
   - **LightGBM**: 400 estimators, learning_rate=0.02, max_depth=7, num_leaves=31, alpha=0.05, lambda=0.5
   - **ExtraTreesRegressor**: 500 estimators (NEW), min_leaf_samples=5, min_split_samples=10
   - **CatBoost**: 500 iterations, learning_rate=0.015, depth=7 (NEW, optional)

2. **Ridge Stacking Meta-Learner (NEW)**:
   - Base model predictions stacked: `[RF, XGB, LGBM, CatBoost, ExtraTrees]`
   - Ridge regression (α=1.0) learns optimal weights: RF=+2.283, XGB=-0.098, LGBM=-0.132, ET=-1.100, CAT=0.0
   - Meta-learner outperforms fixed weighted average: **R²=0.8680** vs 0.5606 baseline
   - Achieved **RMSE reduction of 45.1%** (0.1509 vs ~0.275)

3. **Target Engineering: Alpha (NEW)**:
   - **Formula**: `α = Fund_Return_1Y - NIFTY500_TRI_Return_1Y - TER_Penalty`
   - Measures direct excess return vs benchmark, discounted for costs
   - More interpretable and signal-rich than synthetic quality score
   - Fallback to quality target if NAV history unavailable

4. **New Feature - regime_beta (42nd feature)**:
   - **Formula**: `regime_beta = excess_return_1y / volatility_1yr` (clipped to [-2.0, 2.0])
   - Proxy for systematic risk exposure beyond categorical tiers
   - Strong signal: correlation with excess_return_1y = r=0.8436
   - Enables risk regime responsiveness via learned model coefficients

5. **Backward Compatibility & Graceful Degradation**:
   - CatBoost optional (failing gracefully if not installed)
   - Meta-learner auto-detected (fallback to weighted ensemble if unavailable)
   - Feature alignment auto-reindexed in inference
   - All 16 system tests pass with enhanced ensemble ✓

---

## Pipeline Flow
Why not Deep Leaning and only standard machine learning

SVM Multi Class
XGBoost
DecisionTree
Dencity Tree Classification
RF
Ensembling
```
bank_user_dataset.csv ──► Preprocessing (15 features)(PCA) ──► Risk Labelling (bell-curve)
                                    │                              │
                                    ▼                              ▼
                         Multi-Metric Risk Matrix          Central Model Training
                         (4 sub-dimensions)                (RiskMLP + CrossEntropyLoss)
                                                            Plot AUC Curve
                                                                   │
mutual_fund_data.csv ──────────────────────────────────┐           │
FundPerf/*.xlsx + Nifty500/*.csv ──────────────────────┤           │
                                                        │           ▼
                                         ┌──────────────┤   FL Simulation (FedProx + DP) (Have some slides)
                                         │              │           │
                                         │              │           ▼
                                         │       RiskMLP → Risk Tier Prediction
                                         │              │
                                         │       Benchmark Feature Join (FundPerf + Nifty500)
                                         │       → benchmark_status / benchmarked_flag
                                         │       → fund_return / benchmark_return / excess_return
                                         │              │
                                         │       Two-Step Risk Tier Assignment (16,346 funds)
                                         │       → NAV z-score bands (14,330) + keyword fallback (2,016)
                                         │       → SEBI floor/ceiling clamp (CATEGORY_RISK_BOUNDS)
                                         │              │
                                         │       Horizon Filter → Fund Pool
                                         │              │
                                         │       Enhanced 5-Model Ensemble Scoring (42 features, 154,060-row forward panel)
                                         │       → RF + XGB + LGBM + ExtraTrees + CatBoost (base models)
                                         │       → Ridge Stacking Meta-Learner (learned weights, R²=0.8680)
                                         │       → Alpha target (excess return - NIFTY TRI return - TER)
                                         │       → regime_beta feature (excess_return / volatility)
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
  14,330 funds  →  risk_tier_source = "nav_history"
   2,016 funds  →  risk_tier_source = "keyword"   (no NAV history)
       0 funds  →  unmapped  (full coverage across all SEBI categories)
```

Ref: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 — product risk-o-meter.

---

## Module Map

| File | Responsibility |
|------|---------------|
| `src/preprocessing.py` | Load, clean, aggregate, engineer 15 risk features (12 financial + Age_Risk_Proxy + Occupation_Stability_Score + 3 v2 features) |
| `src/risk_labeling.py` | Weighted rule-based scoring → 5-class bell-curve label; **v3: `compute_risk_matrix()` — 4-dimension risk decomposition** |
| `src/central_model.py` | RiskMLP (15→256→128→64→32→5, GELU, residual, CrossEntropyLoss), train/predict/save/load |
| `src/fl_simulation.py` | Manual FedProx + FedAvg loop, 3,750 virtual devices; incremental: on-device pseudo-labelling |
| `src/fl_client.py` | Per-device FL client: local FedProx training + Gaussian DP noise |
| `src/privacy_analysis.py` | Rényi DP ε accounting, σ→ε trade-off table |
| `src/cluster_recommender.py` | KMeans(k=5) on temperature-softmax(T=0.2) embeddings — evaluation only |
| `src/recommender.py` | Fund filter by risk tier + ensemble ranking + `allocate_portfolio()` + **v3: `recommend_funds_by_horizon()`, `recommend_diversified_portfolio()`**; **v4: two-step tier assignment, AMC pre-cap (≤40%), iterative allocation cap** |
| `src/benchmark_features.py` | FundPerf + Nifty500 ingestion, benchmark mapping, excess-return computation, and feature attachment to fund universe |
| `src/ensemble_recommender.py` | XGBoost + RF + LightGBM fund scorer (3-model ensemble, 46 features, 154,060-row forward-looking panel) |
| `src/nav_history.py` | 21M-row parquet reader; 10 per-scheme NAV metrics; cache layer; **v4: `compute_fund_risk_bands()` — two-step realized-risk classification with SEBI floor/ceiling clamp** |
| `src/gpt_explainer.py` | Free LLM: **v3: Groq Llama-3.3-70B / OpenRouter Gemma-2-9B / HF Qwen-2.5-3B** / rule-based; **v3.1: C3 TTL cache, C1 JSON prompt, C2 risk-class few-shot** |
| `src/evaluation.py` | All 8 evaluation metrics + dashboard plots |
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

  Loss:  CrossEntropyLoss(label_smoothing=0.05)
  Total params: ~83,000
```

---

## LLM Provider Stack (v3.1)

| Priority | Provider | Model | Params | Why |
|----------|----------|-------|--------|-----|
| 1 | Groq | llama-3.3-70b-versatile | 70B | MT-Bench 8.6, superior financial reasoning, free tier |
| 2 | OpenRouter | google/gemma-2-9b-it:free | 9B | MMLU 71.3%, strong instruction-following, free |
| 3 | HuggingFace | Qwen/Qwen2.5-3B-Instruct | 3B | IFEval 76.1%, best <5B model, no API key needed |
| 4 | Rule-based | — | — | Deterministic, always available |

All providers: temperature=0.3, max_tokens=400, SEBI-advisor system prompt.

**v3.1 LLM improvements:**
- **C1 JSON prompt**: Requests `{"summary", "rationale", "recommendation"}` JSON; fallback-parses to readable paragraphs
- **C2 Few-shot**: 1 risk-tier-specific example per prompt — guides tone, data-citation style, and length (<180 words)
- **C3 TTL cache**: 1-hour in-memory cache (MD5 key = `provider:prompt`); avoids redundant API calls in multi-horizon sessions

---

## Ensemble Fund-Scoring Architecture (v3.3)

```
  mutual_fund_data.csv ──► build_fund_features() ──────────────────────────────────────┐
  FundPerf/*.xlsx ─────► +benchmark features (excess_return, consistency_1y)            │
  Nifty500/*.csv ──────► +Nifty consistency (% rolling 1Y windows beating Nifty500 TRI) │
  SEBI TER CSV ────────► +real_ter (62.8% coverage), ter_missing_flag                   │
  nav_metrics.csv ─────► +NAV performance features (CAGR/Sharpe/Drawdown/Momentum)     │
  (cached from parquet) ──────────────────────────────► Feature Matrix (46 features)   │
                                                                           ┌────────────┘
                          Forward-looking panel: 154,060 rows              │
                          (scheme, as_of_date) snapshots                   │
                          Segment: equity=47,613 / non_equity=106,447      │
                          Positive-rate: 22.1%                             │
                                                                           ▼
                         Binary quality label: top-20% fund in next 3 years
                                                                           │
                                              ┌────────────────────────────┼───────────────────┐
                                              ▼                            ▼                   ▼
                                   RandomForest(200)              XGBoost(200)          LightGBM(200)
                                     weight: 0.35                  weight: 0.40          weight: 0.25
                                     R²=0.8378                     R²=0.3145             R²=0.3648
                                              │                            │                   │
                                              └──────────── weighted avg ──────────────────────┘
                                                                   │
                                                                   ▼
                                                       ensemble_score per fund
```

---

## Evaluation Results (v3.3)

| # | Metric | Target | Result | Status |
|---|--------|--------|--------|--------|
| 1 | Cluster Silhouette (temperature-softmax) | ≥ 0.80 | **0.9560** (DB=0.0668, Purity=0.9488) | ✅ PASS |
| 2 | F1 Score (FL incremental, pseudo-label, 5 waves×3 rounds) | > 0.80 | **0.9439** | ✅ PASS |
| 3 | Differential Privacy (ε, δ) — tight Mironov amplification | formal guarantee | **ε=0.4862, δ=1e-5** | ✅ PASS |
| 4 | Federated Loss Stability | Δ < 10% | **0.32%** | ✅ PASS |
| 5 | GPT Correctness | ≥ 75% | **100%** | ✅ PASS |
| 6 | Brier Score (Model Calibration) | < 0.10 | **0.0932** | ✅ PASS |
| 7 | Accuracy Parity Ratio (Age Fairness, Hardt et al. 2016) | ≥ 0.90 | **0.9882** | ✅ PASS |
| 8 | FL-Central Accuracy Gap | < 2.0 pp | **0.1600 pp** | ✅ PASS |

Additional metrics (not pass/fail thresholds):  
- ROC-AUC (One-vs-Rest): Macro=**0.9983**, Weighted=**0.9980**  
- Central model val accuracy: **95.45%** (val_acc=0.9545 at epoch 18/40)  
- FL global model accuracy: **94.35%** | Central on FL split: **94.51%**  
- Alternative classifier comparison (Step 3b): MLP val_F1=0.9463 vs SVM=0.9603 vs XGB=0.9218 vs RF=0.8608 vs VotingEnsemble=0.9322  
- MLP-FL vs SVM-FL: MLP wins (macro F1 0.9439 vs 0.3328) — non-linear interactions require MLP depth  
- Ensemble (46 features, 154,060 forward-panel rows): RF R²=**0.8378**, XGB R²=**0.3145**, LGBM R²=**0.3648**  
- TER coverage: 62.8% real (9,004/14,330); Benchmark-ready: 15.6% (2,232/14,330)

System tests: **16/16 pass** (v3.3)  

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
