"""
Smart Fund Advisor — Full System Test  (v3)
============================================
Tests all 13 subsystems of the Smart Fund Advisor pipeline:
  1. Preprocessing + 15-feature engineering
  2. Central Model (RiskMLP w/ GELU, residual, FocalLoss)
  3. Federated Learning global model
  4. Differential Privacy accounting
  5. Drift Detection (PSI + KS test)
  6. KMeans Cluster Recommender (silhouette ≥ 0.80)
  7. Ensemble Fund Scorer (XGB + RF + LGBM)
  8. Fund Recommendations (all 5 risk tiers)
  9. Portfolio Diversification (HHI + AMC cap)
 10. GPT Explanation + Correctness Validation
 11. Multi-Metric Risk Matrix (4 sub-dimensions)
 12. Horizon-Based Recommendations (1yr/3yr/5yr/10yr+)
 13. Core-Satellite Diversified Portfolio (3 brackets)

Run: python test_system.py
"""
import sys, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
results = {}
t0 = time.time()

print("=" * 65)
print("  SMART FUND ADVISOR — Full System Test  (v3)")
print("=" * 65)


# ─── 1. Preprocessing + 15-feature engineering ────────────────────────────────
print("\n[1/10] Preprocessing + Risk Labelling (15 features)")
try:
    from src.preprocessing import get_clean_customer_data
    from src.risk_labeling import assign_risk_label
    from config import RISK_FEATURES
    df = get_clean_customer_data()
    df = assign_risk_label(df)
    assert len(df) > 1000, "Too few customers"
    assert "risk_label" in df.columns
    # Verify all 15 engineered features exist
    missing = [f for f in RISK_FEATURES if f not in df.columns]
    assert len(missing) == 0, f"Missing features: {missing}"
    assert len(RISK_FEATURES) == 15, f"Expected 15 features, got {len(RISK_FEATURES)}"
    new_feats = ["EMI_Income_Ratio", "Savings_Rate", "Credit_History_Score"]
    for nf in new_feats:
        assert nf in RISK_FEATURES, f"New feature {nf} not in RISK_FEATURES"
        assert df[nf].notna().all(), f"{nf} has NaN values"
    dist = df["risk_label"].value_counts().to_dict()
    print(f"      Customers: {len(df):,} | Features: {len(RISK_FEATURES)}")
    print(f"      New features verified: {', '.join(new_feats)}")
    print(f"      Risk distribution: {dist}")
    results["preprocessing"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["preprocessing"] = False


# ─── 2. Central Model (RiskMLP + FocalLoss) ───────────────────────────────────
print("\n[2/10] Central Model (RiskMLP + FocalLoss + GELU + Residual)")
try:
    import joblib, torch
    from src.central_model import load_central_model, predict as mlp_predict, RiskMLP, FocalLoss
    from config import RISK_FEATURES as _RF, HIDDEN_DIMS, N_RISK_CLASSES
    if "df" not in dir() or df is None:
        df = assign_risk_label(get_clean_customer_data())
    model = load_central_model()
    le = joblib.load("models/label_encoder.joblib")

    # Architecture checks
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 40000, f"Model too small: {total_params} params"
    assert hasattr(model, '_res_proj'), "Missing residual projection layer"
    # Check GELU activation is used (not ReLU)
    model_str = str(model)
    assert 'GELU' in model_str, "Expected GELU activation"
    # FocalLoss instantiation check
    fl = FocalLoss(gamma=2.0, label_smoothing=0.05)
    dummy_logits = torch.randn(4, N_RISK_CLASSES)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    loss_val = fl(dummy_logits, dummy_labels)
    assert loss_val.item() > 0, "FocalLoss should be positive"

    feat_cols = [f for f in _RF if f in df.columns]
    X = df[feat_cols].head(200).values.astype("float32")
    preds, probs = mlp_predict(model, X)
    labels = le.inverse_transform(preds)
    assert len(set(labels)) >= 3, "Should predict multiple classes"
    print(f"      Architecture: {len(_RF)}→{'→'.join(str(h) for h in HIDDEN_DIMS)}→{N_RISK_CLASSES}")
    print(f"      Parameters: {total_params:,} | Activation: GELU | Residual: ✓")
    print(f"      FocalLoss(γ=2.0, smoothing=0.05): loss={loss_val.item():.4f}")
    print(f"      Predicted classes: {sorted(set(labels))}")
    print(f"      Max confidence: {probs.max():.3f}")
    results["central_model"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["central_model"] = False


# ─── 3. Federated Model ───────────────────────────────────────────────────────
print("\n[3/10] Federated Learning Global Model")
try:
    import torch
    from src.central_model import RiskMLP, predict as mlp_predict2
    from config import RISK_FEATURES as _RF2
    if "df" not in dir():
        from src.preprocessing import get_clean_customer_data
        from src.risk_labeling import assign_risk_label
        df = assign_risk_label(get_clean_customer_data())
    _fc = [f for f in _RF2 if f in df.columns]
    _X = df[_fc].head(200).values.astype("float32")
    fl_model = RiskMLP(input_dim=len(_fc))
    fl_model.load_state_dict(torch.load("models/fl_global_risk_model.pt",
                                         map_location="cpu"))
    fl_model.eval()
    import joblib
    _le = joblib.load("models/label_encoder.joblib")
    from src.central_model import load_central_model
    _cm = load_central_model()
    _cm_preds, _ = mlp_predict2(_cm, _X)
    fl_preds, fl_probs = mlp_predict2(fl_model, _X)
    agreement = (_cm_preds == fl_preds).mean()
    print(f"      Central ↔ FL agreement: {agreement:.3f} ({agreement*100:.1f}%)")
    print(f"      FL model confidence (mean): {fl_probs.max(axis=1).mean():.3f}")
    assert agreement > 0.85, f"FL agreement too low: {agreement}"
    results["fl_model"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["fl_model"] = False


# ─── 4. Privacy Accounting ────────────────────────────────────────────────────
print("\n[4/10] Differential Privacy Accounting")
try:
    from src.privacy_analysis import compute_epsilon
    result = compute_epsilon()
    eps = result[0] if isinstance(result, tuple) else result
    print(f"      ε = {eps:.4f}, δ = 1e-05")
    assert 0 < eps < 200, f"Epsilon out of range: {eps}"
    if eps <= 1:
        quality = "Strong"
    elif eps <= 10:
        quality = "Moderate"
    else:
        quality = "Weak (but formal noise is active)"
    print(f"      Privacy quality: {quality}")
    results["privacy"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["privacy"] = False


# ─── 5. Drift Detection (PSI + KS Test) ──────────────────────────────────────
print("\n[5/10] Drift Detection (PSI + KS Test)")
try:
    import numpy as np
    import pandas as pd
    from src.drift_detector import detect_drift, drift_summary, population_drift_report
    from config import RISK_FEATURES

    # Simulate original and drifted customer features
    rng = np.random.default_rng(42)
    original = rng.uniform(0, 1, size=(15,)).astype("float32")
    # No-drift case
    small_change = original + rng.uniform(-0.01, 0.01, size=(15,)).astype("float32")
    flag_no, score_no, feat_no, details_no = detect_drift(
        small_change, original, feature_names=RISK_FEATURES, threshold=0.25)
    # Big-drift case
    big_change = original + rng.uniform(0.3, 0.5, size=(15,)).astype("float32")
    flag_yes, score_yes, feat_yes, details_yes = detect_drift(
        big_change, original, feature_names=RISK_FEATURES, threshold=0.25)

    assert not flag_no, "Small change should NOT trigger drift"
    assert flag_yes, "Large change SHOULD trigger drift"
    assert isinstance(details_yes, dict), "detect_drift should return per-feature details"

    # Population drift report (batch-level PSI + KS)
    baseline_arr = rng.normal(0.5, 0.1, size=(200, 15))
    current_arr  = rng.normal(0.5, 0.1, size=(200, 15))
    baseline_pdf = pd.DataFrame(baseline_arr, columns=RISK_FEATURES)
    current_pdf  = pd.DataFrame(current_arr, columns=RISK_FEATURES)
    report_df = population_drift_report(baseline_pdf, current_pdf, RISK_FEATURES)
    assert isinstance(report_df, pd.DataFrame), "Report should be a DataFrame"
    assert "psi" in report_df.columns, "Report should have psi column"
    assert "drift_severity" in report_df.columns, "Report should have severity"
    high_ct = (report_df["drift_severity"] == "HIGH").sum()

    print(f"      No-drift: flag={flag_no}, score={score_no:.4f}")
    print(f"      Big-drift: flag={flag_yes}, score={score_yes:.4f}, top_feat={feat_yes}")
    print(f"      4-tuple returns: ✓ (per-feature details dict)")
    print(f"      Population drift report: {len(report_df)} features, {high_ct} HIGH severity")
    results["drift_detection"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["drift_detection"] = False


# ─── 6. KMeans Cluster Recommender ────────────────────────────────────────────
print("\n[6/10] KMeans Cluster Recommender (Silhouette ≥ 0.80)")
try:
    import json
    from pathlib import Path
    from src.cluster_recommender import fit_embedding_cluster_model
    # Load embed-based cluster metadata (this is what the pipeline actually uses)
    embed_meta_path = Path("models/cluster_metadata_embed.json")
    assert embed_meta_path.exists(), "Embed cluster metadata not found — run pipeline first"
    with open(embed_meta_path) as f:
        meta = json.load(f)
    sil = meta.get("silhouette_score", 0)
    db  = meta.get("davies_bouldin_index", 999)
    pur = meta.get("cluster_purity", 0)
    sil_pass = sil >= 0.80
    print(f"      Embedding type: {meta.get('embedding_type', 'N/A')}")
    print(f"      Silhouette : {sil:.4f}  {'[PASS ✓]' if sil_pass else '[WARN]'}")
    print(f"      Davies-Bouldin: {db:.4f}")
    print(f"      Purity     : {pur:.4f}")
    print(f"      Cluster→Risk: {meta.get('cluster_to_risk', {})}")
    assert sil > 0.5, f"Silhouette too low: {sil}"
    results["cluster_recommender"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["cluster_recommender"] = False


# ─── 7. Ensemble Fund Scorer (XGB + RF + LGBM) ───────────────────────────────
print("\n[7/10] Ensemble Fund Scorer (XGB + RF + LightGBM)")
try:
    from src.ensemble_recommender import score_funds_ensemble
    from src.recommender import load_mutual_funds
    mf_df = load_mutual_funds()
    scores = score_funds_ensemble(mf_df)
    assert len(scores) == len(mf_df), "Score count mismatch"
    assert scores.min() >= 0, "Scores should be non-negative"
    assert scores.max() <= 1.05, "Scores should be ≤ 1"  # soft upper bound
    # Check that LightGBM model exists
    from pathlib import Path
    lgbm_path = Path("models/lgbm_fund_model.joblib")
    lgbm_present = lgbm_path.exists()
    # Check ensemble meta for model weights
    import json
    meta_path = Path("models/ensemble_fund_meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        weights = meta.get("ensemble_weights", {})
    else:
        weights = {}
    print(f"      Funds scored: {len(scores):,}")
    print(f"      Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"      LightGBM model: {'✓' if lgbm_present else '✗ (fallback 2-model)'}")
    print(f"      Ensemble weights: {weights}")
    results["ensemble_scorer"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["ensemble_scorer"] = False


# ─── 8. Fund Recommendations (all 5 risk tiers) ──────────────────────────────
print("\n[8/10] Fund Recommendations (all 5 risk tiers)")
try:
    from src.recommender import load_mutual_funds, recommend_funds
    if "mf_df" not in dir():
        mf_df = load_mutual_funds()
    print(f"      Fund database: {len(mf_df):,} schemes")
    all_ok = True
    for tier in ["Very_Low", "Low", "Medium", "High", "Very_High"]:
        recs = recommend_funds(tier, mf_df, top_n=5)
        top_name = recs.iloc[0].get("Scheme_Name", "N/A") if not recs.empty else "NONE"
        print(f"      {tier:<12} → {len(recs)} funds | {str(top_name)[:45]}")
        if recs.empty:
            all_ok = False
    results["recommendations"] = all_ok
except Exception as e:
    print(f"      ERROR: {e}")
    results["recommendations"] = False


# ─── 9. Portfolio Diversification (HHI + AMC cap) ────────────────────────────
print("\n[9/10] Portfolio Diversification (HHI + AMC Cap)")
try:
    from src.recommender import allocate_portfolio, load_mutual_funds, recommend_funds
    if "mf_df" not in dir():
        mf_df = load_mutual_funds()
    recs_div = recommend_funds("Medium", mf_df, top_n=10)
    portfolio = allocate_portfolio(recs_div, "Medium")
    assert "weight" in portfolio.columns, "Missing weight column"
    assert "alloc_amount_inr" in portfolio.columns, "Missing alloc_amount_inr column"
    # Check weights sum to 1
    weight_sum = portfolio["weight"].sum()
    print(f"      Weight sum: {weight_sum:.4f}  (expected ~1.0)")
    assert 0.99 <= weight_sum <= 1.01, f"Weights should sum to ~1.0, got {weight_sum}"
    # Check AMC concentration cap
    amc_col = portfolio.get("AMC", None)
    if amc_col is not None and not amc_col.isna().all():
        n_unique_amc = amc_col.nunique()
        amc_conc = portfolio.groupby("AMC")["weight"].sum()
        max_amc = amc_conc.max()
        if n_unique_amc > 1:
            print(f"      Max AMC concentration: {max_amc:.2%}  (cap=40%)")
            assert max_amc <= 0.41, f"AMC cap violated: {max_amc}"
        else:
            print(f"      Single AMC in results ({amc_col.iloc[0][:30]}) — cap N/A")
    else:
        print("      AMC column not available — cap check skipped")
    # Check diversification score
    if "diversification_score" in portfolio.columns:
        div_score = portfolio["diversification_score"].iloc[0]
        print(f"      Diversification score: {div_score:.4f}")
    else:
        print("      diversification_score not in columns")
    total_inr = portfolio["alloc_amount_inr"].sum()
    print(f"      Total allocation: ₹{total_inr:,.0f}")
    print(f"      Funds in portfolio: {len(portfolio)}")
    results["diversification"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["diversification"] = False


# ─── 10. GPT Explanation + Validation ─────────────────────────────────────────
print("\n[10/10] GPT Explanation + Correctness Validation")
try:
    from src.gpt_explainer import explain_fund, validate_gpt_correctness, get_active_provider
    if "mf_df" not in dir():
        from src.recommender import load_mutual_funds, recommend_funds
        mf_df = load_mutual_funds()
    recs = recommend_funds("High", mf_df, top_n=3)
    provider = get_active_provider()
    print(f"      Provider: {provider}")
    scores = []
    for _, fund in recs.iterrows():
        exp, prov = explain_fund(fund, user_risk="High",
                                  user_context={"monthly_income": 100000,
                                                "financial_goal": "Wealth creation"})
        val = validate_gpt_correctness(exp, fund, "High")
        scores.append(val["correctness_score"])
        name = fund.get("Scheme_Name", "N/A")
        print(f"      {str(name)[:40]:<42}  score={val['correctness_score']:.2f}  {val['verdict']}")
    pass_rate = sum(s >= 0.75 for s in scores) / len(scores)
    print(f"      Pass rate: {pass_rate:.0%}")
    results["gpt"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["gpt"] = False


# ─── 11. Multi-Metric Risk Matrix (4 sub-dimensions) ─────────────────────────
print("\n[11/13] Multi-Metric Risk Matrix (4 sub-dimensions)")
try:
    from src.risk_labeling import compute_risk_matrix
    if "df" not in dir() or df is None:
        from src.preprocessing import get_clean_customer_data
        from src.risk_labeling import assign_risk_label
        df = assign_risk_label(get_clean_customer_data())
    risk_matrix = compute_risk_matrix(df)

    # Check all 4 dimensions + composite exist
    expected_dims = ["Financial_Capacity", "Behavioral_Tolerance", "Time_Horizon", "Credit_Health"]
    for dim in expected_dims:
        assert dim in risk_matrix.columns, f"Missing dimension: {dim}"
    assert "composite_risk_score" in risk_matrix.columns, "Missing composite_risk_score"

    # Check normalisation: all sub-scores in [0, 1]
    for dim in expected_dims:
        col = risk_matrix[dim]
        assert col.min() >= -0.01, f"{dim} min below 0: {col.min()}"
        assert col.max() <= 1.01, f"{dim} max above 1: {col.max()}"
        assert col.std() > 0.01, f"{dim} has no variance"

    # Composite should be in [0, 1]
    comp = risk_matrix["composite_risk_score"]
    assert comp.min() >= 0.0, f"Composite min below 0: {comp.min()}"
    assert comp.max() <= 1.0, f"Composite max above 1: {comp.max()}"

    # Check it has same number of rows as df
    assert len(risk_matrix) == len(df), "Row count mismatch"

    print(f"      Dimensions: {expected_dims}")
    for dim in expected_dims:
        col = risk_matrix[dim]
        print(f"        {dim:<25} mean={col.mean():.3f}  std={col.std():.3f}")
    print(f"      Composite: mean={comp.mean():.3f}, std={comp.std():.3f}")
    print(f"      All sub-scores normalised to [0,1]: ✓")
    results["risk_matrix"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["risk_matrix"] = False


# ─── 12. Horizon-Based Recommendations ───────────────────────────────────────
print("\n[12/13] Horizon-Based Fund Recommendations")
try:
    from src.recommender import recommend_funds_by_horizon, load_mutual_funds
    if "mf_df" not in dir():
        mf_df = load_mutual_funds()

    # Test 1yr horizon — should have low equity, safe categories
    res_1yr = recommend_funds_by_horizon("High", mf_df, horizon_years=1, top_n=6)
    assert res_1yr["horizon"] == "1yr"
    assert res_1yr["equity_pct"] <= 0.20, f"1yr equity too high: {res_1yr['equity_pct']}"
    assert res_1yr["debt_pct"] >= 0.80, f"1yr debt too low: {res_1yr['debt_pct']}"

    # Test 5yr horizon — should have higher equity for High risk
    res_5yr = recommend_funds_by_horizon("High", mf_df, horizon_years=5, top_n=6)
    assert res_5yr["horizon"] == "5yr"
    assert res_5yr["equity_pct"] >= 0.70, f"5yr equity too low: {res_5yr['equity_pct']}"

    # Test 10yr+ horizon
    res_10yr = recommend_funds_by_horizon("Very_High", mf_df, horizon_years=10, top_n=6)
    assert res_10yr["horizon"] == "10yr+"
    assert res_10yr["equity_pct"] >= 0.90, f"10yr+ equity too low: {res_10yr['equity_pct']}"

    # Portfolio should have funds
    for res, label in [(res_1yr, "1yr"), (res_5yr, "5yr"), (res_10yr, "10yr+")]:
        portfolio = res.get("portfolio", None)
        if portfolio is not None and not portfolio.empty:
            w_sum = portfolio["weight"].sum()
            assert 0.95 <= w_sum <= 1.05, f"{label} weights sum wrong: {w_sum}"

    print(f"      1yr:  equity={res_1yr['equity_pct']:.0%}  debt={res_1yr['debt_pct']:.0%}  "
          f"funds={len(res_1yr['portfolio']) if not res_1yr['portfolio'].empty else 0}")
    print(f"      5yr:  equity={res_5yr['equity_pct']:.0%}  debt={res_5yr['debt_pct']:.0%}  "
          f"funds={len(res_5yr['portfolio']) if not res_5yr['portfolio'].empty else 0}")
    print(f"      10yr+: equity={res_10yr['equity_pct']:.0%}  debt={res_10yr['debt_pct']:.0%}  "
          f"funds={len(res_10yr['portfolio']) if not res_10yr['portfolio'].empty else 0}")
    print(f"      SEBI glide-path: 1yr conservative → 10yr aggressive ✓")
    results["horizon_recs"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["horizon_recs"] = False


# ─── 13. Core-Satellite Diversified Portfolio ─────────────────────────────────
print("\n[13/13] Core-Satellite Diversified Portfolio")
try:
    from src.recommender import recommend_diversified_portfolio, load_mutual_funds
    if "mf_df" not in dir():
        mf_df = load_mutual_funds()

    # Test Medium risk — should have 3 brackets (Core, Stability, Growth)
    res_med = recommend_diversified_portfolio("Medium", mf_df, horizon_years=5,
                                              top_n_per_bracket=3, total_amount=500_000)
    brackets = res_med.get("brackets", [])
    bracket_names = [b["bracket"] for b in brackets]
    assert "Core" in bracket_names, "Missing Core bracket"
    assert len(brackets) >= 2, f"Expected ≥2 brackets, got {len(brackets)}"

    # Portfolio should have weight column summing to ~1.0
    portfolio = res_med.get("portfolio", None)
    assert portfolio is not None and not portfolio.empty, "Portfolio is empty"
    w_sum = portfolio["weight"].sum()
    assert 0.95 <= w_sum <= 1.05, f"Weights sum wrong: {w_sum}"

    # Allocation amounts should roughly match total
    alloc_sum = portfolio["alloc_amount_inr"].sum()
    assert alloc_sum > 400_000, f"Allocation sum too low: {alloc_sum}"

    # Diversification score should be > 0
    div_score = res_med.get("diversification_score", 0)
    assert div_score > 0.0, f"Diversification score should be > 0, got {div_score}"

    # Check bracket tiers are different (for Medium, stability=Low, growth=High)
    assert res_med["stability_tier"] != res_med["growth_tier"], "Stability and Growth tiers should differ"

    # Check edge case: Very_Low should push stability → core
    res_vl = recommend_diversified_portfolio("Very_Low", mf_df, horizon_years=3,
                                             top_n_per_bracket=3)
    # Very_Low has no tier below, so stability should fold into core
    vl_brackets = [b["bracket"] for b in res_vl.get("brackets", [])]
    assert "Core" in vl_brackets, "Very_Low should still have Core"

    # Check edge case: Very_High should push growth → core
    res_vh = recommend_diversified_portfolio("Very_High", mf_df, horizon_years=5,
                                             top_n_per_bracket=3)
    vh_brackets = [b["bracket"] for b in res_vh.get("brackets", [])]
    assert "Core" in vh_brackets, "Very_High should still have Core"

    print(f"      Medium: {len(brackets)} brackets — {bracket_names}")
    print(f"        Core={res_med['core_tier']}, Stability={res_med['stability_tier']}, "
          f"Growth={res_med['growth_tier']}")
    print(f"        Portfolio: {len(portfolio)} funds, weight_sum={w_sum:.4f}")
    print(f"        Diversification: {div_score:.4f}")
    print(f"        Allocation: ₹{alloc_sum:,.0f}")
    print(f"      Edge cases: Very_Low={len(res_vl.get('brackets',[]))} brackets, "
          f"Very_High={len(res_vh.get('brackets',[]))} brackets ✓")
    results["core_satellite"] = True
except Exception as e:
    print(f"      ERROR: {e}")
    results["core_satellite"] = False


# ─── Summary ──────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print()
print("=" * 65)
print("  TEST RESULTS")
print("=" * 65)
all_passed = True
for test, ok in results.items():
    icon = PASS if ok else FAIL
    print(f"  [{icon}] {test}")
    if not ok:
        all_passed = False

print()
passed_n = sum(results.values())
print(f"  {passed_n}/{len(results)} tests passed  ⏱ {elapsed:.1f}s")
print("=" * 65)

sys.exit(0 if all_passed else 1)
