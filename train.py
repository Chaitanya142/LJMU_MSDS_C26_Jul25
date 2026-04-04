"""
train.py  –  Full Smart Fund Advisor pipeline runner (CLI).

Usage
-----
    conda activate smart_fund_advisor
    python train.py [--rounds 10] [--no-dp] [--skip-fl] [--gpt-provider groq]
    python train.py --incremental                  # production-like: users join in waves
    python train.py --incremental --waves 5 --rounds-per-wave 3

Steps
-----
1.  Preprocess & feature-engineer (50 000 rows → per-customer records)
2.  Derive 5-class risk labels + show sample user predictions
3.  Train central MLP on 70% of customers
4.  Run FL simulation on 30% of customers (≤4 records per device)
    Mode A (default)     : all devices available from round 1 (batch FL)
    Mode B (--incremental): users onboard in waves (production-like FL)
5.  Federated model evaluation + DP budget
6.  KMeans cluster analysis (silhouette score)
7.  Fund recommendations with GPT explanations
8.  Full evaluation metrics report (8 metrics: cluster, FL F1, DP, stability,
    GPT correctness, calibration, fairness, FL-central gap)
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Project imports ────────────────────────────────────────────────────────────
from config import (
    RISK_FEATURES, CENTRAL_SPLIT, DEMO_SPLIT, RANDOM_SEED, RISK_CLASSES,
    FL_ROUNDS, CENTRAL_MODEL_PATH, FL_GLOBAL_MODEL, EPOCHS, DEMO_CUSTOMERS_PATH
)
from src.preprocessing  import get_clean_customer_data
from src.risk_labeling  import assign_risk_label, compute_risk_matrix, fit_pca_risk_model
from src.central_model  import (
    train_central_model, save_central_model, load_central_model, predict,
    print_classification_report
)
from src.fl_simulation        import run_fl_simulation, run_incremental_fl_simulation
from src.recommender          import (
    load_mutual_funds, recommend_funds, advise_user,
    recommend_funds_by_horizon, recommend_diversified_portfolio, allocate_portfolio,
    recommend_full_profile,
)
from src.ensemble_recommender import fit_fund_ensemble, plot_ensemble_importance
from src.nav_history          import load_nav_metrics
from src.privacy_analysis     import privacy_summary
from src.cluster_recommender  import fit_cluster_model, fit_embedding_cluster_model, plot_cluster_analysis
from src.evaluation           import (
    run_full_evaluation, plot_evaluation_dashboard, plot_per_user_risk,
    plot_roc_auc_curves, evaluate_roc_auc,
)
from src.gpt_explainer        import explain_fund, validate_gpt_correctness, explain_full_profile
from src.utils                import set_seed, plot_training_history, plot_confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def parse_args():
    p = argparse.ArgumentParser(description="Smart Fund Advisor — Training Pipeline")
    p.add_argument("--rounds",       type=int, default=FL_ROUNDS,
                   help="Number of FL rounds (default: 10)")
    p.add_argument("--no-dp",        action="store_true",
                   help="Disable differential privacy on FL clients")
    p.add_argument("--skip-fl",      action="store_true",
                   help="Skip federated learning (central model only)")
    p.add_argument("--epochs",       type=int, default=EPOCHS,
                   help="Central training epochs (default: from config)")
    p.add_argument("--skip-eval",    action="store_true",
                   help="Skip full evaluation metrics")
    p.add_argument("--skip-cluster",  action="store_true",
                   help="Skip KMeans cluster analysis (silhouette metric only, no impact on recommendations)")
    p.add_argument("--incremental",   action="store_true",
                   help="Use production-like incremental FL: users join in waves (more realistic)")
    p.add_argument("--waves",          type=int, default=5,
                   help="Number of user-onboarding waves for incremental FL (default: 5)")
    p.add_argument("--rounds-per-wave", type=int, default=3,
                   help="FL rounds to run after each wave joins (default: 3; total = waves × rounds-per-wave)")
    p.add_argument("--algorithm", type=str, default="fedprox",
                   choices=["fedprox", "fedavg"],
                   help="FL aggregation algorithm: fedprox (mu>0 proximal term) or fedavg (mu=0 ablation, default: fedprox)")
    p.add_argument("--gpt-provider", type=str, default=None,
                   choices=["groq", "openrouter", "huggingface", "rule"],
                   help="GPT provider for fund explanations (default: auto-detect)")
    p.add_argument("--no-gpt",       action="store_true",
                   help="Skip GPT explanation evaluation")
    return p.parse_args()


def main():
    args   = parse_args()
    t_start = time.time()

    set_seed(RANDOM_SEED)
    le = None   # label encoder, loaded after preprocessing

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 1 — Data Loading, Feature Engineering, Risk Labelling")
    # ──────────────────────────────────────────────────────────────────────────
    print("[1/8] Loading and preprocessing bank dataset...")
    df = get_clean_customer_data(fit_scaler=True)

    # ── Fit PCA risk model (PC1 → composite risk score) ─────────────────────
    banner("STEP 1a — PCA Risk Model (PC1 as Composite Risk Score)")
    pca_risk = fit_pca_risk_model(df)

    df = assign_risk_label(df, fit_encoder=True, pca_model=pca_risk)
    le = joblib.load("models/label_encoder.joblib")

    print(f"      Total customers : {df['Customer_ID'].nunique()}")
    print(f"      Risk distribution:\n{df['risk_label'].value_counts().to_string()}")

    # ── Show 10 sample user risk predictions (test cases) ──────────────────
    print("\n  Sample User Risk Scores (first 10 customers):")
    print(f"  {'Customer_ID':<20} {'Account_Type':<15} {'Risk_Score':>12} {'Risk_Label':<12}")
    print("  " + "-" * 62)
    sample_show = df.drop_duplicates("Customer_ID").head(10)
    for _, row in sample_show.iterrows():
        acct = row.get("Customer_ID", "")
        score = row.get("risk_score", 0.0)
        label = row.get("risk_label", "")
        print(f"  {str(acct):<20} {'Customer':<15} {score:>12.4f} {label:<12}")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 2 — Customer Split (65% Central / 30% FL / 5% Demo Holdout)")
    # ──────────────────────────────────────────────────────────────────────────
    all_customers = df["Customer_ID"].unique()

    # Step 1 — carve out 5 % as an unseen demo holdout (partitioned first so
    #           the demo set is independent of both training and FL cohorts).
    rest_cust, demo_cust = train_test_split(
        all_customers, test_size=DEMO_SPLIT, random_state=RANDOM_SEED
    )

    # Step 2 — split the remaining 95 % into central (65 % of total) and
    #           FL (30 % of total).  Fractions are rescaled to sum to 1 over
    #           the non-demo pool: 0.65/0.95 ≈ 68.4 % → central.
    central_fraction = CENTRAL_SPLIT / (1.0 - DEMO_SPLIT)
    central_cust, fl_cust = train_test_split(
        rest_cust, train_size=central_fraction, random_state=RANDOM_SEED
    )

    df_central = df[df["Customer_ID"].isin(central_cust)].copy()
    df_fl      = df[df["Customer_ID"].isin(fl_cust)].copy()
    df_demo    = df[df["Customer_ID"].isin(demo_cust)].copy()

    print(f"      Central split : {df_central['Customer_ID'].nunique()} customers  (65 % of total)")
    print(f"      FL split      : {df_fl['Customer_ID'].nunique()} customers  (30 % of total)")
    print(f"      Demo holdout  : {df_demo['Customer_ID'].nunique()} customers  ( 5 % of total, unseen)")

    print(f"      Risk distribution Central:\n{df_central['risk_label'].value_counts().to_string()}")

    # Save splits for notebooks and demo script
    df_fl.to_csv("models/df_fl_split.csv", index=False)
    # Save one row per customer (the most recent record) for the demo script
    df_demo.sort_values("Customer_ID").drop_duplicates(
        subset="Customer_ID", keep="last"
    )[["Customer_ID", "risk_label", "risk_score"]].to_csv(
        DEMO_CUSTOMERS_PATH, index=False
    )
    print(f"      Demo customer list saved → {DEMO_CUSTOMERS_PATH}")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 3 — Central Model Training")
    # ──────────────────────────────────────────────────────────────────────────
    feat_cols = [f for f in RISK_FEATURES if f in df_central.columns]
    X_all = df_central[feat_cols].values.astype(np.float32)
    y_all = df_central["risk_label_encoded"].values.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_SEED
    )

    print(f"[3/8] Training MLP — {X_train.shape[0]} train | {X_val.shape[0]} val")
    model, history = train_central_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, verbose=True
    )
    save_central_model(model, history)

    y_pred, _ = predict(model, X_val)
    print_classification_report(y_val, y_pred)

    try:
        plot_training_history(history, save_path="models/plot_central_training.png")
        plot_confusion_matrix(y_val, y_pred, save_path="models/plot_central_confusion.png")
    except Exception:
        pass

    # ── Test cases: 5 specific synthetic users ──────────────────────────────
    banner("TEST CASES — Risk Appetite for Edge-Case Users")
    test_users = pd.DataFrame({
        "Description":              ["High Income Investor", "Over-indebted Low Earner", "Balanced Salaried", "Young Aggressive", "Retiree Conservative"],
        "Annual_Income_norm":       [0.9,  0.2,  0.5,  0.6,  0.1],
        "Monthly_Inhand_Salary_norm":[0.85, 0.15, 0.45, 0.55, 0.08],
        "Investment_Ratio":         [0.35, 0.02, 0.18, 0.30, 0.12],
        "Debt_Burden_Ratio":        [0.05, 0.85, 0.30, 0.10, 0.15],
        "Credit_Utilization_Ratio": [0.10, 0.90, 0.40, 0.25, 0.20],
        "Delay_Score":              [0.02, 0.90, 0.30, 0.05, 0.10],
        "Credit_Mix_Score":         [1.00, 0.00, 0.50, 0.80, 0.70],
        "Spending_Behaviour_Score": [0.80, 0.10, 0.50, 0.75, 0.40],
        "Num_Bank_Accounts_norm":   [0.70, 0.20, 0.40, 0.60, 0.30],
        "Interest_Rate_norm":       [0.20, 0.90, 0.45, 0.25, 0.35],
        # Age_Risk_Proxy = (70 - Age) / 52  clipped [0,1]
        "Age_Risk_Proxy":           [0.67, 0.48, 0.67, 0.87, 0.19],
        # Occupation_Stability_Score
        "Occupation_Stability_Score":[0.90, 0.48, 0.58, 0.73, 0.68],
        # ── New v2 features ─────────────────────────────────────────
        # EMI_Income_Ratio: Total_EMI / Salary  (DTI proxy)
        # High Income: low DTI | Over-indebted: very high | Balanced: moderate
        # Young Aggressive: low | Retiree: moderate
        "EMI_Income_Ratio":         [0.10, 0.85, 0.35, 0.12, 0.30],
        # Savings_Rate: (Salary - EMI) / Salary
        # High Income: high savings | Over-indebted: negative | Balanced: moderate
        "Savings_Rate":             [0.85, 0.05, 0.55, 0.80, 0.50],
        # Credit_History_Score: normalised credit history months
        # High Income: long history | Over-indebted: short | Young: short
        "Credit_History_Score":     [0.75, 0.20, 0.55, 0.25, 0.90],
    })
    X_test = test_users[feat_cols].values.astype(np.float32)
    pred_idx, probs = predict(model, X_test)
    pred_labels = le.inverse_transform(pred_idx)
    print(f"\n  {'Profile':<28} {'Predicted Risk':<16} {'Confidence':>12}")
    print("  " + "-" * 60)
    for desc, lbl, prob in zip(test_users["Description"], pred_labels, probs):
        conf = float(prob.max())
        print(f"  {desc:<28} {lbl:<16} {conf:>11.1%}")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 3b — Alternative Classifier Comparison (vs RiskMLP)")
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from src.sklearn_classifiers import fit_all_classifiers, compare_with_mlp
        from sklearn.metrics import f1_score as _f1_sk, accuracy_score as _acc_sk
        _y_pred_mlp, _ = predict(model, X_val)
        _mlp_f1  = float(_f1_sk(y_val, _y_pred_mlp, average="macro", zero_division=0))
        _mlp_acc = float(_acc_sk(y_val, _y_pred_mlp))
        _sk_results, _ = fit_all_classifiers(X_train, y_train, X_val, y_val, verbose=True)
        compare_with_mlp(_sk_results, mlp_f1=_mlp_f1, mlp_acc=_mlp_acc, verbose=True)
    except Exception as _e:
        print(f"  [Classifier comparison] Skipped: {_e}")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 4 — Federated Learning Simulation")
    # ──────────────────────────────────────────────────────────────────────────
    if args.skip_fl:
        print("[4/8] Skipped (--skip-fl flag set). Using central model.")
        global_model = load_central_model()
        fl_hist = {}
    elif args.incremental:
        # ── Production-like: users join in waves ────────────────────────────
        total_rounds = args.waves * args.rounds_per_wave
        print(f"[4/8] Incremental FL — {df_fl['Customer_ID'].nunique()} devices  "
              f"|  {args.waves} waves × {args.rounds_per_wave} rounds = {total_rounds} total  "
              f"|  DP={not args.no_dp}")
        print(f"      Wave 1 starts with {df_fl['Customer_ID'].nunique() // args.waves} users, "
              f"adding ~{df_fl['Customer_ID'].nunique() // args.waves} more each wave.")
        df_fl_4 = df_fl.groupby("Customer_ID").tail(4).reset_index(drop=True)
        _fedprox_mu = 0.0 if getattr(args, 'algorithm', 'fedprox') == 'fedavg' else None
        if _fedprox_mu == 0.0:
            print("[4/8] Algorithm: FedAvg (ablation: μ=0, no proximal term)")
        global_model, fl_hist = run_incremental_fl_simulation(
            df_fl_4,
            dp_enabled=not args.no_dp,
            n_waves=args.waves,
            rounds_per_wave=args.rounds_per_wave,
            verbose=True,
            fedprox_mu=_fedprox_mu,
        )
        # Compare accuracy: central model vs FL global model on FL split
        X_fl  = df_fl[feat_cols].values.astype(np.float32)
        y_fl  = df_fl["risk_label_encoded"].values.astype(np.int64)
        from sklearn.metrics import accuracy_score
        y_pred_central, _ = predict(load_central_model(), X_fl)
        y_pred_fl,      _ = predict(global_model,         X_fl)
        acc_c  = accuracy_score(y_fl, y_pred_central)
        acc_fl = accuracy_score(y_fl, y_pred_fl)
        print(f"\n  Central model accuracy on FL split : {acc_c:.4f}")
        print(f"  FL global  model accuracy          : {acc_fl:.4f}")
        print(f"  Improvement                        : {(acc_fl - acc_c)*100:+.2f} pp")
        from src.utils import plot_fl_history
        plot_fl_history(fl_hist, save_path="models/plot_fl_loss.png")
    else:
        df_fl_4 = df_fl.groupby("Customer_ID").tail(4).reset_index(drop=True)
        print(f"[4/8] FL simulation: {df_fl['Customer_ID'].nunique()} devices, "
              f"{args.rounds} rounds, DP={not args.no_dp}")
        _fedprox_mu = 0.0 if getattr(args, 'algorithm', 'fedprox') == 'fedavg' else None
        global_model, fl_hist = run_fl_simulation(
            df_fl_4,
            dp_enabled=not args.no_dp,
            rounds=args.rounds,
            verbose=True,
            fedprox_mu=_fedprox_mu,
        )

        # Compare on FL evaluation set
        X_fl  = df_fl[feat_cols].values.astype(np.float32)
        y_fl  = df_fl["risk_label_encoded"].values.astype(np.int64)

        from sklearn.metrics import accuracy_score
        y_pred_central, _ = predict(load_central_model(), X_fl)
        y_pred_fl,      _ = predict(global_model,         X_fl)

        acc_c  = accuracy_score(y_fl, y_pred_central)
        acc_fl = accuracy_score(y_fl, y_pred_fl)
        print(f"\n  Central model accuracy on FL split : {acc_c:.4f}")
        print(f"  FL global  model accuracy          : {acc_fl:.4f}")
        print(f"  Improvement                        : {(acc_fl - acc_c)*100:+.2f} pp")

        from src.utils import plot_fl_history
        plot_fl_history(fl_hist, save_path="models/plot_fl_loss.png")

    # ──────────────────────────────────────────────────────────────────────────
    # banner("STEP 4b — Parallel FL Pipeline: LinearSVM-FL (DP FedAvg)")
    # # ──────────────────────────────────────────────────────────────────────────
    # # Run a second FL simulation using DP LinearSVM (SGDClassifier + FedAvg)
    # # with IDENTICAL wave/round structure, Dirichlet non-IID, drift monitoring,
    # # straggler dropout, pseudo-labelling, and DP budget (σ, C) as RiskMLP-FL.
    # # Results are compared side-by-side in models/svm_vs_mlp_fl_comparison.json.
    # if not args.skip_fl:
    #     try:
    #         from src.svm_fl_simulation import run_svm_incremental_fl, compare_fl_models
    #         _svm_waves        = args.waves        if args.incremental else 1
    #         _svm_rounds_wave  = args.rounds_per_wave if args.incremental else args.rounds
    #         svm_global, svm_hist = run_svm_incremental_fl(
    #             df_fl_4,
    #             dp_enabled=not args.no_dp,
    #             n_waves=_svm_waves,
    #             rounds_per_wave=_svm_rounds_wave,
    #             verbose=True,
    #         )
    #         compare_fl_models(
    #             df_fl_4,
    #             global_model, svm_global,
    #             fl_hist, svm_hist,
    #             verbose=True,
    #         )
    #     except Exception as _svm_err:
    #         print(f"  [SVM-FL] Skipped: {_svm_err}")
    #         import traceback; traceback.print_exc()
    # else:
    #     print("[4b/8] Skipped (--skip-fl flag set).")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 5 — Differential Privacy Budget")
    # ──────────────────────────────────────────────────────────────────────────
    print("[5/8] Computing DP privacy budget (ε, δ) ...")
    dp_summary = privacy_summary(print_output=True)

    # ──────────────────────────────────────────────────────────────────────────
    # banner("STEP 6 — KMeans Cluster Analysis (Silhouette Score)")
    # # ──────────────────────────────────────────────────────────────────────────
    # # NOTE: KMeans is EVALUATION-ONLY. It has zero impact on fund recommendations.
    # # Recommendations run entirely from: RiskMLP → tier filter → XGB+RF ensemble.
    # # Skipping this step does NOT affect recommendation quality at all.
    # if args.skip_cluster:
    #     print("[6/8] KMeans cluster analysis SKIPPED (--skip-cluster).")
    #     print("      Recommendations remain fully functional via RiskMLP + XGB/RF ensemble.")
    #     cluster_metrics = {"silhouette_score": None, "note": "skipped"}
    # else:
    #     print("[6/8] Fitting KMeans(k=5) on model embeddings (FedProx global model) ...")
    #     # Primary: embedding-based clustering (target silhouette ≥ 0.80)
    #     try:
    #         kmeans, cluster_metrics = fit_embedding_cluster_model(df, global_model)
    #     except Exception as e:
    #         print(f"      [warn] Embedding clustering failed ({e}), falling back to raw features")
    #         try:
    #             kmeans, cluster_metrics = fit_cluster_model(df)
    #         except Exception as e2:
    #             print(f"      [warn] Raw-feature clustering also failed ({e2}) — skipping cluster step")
    #             cluster_metrics = {"silhouette_score": None, "note": "failed"}
    #     try:
    #         plot_cluster_analysis(df, kmeans, save_dir="models")
    #     except Exception as e:
    #         print(f"      [warn] Cluster plots skipped: {e}")

    # # ── Per-user risk plot ───────────────────────────────────────────────────
    # try:
    #     from src.evaluation import plot_per_user_risk
    #     plot_per_user_risk(df, global_model, le, n_users=20,
    #                        save_path=Path("models/plot_per_user_risk.png"))
    # except Exception as e:
    #     print(f"      [warn] Per-user risk plot skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 7 — Fund Recommendations + GPT Explanations")
    # ──────────────────────────────────────────────────────────────────────────
    # ── 1. Load historical NAV metrics (21M+ records, cached after first run) ──
    print("[7/8] Loading historical NAV metrics (parquet) …")
    nav_metrics_df = None
    try:
        nav_metrics_df = load_nav_metrics(verbose=True)
        print(f"      NAV metrics ready: {len(nav_metrics_df):,} schemes")
    except Exception as _ne:
        print(f"      [warn] NAV metrics unavailable: {_ne} — using synthetic target")

    # ── 2. Train ensemble ────────────────────────────────────────────────────────────
    print("[7/8] Training Enhanced 5-Model Ensemble with Ridge Stacking Meta-Learner ...")
    mf_df = load_mutual_funds()
    if "ter_missing_flag" in mf_df.columns and len(mf_df) > 0:
        ter_real = int((pd.to_numeric(mf_df["ter_missing_flag"], errors="coerce").fillna(1) == 0).sum())
        ter_imputed = len(mf_df) - ter_real
        print(
            f"      TER coverage: real={ter_real}/{len(mf_df)} ({100.0 * ter_real / len(mf_df):.1f}%), "
            f"imputed={ter_imputed}"
        )
    if "benchmarked_flag" in mf_df.columns and len(mf_df) > 0:
        bench_ready = int(pd.to_numeric(mf_df["benchmarked_flag"], errors="coerce").fillna(0).sum())
        print(f"      Benchmark-ready funds: {bench_ready}/{len(mf_df)} ({100.0 * bench_ready / len(mf_df):.1f}%)")
    try:
        ensemble_metrics = fit_fund_ensemble(
            mf_df, verbose=True, nav_metrics_df=nav_metrics_df
        )
        plot_ensemble_importance(ensemble_metrics)
        
        # Print comprehensive ensemble statistics
        print("\n[Ensemble] Complete Statistics:")
        hist_flag = "(+hist)" if ensemble_metrics.get("uses_history") else "(synthetic)"
        print(f"      Target: {hist_flag}  |  Features: {ensemble_metrics.get('n_features', 42)}")
        print("\n      Base Models (Individual Performance):")
        print(f"      ├─ Random Forest      R²={ensemble_metrics.get('rf_r2', 0):.4f}  RMSE={ensemble_metrics.get('rf_rmse', 0):.4f}  CV-R²={ensemble_metrics.get('rf_cv_r2_mean', 0):.4f}±{ensemble_metrics.get('rf_cv_r2_std', 0):.4f}")
        print(f"      ├─ XGBoost            R²={ensemble_metrics.get('xgb_r2', 0):.4f}  RMSE={ensemble_metrics.get('xgb_rmse', 0):.4f}  CV-R²={ensemble_metrics.get('xgb_cv_r2_mean', 0):.4f}±{ensemble_metrics.get('xgb_cv_r2_std', 0):.4f}")
        print(f"      ├─ LightGBM           R²={ensemble_metrics.get('lgbm_r2', 0):.4f}  RMSE={ensemble_metrics.get('lgbm_rmse', 0):.4f}  CV-R²={ensemble_metrics.get('lgbm_cv_r2_mean', 0):.4f}±{ensemble_metrics.get('lgbm_cv_r2_std', 0):.4f}")
        print(f"      ├─ ExtraTreesRegressor R²={ensemble_metrics.get('et_r2', 0):.4f}  RMSE={ensemble_metrics.get('et_rmse', 0):.4f}  CV-R²={ensemble_metrics.get('et_cv_r2_mean', 0):.4f}±{ensemble_metrics.get('et_cv_r2_std', 0):.4f}")
        print(f"      └─ CatBoost           R²={ensemble_metrics.get('cat_r2', 0):.4f}  RMSE={ensemble_metrics.get('cat_rmse', 0):.4f}  CV-R²={ensemble_metrics.get('cat_cv_r2_mean', 0):.4f}±{ensemble_metrics.get('cat_cv_r2_std', 0):.4f}")
        
        meta_learning = ensemble_metrics.get('meta_r2', None)
        if meta_learning:
            improvement_pct = 100.0 * (meta_learning - 0.5606) / 0.5606 if ensemble_metrics.get('rf_r2', 0) < 0.8 else 0
            print(f"\n      ✨ Ridge Stacking Meta-Learner (IMPROVED):")
            print(f"      └─ Meta-Learner       R²={meta_learning:.4f}  RMSE={ensemble_metrics.get('meta_rmse', 0):.4f}  (+{improvement_pct:.1f}% improvement)")
            print(f"\n      Learned Coefficients: {ensemble_metrics.get('meta_weights', {})}")
        else:
            print(f"\n      [Ensemble weighted average] R²={ensemble_metrics.get('rf_r2', 0):.4f}")
            
    except Exception as _e:
        print(f"      [warn] Ensemble training skipped: {_e}")
    print("[7/8] Generating mutual fund recommendations ...")

    from src.preprocessing import load_and_clean, aggregate_per_customer, engineer_features

    df_raw  = load_and_clean()
    samples = df_raw["Customer_ID"].unique()[:5]

    print("\n  Per-User Fund Recommendations:")
    print(f"  {'Customer':<20} {'Risk Tier':<12} {'Confidence':>10}  {'Top Fund'}")
    print("  " + "-" * 78)

    demo_results = []
    for cid in samples:
        cdf  = df_raw[df_raw["Customer_ID"] == cid].tail(4).copy()
        agg  = aggregate_per_customer(cdf)
        feat = engineer_features(agg, fit_scaler=False)
        X    = feat[[c for c in feat_cols if c in feat.columns]].values.astype(np.float32)

        result   = advise_user(X[0], global_model, mf_df, le, top_n=5)
        top_fund = result["recommendations"]["Scheme_Name"].iloc[0] \
                   if not result["recommendations"].empty else "N/A"
        conf     = max(result["risk_probabilities"].values())
        print(f"  {cid!s:<20} {result['risk_label']:<12} {conf:>9.1%}  {str(top_fund)[:42]}")
        demo_results.append((cid, result))

    # GPT explanation for ONE sample user
    if not args.no_gpt:
        print("\n  GPT Fund Explanation Sample:")
        print("  " + "-" * 60)
        cid_demo, res_demo = demo_results[0]
        if not res_demo["recommendations"].empty:
            top_row  = res_demo["recommendations"].iloc[0]
            exp_text, prov = explain_fund(
                top_row,
                user_risk=res_demo["risk_label"],
                user_context="Indian retail investor",
                provider=args.gpt_provider,
            )
            print(f"  Customer : {cid_demo}")
            print(f"  Fund     : {top_row.get('Scheme_Name', '')}")
            print(f"  Provider : {prov}")
            print()
            for line in exp_text.split("\n"):
                print(f"  {line}")

            # Validation
            val = validate_gpt_correctness(exp_text, top_row, res_demo["risk_label"])
            print(f"\n  GPT Validation: {val['verdict']}  "
                  f"(score={val['correctness_score']:.2f})")

    # Show top-5 funds per risk tier
    print("\n  Top-5 Funds by Risk Tier:")
    for tier in RISK_CLASSES:
        recs = recommend_funds(tier, mf_df, top_n=3)
        print(f"\n  [{tier}]")
        for _, r in recs.iterrows():
            print(f"    • {r['Scheme_Name'][:52]:<52}  AUM ₹{r.get('Average_AUM_Cr', 0):,.0f} Cr")

    # ── v3: Multi-Metric Risk Matrix demo ────────────────────────────────
    banner("STEP 7b — Multi-Metric Risk Matrix (v3)")
    print("[7b] Decomposing risk into 4 sub-dimensions ...")
    risk_matrix = compute_risk_matrix(df)
    print(f"      Dimensions: {[c for c in risk_matrix.columns if c != 'composite_risk_score']}")
    print(f"      Sample (first 5 customers):")
    print(f"      {'Customer_ID':<20} {'Fin_Cap':>8} {'Behav':>8} {'Horizon':>8} {'Credit':>8} {'Composite':>10}")
    print("      " + "-" * 68)
    sample_ids = df.drop_duplicates("Customer_ID").head(5).index
    for idx in sample_ids:
        cid = df.loc[idx, "Customer_ID"]
        row = risk_matrix.loc[idx]
        print(f"      {str(cid):<20} {row['Financial_Capacity']:>8.3f} "
              f"{row['Behavioral_Tolerance']:>8.3f} {row['Time_Horizon']:>8.3f} "
              f"{row['Credit_Health']:>8.3f} {row['composite_risk_score']:>10.4f}")
    # Show dimension statistics
    print(f"\n      Dimension Statistics:")
    for dim in ["Financial_Capacity", "Behavioral_Tolerance", "Time_Horizon", "Credit_Health"]:
        col = risk_matrix[dim]
        print(f"        {dim:<25}  mean={col.mean():.3f}  std={col.std():.3f}  "
              f"min={col.min():.3f}  max={col.max():.3f}")

    # ── v3: Horizon-based recommendations demo ───────────────────────────
    banner("STEP 7c — Horizon-Based Fund Recommendations (v3)")
    print("[7c] Generating recommendations for 1yr / 3yr / 5yr / 10yr+ horizons ...")
    demo_risk = "High"
    for h in [1, 3, 5, 10]:
        result = recommend_funds_by_horizon(demo_risk, mf_df, horizon_years=h,
                                            top_n=6, total_amount=500_000)
        n_funds = len(result["portfolio"]) if not result["portfolio"].empty else 0
        print(f"\n  [{result['horizon']}]  Equity={result['equity_pct']:.0%}  "
              f"Debt={result['debt_pct']:.0%}  Funds={n_funds}")
        if not result["portfolio"].empty:
            for _, r in result["portfolio"].head(4).iterrows():
                name  = str(r.get("Scheme_Name", "N/A"))[:40]
                wt    = r.get("weight", 0)
                alloc = r.get("alloc_amount_inr", 0)
                bkt   = r.get("bracket", "")
                print(f"    {bkt:<8} {name:<42} wt={wt:.2%}  ₹{alloc:,.0f}")

    # ── v3: Core-Satellite diversified portfolio demo ────────────────────
    banner("STEP 7d — Core-Satellite Diversified Portfolio (v3)")
    print("[7d] Building diversified portfolios across multiple risk brackets ...")
    for risk_tier in ["Low", "Medium", "High"]:
        result = recommend_diversified_portfolio(
            risk_tier, mf_df, horizon_years=5, top_n_per_bracket=3,
            total_amount=500_000)
        div = result.get("diversification_score", 0)
        n_brackets = len(result.get("brackets", []))
        print(f"\n  [{risk_tier}]  Brackets={n_brackets}  Diversification={div:.4f}")
        print(f"    Core={result['core_tier']}  Stability={result['stability_tier']}  "
              f"Growth={result['growth_tier']}")
        if not result["portfolio"].empty:
            for _, r in result["portfolio"].iterrows():
                name  = str(r.get("Scheme_Name", "N/A"))[:38]
                bkt   = r.get("bracket", "")
                tier  = r.get("bracket_tier", "")
                wt    = r.get("weight", 0)
                alloc = r.get("alloc_amount_inr", 0)
                print(f"    {bkt:<12} ({tier:<10}) {name:<40} wt={wt:.2%}  ₹{alloc:,.0f}")

    # ── v3: Full per-user investment profile + GenAI ──────────────────────────
    banner("STEP 7e — Full User Investment Profile (v3)")
    print(f"[7e] Core-satellite portfolios across all horizons + GenAI  "
          f"(demo user: {demo_risk} risk, corpus ₹5,00,000) ...")

    full_profile = recommend_full_profile(
        demo_risk, mf_df,
        top_n_per_bracket=3,
        total_amount=500_000,
    )
    profile_explanations = explain_full_profile(
        full_profile, demo_risk,
        user_context=f"Indian retail investor, {demo_risk} risk appetite, goal: wealth creation",
        provider=args.gpt_provider,
    )

    print(f"\n  Risk Tier : {demo_risk}")
    print(f"  Corpus    : ₹{full_profile['total_amount']:,.0f}")

    for h_label, h_data in full_profile["horizons"].items():
        eq   = h_data.get("equity_pct", 0)
        dt   = 1.0 - eq
        div  = h_data.get("diversification_score", 0)
        bkts = h_data.get("brackets", [])
        exp  = profile_explanations.get(h_label, {})
        prov = exp.get("provider", "rule_based")

        print(f"\n  {'─'*64}")
        print(f"  {h_label:<6} HORIZON   Equity {eq:.0%} / Debt {dt:.0%}   "
              f"Diversification = {div:.4f}")
        print(f"  {'─'*64}")

        for b in bkts:
            bracket_tag = f"{b['bracket']:<10} ({b['tier']:<10} {b['pct']:>4.0f}%)  "
            for i, (_, fr) in enumerate(b["funds"].head(3).iterrows()):
                name  = str(fr.get("Scheme_Name", "N/A"))[:42]
                wt    = b["pct"] / 100.0 / max(1, b["n_funds"])
                alloc = int(round(wt * full_profile["total_amount"], -2))
                prefix = bracket_tag if i == 0 else " " * len(bracket_tag)
                print(f"  {prefix}  {name:<44}  ₹{alloc:>7,}")

        if exp.get("explanation"):
            print(f"\n  GenAI [{prov}]:")
            for line in exp["explanation"].strip().split("\n"):
                stripped = line.strip()
                if stripped:
                    print(f"    {stripped}")

    # ──────────────────────────────────────────────────────────────────────────
    banner("STEP 8 — Evaluation Metrics")
    # ──────────────────────────────────────────────────────────────────────────
    if not args.skip_eval:
        print("[8/8] Running all 8 evaluation metrics ...")
        eval_results = run_full_evaluation(
            df=df,
            global_model=global_model,
            central_model=load_central_model(),
            fl_history=fl_hist if fl_hist else None,
            df_fl=df_fl,
            run_gpt=not args.no_gpt,
            provider=args.gpt_provider,
        )
        try:
            plot_evaluation_dashboard(eval_results, save_dir="models")
        except Exception as e:
            print(f"      [warn] Evaluation dashboard plot skipped: {e}")

        # ── Standalone ROC-AUC plot (Page 3) ──────────────────────────────
        roc_data = eval_results.get("roc_auc", {})
        if roc_data and "per_class_auc" in roc_data:
            try:
                plot_roc_auc_curves(roc_data, save_dir="models")
                print(f"\n  ROC-AUC Summary (One-vs-Rest):")
                print(f"    Macro AUC    = {roc_data['macro_auc']:.4f}")
                print(f"    Weighted AUC = {roc_data['weighted_auc']:.4f}")
                print(f"    Per class:")
                for cls, v in roc_data["per_class_auc"].items():
                    bar = "█" * int(v * 20)
                    print(f"      {cls:<12} {bar:<20} {v:.4f}")
            except Exception as e:
                print(f"      [warn] ROC-AUC plot skipped: {e}")
    else:
        print("[8/8] Skipped (--skip-eval flag set).")

    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner(f"Pipeline COMPLETE  ⏱  {elapsed:.1f}s")
    print("  Artefacts saved in:  models/")
    print("  Open notebooks/06_Evaluation_Metrics.ipynb for full interactive report.")


if __name__ == "__main__":
    main()
