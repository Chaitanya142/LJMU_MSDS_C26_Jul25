"""
Comprehensive system diagnostics for ensemble model.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from src.recommender import load_mutual_funds
from src.nav_history import load_nav_metrics
from src.ensemble_recommender import build_fund_features, _build_quality_target

print("\n" + "=" * 80)
print("SMART FUND ADVISOR — COMPREHENSIVE SYSTEM DIAGNOSTICS")
print("=" * 80)

# 1. Ensemble metadata
print("\n[1] ENSEMBLE METADATA")
print("-" * 80)
meta_path = Path("models/ensemble_fund_meta.json")
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    
    print(f"Training target: {meta.get('train_target')}")
    print(f"Train rows: {meta.get('n_funds'):,}")
    print(f"Num features: {meta.get('n_features')}")
    print(f"Model type: {meta.get('model_type')}")
    print(f"\nRF  → Train R²={meta.get('rf_r2'):.4f}, CV R²={meta.get('rf_cv_r2_mean'):.4f}±{meta.get('rf_cv_r2_std'):.4f}, RMSE={meta.get('rf_rmse'):.4f}")
    print(f"XGB → Train R²={meta.get('xgb_r2'):.4f}, CV R²={meta.get('xgb_cv_r2_mean'):.4f}±{meta.get('xgb_cv_r2_std'):.4f}, RMSE={meta.get('xgb_rmse'):.4f}")
    print(f"LGBM→ Train R²={meta.get('lgbm_r2'):.4f}, CV R²={meta.get('lgbm_cv_r2_mean'):.4f}±{meta.get('lgbm_cv_r2_std'):.4f}, RMSE={meta.get('lgbm_rmse'):.4f}")
    
    dq = meta.get('data_quality_modes', {})
    print(f"\nData Quality: Mode A={dq.get('A_nav_history'):,} | Mode B={dq.get('B_fundperf_only'):,} | Mode C={dq.get('C_low_information'):,}")
    
    fp = meta.get('forward_panel', {})
    pos_rate = fp.get('panel_positive', 0) / fp.get('panel_rows', 1) if fp.get('panel_rows') else 0
    print(f"Forward panel: {fp.get('panel_rows'):,} rows, {fp.get('panel_positive'):,} positives ({pos_rate:.2%})")
    
    print("\nTop 15 XGB Feature Importances:")
    xgb_imp = meta.get('xgb_feature_importance', {})
    sorted_imp = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    for fname, imp in sorted_imp:
        print(f"  {fname:35s} {imp:7.4f}")

# 2. Load and analyze data
print("\n[2] DATA ANALYSIS")
print("-" * 80)
try:
    mf_df = load_mutual_funds()
    print(f"Loaded {len(mf_df):,} mutual funds")
    
    nav_metrics = load_nav_metrics(verbose=False)
    print(f"Loaded NAV metrics for {len(nav_metrics):,} schemes")
    
    X_df, feat_cols = build_fund_features(mf_df, nav_metrics)
    print(f"Feature matrix shape: {X_df.shape}")
    
    # Check for NaN, inf
    nan_count = X_df.isna().sum().sum()
    inf_count = np.isinf(X_df.values.astype(float)).sum()
    print(f"NaN values: {nan_count}, Inf values: {inf_count}")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    stats = X_df.describe().loc[['mean', 'std', 'min', 'max']]
    print(f"  Mean (across all features): {X_df.mean().mean():.4f}")
    print(f"  Std  (across all features): {X_df.std().mean():.4f}")
    print(f"  Min values range: [{X_df.min().min():.4f}, {X_df.values.min():.4f}]")
    print(f"  Max values range: [{X_df.max().max():.4f}, {X_df.values.max():.4f}]")
    
    # Feature correlation analysis
    print(f"\nFeature Correlation Analysis (top 10 most correlated pairs):")
    corr_matrix = X_df.corr().abs()
    # Get upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_pairs = upper.stack().sort_values(ascending=False)[:10]
    for (f1, f2), corr_val in corr_pairs.items():
        if corr_val > 0.95:
            print(f"  [HIGH] {f1:35s} <-> {f2:35s}  r={corr_val:.4f} ⚠️  REDUNDANT")
        elif corr_val > 0.80:
            print(f"  [MED]  {f1:35s} <-> {f2:35s}  r={corr_val:.4f}")
        else:
            print(f"  [LOW]  {f1:35s} <-> {f2:35s}  r={corr_val:.4f}")
    
    # VIF (simplified check for high correlations)
    high_corr_features = []
    for col in X_df.columns:
        other_corrs = corr_matrix[col].drop(col)
        if (other_corrs > 0.95).any():
            high_corr_features.append(col)
    
    if high_corr_features:
        print(f"\n⚠️  {len(high_corr_features)} features with correlation > 0.95 to other features:")
        for feat in high_corr_features[:5]:
            print(f"     - {feat}")

except Exception as e:
    print(f"Error during data analysis: {e}")

# 3. Target variable analysis
print("\n[3] TARGET VARIABLE ANALYSIS")
print("-" * 80)
try:
    y_rule = _build_quality_target(mf_df, X_df, nav_metrics)
    print(f"Target shape: {y_rule.shape}")
    print(f"Target stats: min={y_rule.min():.4f}, max={y_rule.max():.4f}, mean={y_rule.mean():.4f}, std={y_rule.std():.4f}")
    print(f"Target distribution:")
    print(f"  0-10%: {(y_rule < 0.1).sum():,} ({(y_rule < 0.1).sum() / len(y_rule) * 100:.1f}%)")
    print(f"  10-50%: {((y_rule >= 0.1) & (y_rule < 0.5)).sum():,} ({((y_rule >= 0.1) & (y_rule < 0.5)).sum() / len(y_rule) * 100:.1f}%)")
    print(f"  50-90%: {((y_rule >= 0.5) & (y_rule < 0.9)).sum():,} ({((y_rule >= 0.5) & (y_rule < 0.9)).sum() / len(y_rule) * 100:.1f}%)")
    print(f"  90-100%: {(y_rule >= 0.9).sum():,} ({(y_rule >= 0.9).sum() / len(y_rule) * 100:.1f}%)")
except Exception as e:
    print(f"Error during target analysis: {e}")

# 4. Output recommendations
print("\n[4] IDENTIFIED ISSUES & RECOMMENDATIONS")
print("-" * 80)
issues = []

if meta.get('xgb_r2', 0) < 0.5 or meta.get('lgbm_r2', 0) < 0.5:
    issues.append("❌ XGB/LGBM train R² < 0.5: Model underfitting → increase model capacity or improve target signal")

gap_xgb = meta.get('xgb_r2', 0) - meta.get('xgb_cv_r2_mean', 0)
gap_lgbm = meta.get('lgbm_r2', 0) - meta.get('lgbm_cv_r2_mean', 0)
if gap_xgb > 0.20 or gap_lgbm > 0.20:
    issues.append(f"⚠️  Large generalization gap (XGB={gap_xgb:.3f}, LGBM={gap_lgbm:.3f}): Overfitting → increase regularization")

if meta.get('rf_r2', 0) < 0.75:
    issues.append("⚠️  RF train R² < 0.75: Lower than expected → consider feature engineering")

if pos_rate < 0.15 or pos_rate > 0.35:
    issues.append(f"⚠️  Target imbalance ({pos_rate:.2%}): Class imbalance may hurt model → consider weighted loss or resampling")

if len(feat_cols) > 50:
    issues.append(f"⚠️  Too many features ({len(feat_cols)}): May introduce noise → consider feature selection")

if high_corr_features:
    issues.append(f"⚠️  Multicollinearity detected ({len(high_corr_features)} features): Remove redundant features")

if not issues:
    issues.append("✅ No critical issues detected")

for issue in issues:
    print(f"  {issue}")

print("\n" + "=" * 80)
