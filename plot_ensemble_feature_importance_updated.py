"""
plot_ensemble_feature_importance_updated.py

Enhanced plotting and JSON export for ALL 5-model ensemble feature importances.
- Plots all 5 base models (RF, XGB, LGBM, ET, CatBoost) + Meta-learner weights
- Exports feature importances to JSON files in descending order
- Creates comprehensive summary JSON with all models
"""

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict

# Configuration
MODELS_DIR = Path(__file__).resolve().parent / "models"
ENSEMBLE_META_PATH = MODELS_DIR / "ensemble_fund_meta.json"


def normalize_importance(imp_dict: Dict) -> Dict:
    """Normalize importance values to sum to 1.0."""
    total = sum(imp_dict.values())
    if total == 0:
        return {k: 0.0 for k in imp_dict.keys()}
    return {k: float(v) / total for k, v in imp_dict.items()}


def sort_importance_descending(imp_dict: Dict, top_n: Optional[int] = None) -> Dict:
    """
    Sort feature importances in descending order.
    
    Parameters
    ----------
    imp_dict : Dict
        Feature name -> importance value
    top_n : Optional[int]
        If specified, return only top N features
    
    Returns
    -------
    Dict
        OrderedDict sorted by importance (descending)
    """
    sorted_items = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_items = sorted_items[:top_n]
    return OrderedDict(sorted_items)


def export_importance_json(
    meta: Dict,
    save_dir: Optional[Path] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Export feature importances to JSON files for each model.
    
    Parameters
    ----------
    meta : Dict
        Ensemble metadata dict (loaded from ensemble_fund_meta.json)
    save_dir : Optional[Path]
        Output directory (default: models/)
    top_n : Optional[int]
        If specified, export only top N features per model
    
    Returns
    -------
    Dict[str, Path]
        Mapping of model names to their JSON file paths
    """
    save_dir = Path(save_dir) if save_dir else MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Extract feature importances for all 5 base models
    models_data = {
        "random_forest": {
            "importance": meta.get("rf_feature_importance", {}),
            "r2": meta.get("rf_r2", 0),
            "rmse": meta.get("rf_rmse", 0),
            "cv_r2_mean": meta.get("rf_cv_r2_mean", 0),
            "cv_r2_std": meta.get("rf_cv_r2_std", 0),
        },
        "xgboost": {
            "importance": meta.get("xgb_feature_importance", {}),
            "r2": meta.get("xgb_r2", 0),
            "rmse": meta.get("xgb_rmse", 0),
            "cv_r2_mean": meta.get("xgb_cv_r2_mean", 0),
            "cv_r2_std": meta.get("xgb_cv_r2_std", 0),
        },
        "lightgbm": {
            "importance": meta.get("lgbm_feature_importance", {}),
            "r2": meta.get("lgbm_r2", 0),
            "rmse": meta.get("lgbm_rmse", 0),
            "cv_r2_mean": meta.get("lgbm_cv_r2_mean", 0),
            "cv_r2_std": meta.get("lgbm_cv_r2_std", 0),
        },
        "extratrees": {
            "importance": meta.get("et_feature_importance", {}),
            "r2": meta.get("et_r2", 0),
            "rmse": meta.get("et_rmse", 0),
            "cv_r2_mean": meta.get("et_cv_r2_mean", 0),
            "cv_r2_std": meta.get("et_cv_r2_std", 0),
        },
        "catboost": {
            "importance": meta.get("cat_feature_importance", {}),
            "r2": meta.get("cat_r2", 0),
            "rmse": meta.get("cat_rmse", 0),
            "cv_r2_mean": meta.get("cat_cv_r2_mean", 0),
            "cv_r2_std": meta.get("cat_cv_r2_std", 0),
        },
    }
    
    # Export JSON for each model
    for model_name, model_data in models_data.items():
        imp_dict = model_data["importance"]
        if not imp_dict:
            print(f"⚠️  {model_name.upper()} has no importance data, skipping JSON export")
            continue
        
        # Normalize and sort importances
        normalized = normalize_importance(imp_dict)
        sorted_imp = sort_importance_descending(normalized, top_n=top_n)
        
        # Create output structure
        output_data = {
            "model": model_name,
            "performance_metrics": {
                "r2": float(model_data["r2"]),
                "rmse": float(model_data["rmse"]),
                "cv_r2_mean": float(model_data["cv_r2_mean"]),
                "cv_r2_std": float(model_data["cv_r2_std"]),
            },
            "total_features": len(imp_dict),
            "exported_features": len(sorted_imp),
            "feature_importances": dict(sorted_imp),
        }
        
        # Write JSON file
        output_path = save_dir / f"importance_{model_name}_sorted.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        output_files[model_name] = output_path
        print(f"✓ {model_name.upper():12} → {output_path}")
    
    return output_files


def export_ensemble_summary_json(
    meta: Dict,
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Export comprehensive ensemble summary with all models and meta-learner weights.
    
    Parameters
    ----------
    meta : Dict
        Ensemble metadata dict
    save_dir : Optional[Path]
        Output directory
    
    Returns
    -------
    Path
        Path to summary JSON file
    """
    save_dir = Path(save_dir) if save_dir else MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate all feature importances
    all_importances = {
        "rf": meta.get("rf_feature_importance", {}),
        "xgb": meta.get("xgb_feature_importance", {}),
        "lgbm": meta.get("lgbm_feature_importance", {}),
        "et": meta.get("et_feature_importance", {}),
        "cat": meta.get("cat_feature_importance", {}),
    }
    
    # Compute ensemble-wide importance (average of all models)
    all_features = set()
    for imp_dict in all_importances.values():
        all_features.update(imp_dict.keys())
    
    ensemble_importance = {}
    for feature in all_features:
        importances = []
        for imp_dict in all_importances.values():
            if feature in imp_dict:
                importances.append(imp_dict[feature])
        if importances:
            ensemble_importance[feature] = float(np.mean(importances))
    
    # Normalize and sort
    ensemble_importance_norm = normalize_importance(ensemble_importance)
    ensemble_importance_sorted = sort_importance_descending(ensemble_importance_norm)
    
    # Create summary
    summary_data = {
        "ensemble_config": {
            "model_type": meta.get("model_type", "RF+XGB+LGBM+ET+CAT+META"),
            "ensemble_mode": meta.get("ensemble_mode", "stacking"),
            "n_models": meta.get("n_models", 5),
            "n_features": meta.get("n_features", 0),
            "n_funds_trained": meta.get("n_funds", 0),
            "uses_history": meta.get("uses_history", False),
        },
        "base_model_performance": {
            "random_forest": {
                "r2": float(meta.get("rf_r2", 0)),
                "rmse": float(meta.get("rf_rmse", 0)),
                "cv_r2": f"{meta.get('rf_cv_r2_mean', 0):.4f}±{meta.get('rf_cv_r2_std', 0):.4f}",
            },
            "xgboost": {
                "r2": float(meta.get("xgb_r2", 0)),
                "rmse": float(meta.get("xgb_rmse", 0)),
                "cv_r2": f"{meta.get('xgb_cv_r2_mean', 0):.4f}±{meta.get('xgb_cv_r2_std', 0):.4f}",
            },
            "lightgbm": {
                "r2": float(meta.get("lgbm_r2", 0)),
                "rmse": float(meta.get("lgbm_rmse", 0)),
                "cv_r2": f"{meta.get('lgbm_cv_r2_mean', 0):.4f}±{meta.get('lgbm_cv_r2_std', 0):.4f}",
            },
            "extratrees": {
                "r2": float(meta.get("et_r2", 0)),
                "rmse": float(meta.get("et_rmse", 0)),
                "cv_r2": f"{meta.get('et_cv_r2_mean', 0):.4f}±{meta.get('et_cv_r2_std', 0):.4f}",
            },
            "catboost": {
                "r2": float(meta.get("cat_r2", 0)),
                "rmse": float(meta.get("cat_rmse", 0)),
                "cv_r2": f"{meta.get('cat_cv_r2_mean', 0):.4f}±{meta.get('cat_cv_r2_std', 0):.4f}",
            },
        },
        "meta_learner": {
            "type": "Ridge Stacking",
            "r2": float(meta.get("meta_r2", 0)),
            "rmse": float(meta.get("meta_rmse", 0)),
            "learned_weights": meta.get("meta_weights", {}),
        },
        "feature_importance_summary": {
            "ensemble_average": dict(ensemble_importance_sorted),
            "total_features": len(ensemble_importance_sorted),
            "top_10_features": dict(list(ensemble_importance_sorted.items())[:10]),
        },
        "individual_models_top_10": {
            model_name: dict(list(sort_importance_descending(imp_dict).items())[:10])
            for model_name, imp_dict in all_importances.items()
        },
    }
    
    output_path = save_dir / "ensemble_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✓ ENSEMBLE SUMMARY → {output_path}")
    return output_path


def plot_ensemble_importance_all_models(
    meta: Dict,
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Plot feature importances for ALL 5 models + meta-learner weights.
    Creates a comprehensive multi-panel visualization.
    
    Parameters
    ----------
    meta : Dict
        Ensemble metadata dict
    save_dir : Optional[Path]
        Output directory
    
    Returns
    -------
    Path
        Path to output PNG
    """
    save_dir = Path(save_dir) if save_dir else MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get importances for all models
    models_config = [
        ("Random Forest", meta.get("rf_feature_importance", {}), "#42A5F5", meta.get("rf_r2", 0)),
        ("XGBoost", meta.get("xgb_feature_importance", {}), "#EF5350", meta.get("xgb_r2", 0)),
        ("LightGBM", meta.get("lgbm_feature_importance", {}), "#66BB6A", meta.get("lgbm_r2", 0)),
        ("ExtraTreesRegressor", meta.get("et_feature_importance", {}), "#FFA726", meta.get("et_r2", 0)),
        ("CatBoost", meta.get("cat_feature_importance", {}), "#AB47BC", meta.get("cat_r2", 0)),
    ]
    
    # Get feature list (use first available model with data)
    features = None
    for model_name, imp_dict, _, _ in models_config:
        if imp_dict:
            features = list(imp_dict.keys())
            break
    
    if not features:
        print("❌ No feature importance data available!")
        return None
    
    # Create 2x3 subplots (5 models + 1 meta-learner weights)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"5-Model Ensemble Feature Importances (RF+XGB+LGBM+ET+CAT with Ridge Meta-Learner)\n"
        f"Trained on {meta.get('n_funds', 0):,} Funds with {meta.get('n_features', 0)} Features",
        fontsize=14,
        fontweight="bold"
    )
    
    axes = axes.flatten()
    
    # Plot each model's top features
    for idx, (model_name, imp_dict, color, r2) in enumerate(models_config):
        ax = axes[idx]
        
        if not imp_dict:
            ax.text(0.5, 0.5, f"{model_name}\n(No data)", 
                   ha="center", va="center", fontsize=12)
            ax.set_title(f"{model_name} — R²=N/A")
            ax.axis("off")
            continue
        
        # Normalize and sort
        normalized = normalize_importance(imp_dict)
        sorted_imp = sort_importance_descending(normalized, top_n=15)
        
        features_top = list(sorted_imp.keys())
        importances_top = list(sorted_imp.values())
        
        bars = ax.barh(features_top, importances_top, color=color, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Normalized Importance", fontsize=10)
        ax.set_title(f"{model_name}\nR²={r2:.4f}", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importances_top)):
            ax.text(val, i, f"  {val:.4f}", va="center", fontsize=8)
    
    # Plot meta-learner weights
    meta_weights = meta.get("meta_weights", {})
    ax = axes[5]
    
    if meta_weights:
        models_list = list(meta_weights.keys())
        weights_list = list(meta_weights.values())
        colors_meta = ["#42A5F5", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC"][:len(models_list)]
        
        bars = ax.bar(models_list, weights_list, color=colors_meta, alpha=0.8, edgecolor="white")
        ax.set_ylabel("Learned Weight", fontsize=10)
        ax.set_title(
            f"Ridge Meta-Learner Weights\nR²={meta.get('meta_r2', 0):.4f}",
            fontsize=11,
            fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels
        for bar, weight in zip(bars, weights_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f"{weight:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        # Add horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    
    plt.tight_layout()
    output_path = save_dir / "plot_ensemble_all_models_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ ALL MODELS PLOT → {output_path}")
    return output_path


def plot_ensemble_importance_comparison(
    meta: Dict,
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Create a grouped bar chart comparing top features across all models.
    
    Parameters
    ----------
    meta : Dict
        Ensemble metadata dict
    save_dir : Optional[Path]
        Output directory
    
    Returns
    -------
    Path
        Path to output PNG
    """
    save_dir = Path(save_dir) if save_dir else MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all importances
    models_data = {
        "RF": meta.get("rf_feature_importance", {}),
        "XGB": meta.get("xgb_feature_importance", {}),
        "LGBM": meta.get("lgbm_feature_importance", {}),
        "ET": meta.get("et_feature_importance", {}),
        "CAT": meta.get("cat_feature_importance", {}),
    }
    
    # Get unified feature list
    all_features = set()
    for imp_dict in models_data.values():
        all_features.update(imp_dict.keys())
    
    # Compute ensemble average
    ensemble_imp = {}
    for feature in all_features:
        importances = []
        for imp_dict in models_data.values():
            if feature in imp_dict:
                importances.append(imp_dict[feature])
        if importances:
            ensemble_imp[feature] = np.mean(importances)
    
    # Get top 15 features by ensemble average
    sorted_ensemble = sort_importance_descending(ensemble_imp, top_n=15)
    top_features = list(sorted_ensemble.keys())
    
    # Normalize all models' importances
    normalized_models = {
        name: normalize_importance(imp_dict)
        for name, imp_dict in models_data.items()
    }
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_features))
    width = 0.15
    colors = ["#42A5F5", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC"]
    
    for idx, (model_name, color) in enumerate(zip(models_data.keys(), colors)):
        imp_values = [normalized_models[model_name].get(f, 0) for f in top_features]
        ax.bar(x + idx * width, imp_values, width, label=model_name, color=color, alpha=0.8, edgecolor="white")
    
    ax.set_xlabel("Features (Top 15 by Ensemble Average)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Importance", fontsize=11, fontweight="bold")
    ax.set_title(
        "Ensemble Feature Importance Comparison (Top 15 Features)\n"
        "All 5 Base Models + Ridge Meta-Learner",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(top_features, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    output_path = save_dir / "plot_ensemble_comparison_top15.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ COMPARISON PLOT → {output_path}")
    return output_path


def main():
    """Main execution: Load ensemble metadata and generate all plots/JSON exports."""
    
    print("\n" + "=" * 80)
    print("  5-MODEL ENSEMBLE FEATURE IMPORTANCE ANALYSIS & EXPORT")
    print("=" * 80 + "\n")
    
    # Load ensemble metadata
    if not ENSEMBLE_META_PATH.exists():
        print(f"❌ Ensemble metadata not found: {ENSEMBLE_META_PATH}")
        print("   Run train.py first to generate the ensemble models.\n")
        return
    
    with open(ENSEMBLE_META_PATH) as f:
        meta = json.load(f)
    
    print(f"✓ Loaded ensemble metadata from: {ENSEMBLE_META_PATH}\n")
    
    # Print ensemble summary
    print("ENSEMBLE CONFIGURATION:")
    print(f"  Model Type: {meta.get('model_type', 'N/A')}")
    print(f"  Ensemble Mode: {meta.get('ensemble_mode', 'N/A')}")
    print(f"  Total Models: {meta.get('n_models', 0)}")
    print(f"  Total Features: {meta.get('n_features', 0)}")
    print(f"  Funds Trained: {meta.get('n_funds', 0):,}\n")
    
    print("BASE MODEL PERFORMANCE:")
    print(f"  RF:   R²={meta.get('rf_r2', 0):.4f}   RMSE={meta.get('rf_rmse', 0):.4f}")
    print(f"  XGB:  R²={meta.get('xgb_r2', 0):.4f}   RMSE={meta.get('xgb_rmse', 0):.4f}")
    print(f"  LGBM: R²={meta.get('lgbm_r2', 0):.4f}   RMSE={meta.get('lgbm_rmse', 0):.4f}")
    print(f"  ET:   R²={meta.get('et_r2', 0):.4f}   RMSE={meta.get('et_rmse', 0):.4f}")
    print(f"  CAT:  R²={meta.get('cat_r2', 0):.4f}   RMSE={meta.get('cat_rmse', 0):.4f}\n")
    
    print("META-LEARNER PERFORMANCE:")
    print(f"  Ridge: R²={meta.get('meta_r2', 0):.4f}   RMSE={meta.get('meta_rmse', 0):.4f}\n")
    
    print("META-LEARNER LEARNED WEIGHTS:")
    meta_weights = meta.get("meta_weights", {})
    for model, weight in meta_weights.items():
        print(f"  {model.upper():6} → {weight:+.4f}")
    print()
    
    # Generate all outputs
    print("GENERATING OUTPUTS:\n")
    print("1. JSON EXPORTS (Feature Importances in Descending Order):")
    export_importance_json(meta, MODELS_DIR)
    
    print("\n2. ENSEMBLE SUMMARY JSON:")
    export_ensemble_summary_json(meta, MODELS_DIR)
    
    print("\n3. PLOTS:")
    print("   a) All 5 Models + Meta-Learner Weights:")
    plot_ensemble_importance_all_models(meta, MODELS_DIR)
    
    print("   b) Comparison (Top 15 Features Across Models):")
    plot_ensemble_importance_comparison(meta, MODELS_DIR)
    
    print("\n" + "=" * 80)
    print("  ✓ ANALYSIS COMPLETE!")
    print("=" * 80 + "\n")
    
    # Summary
    print("GENERATED FILES:")
    print(f"  ├─ models/importance_random_forest_sorted.json")
    print(f"  ├─ models/importance_xgboost_sorted.json")
    print(f"  ├─ models/importance_lightgbm_sorted.json")
    print(f"  ├─ models/importance_extratrees_sorted.json")
    print(f"  ├─ models/importance_catboost_sorted.json")
    print(f"  ├─ models/ensemble_summary.json")
    print(f"  ├─ models/plot_ensemble_all_models_importance.png")
    print(f"  └─ models/plot_ensemble_comparison_top15.png\n")


if __name__ == "__main__":
    main()
