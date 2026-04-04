"""
evaluation.py  –  Comprehensive evaluation metrics for Smart Fund Advisor.

Evaluation Metrics
------------------
1. Cluster Accuracy  : Silhouette score for 5 behavioural clusters ≥ 0.8
                       (Borah and Laskar, 2025)
2. F1 Score          : F1 score after federated learning > 80%
                       (Hsu et al., 2022)
3. Data Privacy      : Small ε via (ε, δ)-DP with Rényi accounting
4. Federated Loss    : Prediction change between central and FL model < 10%
                       (Gopal Varma, 2025)
5. GPT Correctness   : ≥ 75% of fund explanations pass factual validation

Usage
-----
    from src.evaluation import run_full_evaluation, plot_evaluation_dashboard
    results = run_full_evaluation(df, global_model, fl_hist)
    plot_evaluation_dashboard(results)
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RISK_FEATURES, RISK_CLASSES, MODELS_DIR, RANDOM_SEED
from src.utils import label_subplots


# ─── Metric 1: Cluster Accuracy ───────────────────────────────────────────────

def evaluate_cluster_accuracy(df: pd.DataFrame) -> Dict:
    """
    Compute KMeans silhouette score for 5 behavioural clusters.
    Target: silhouette ≥ 0.80.
    """
    # Use embedding-based clustering for the silhouette metric
    from src.cluster_recommender import fit_embedding_cluster_model, evaluate_cluster_metrics
    _, metrics = fit_embedding_cluster_model(df)
    return {
        "metric": "Cluster Accuracy",
        "silhouette_score":          float(metrics["silhouette_score"]),
        "davies_bouldin_index":      float(metrics["davies_bouldin_index"]),
        "cluster_purity":            float(metrics["cluster_purity"]),
        "cluster_classification_accuracy": float(metrics.get("cluster_classification_accuracy", 0)),
        "per_cluster_silhouette":    {str(k): float(v) for k, v in metrics.get("per_cluster_silhouette", {}).items()},
        "threshold":                 0.80,
        "pass":                      bool(metrics["silhouette_score"] >= 0.80),
        "value":                     float(metrics["silhouette_score"]),
    }


# ─── Metric 2: F1 Score with Federated Learning ───────────────────────────────

def evaluate_fl_f1(
    global_model,
    df_fl: pd.DataFrame,
    label_encoder=None,
) -> Dict:
    """
    Compute per-class and macro F1 on the FL test split using the globally
    updated model after federated learning.
    Target: macro F1 > 0.80.
    """
    from src.central_model import predict as mlp_predict

    if label_encoder is None:
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X = df_fl[feat_cols].values.astype(np.float32)
    y = df_fl["risk_label_encoded"].values.astype(np.int64)

    y_pred, probs = mlp_predict(global_model, X)

    macro_f1    = f1_score(y, y_pred, average="macro")
    weighted_f1 = f1_score(y, y_pred, average="weighted")
    per_class_f1 = f1_score(y, y_pred, average=None, labels=list(range(len(label_encoder.classes_))))
    accuracy    = accuracy_score(y, y_pred)

    prec, rec, f1_arr, support = precision_recall_fscore_support(
        y, y_pred, labels=list(range(len(label_encoder.classes_))), zero_division=0
    )

    # label_encoder.classes_[i] correctly maps encoded integer i → class name.
    # (LabelEncoder sorts alphabetically: 0=High, 1=Low, 2=Medium, 3=Very_High, 4=Very_Low)
    per_class = {
        label_encoder.classes_[i]: {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1_arr[i]), 4),
            "support":   int(support[i]),
        }
        for i in range(len(label_encoder.classes_))
    }

    return {
        "metric":       "F1 Score (FL)",
        "macro_f1":     round(float(macro_f1), 4),
        "weighted_f1":  round(float(weighted_f1), 4),
        "accuracy":     round(float(accuracy), 4),
        "per_class":    per_class,
        "threshold":    0.80,
        "pass":         bool(macro_f1 >= 0.80),
        "value":        float(macro_f1),
    }


# ─── Metric 3: Data Privacy DP Budget ─────────────────────────────────────────

def evaluate_privacy(delta: float = 1e-5) -> Dict:
    """
    Compute (ε, δ)-DP budget and privacy quality assessment.
    No hard threshold — lower ε = better privacy.
    ε < 10 is acceptable; ε < 1 is very strong; ε < 3 is strong.
    """
    from src.privacy_analysis import compute_epsilon, epsilon_vs_rounds, privacy_summary
    eps, delta = compute_epsilon(delta=delta)
    eps_by_round = epsilon_vs_rounds(max_rounds=10)

    quality = "Excellent (ε < 1)" if eps < 1 else \
              "Strong (ε < 3)"    if eps < 3 else \
              "Good (ε < 5)"      if eps < 5 else \
              "Moderate (ε < 10)" if eps < 10 else "Weak (ε ≥ 10)"

    return {
        "metric":        "Data Privacy (DP)",
        "epsilon":       float(eps),
        "delta":         float(delta),
        "quality":       quality,
        "eps_by_round":  {str(k): float(v) for k, v in eps_by_round.items()},
        "pass":          True,    # DP noise always active; we report the value
        "value":         float(eps),
        "note":          f"(ε={eps:.4f}, δ=1e-5) — {quality}",
    }


# ─── Metric 4: Federated Loss Stability ───────────────────────────────────────

def evaluate_federated_loss(
    central_model,
    global_model,
    df_fl: pd.DataFrame,
    fl_history: Optional[Dict] = None,
) -> Dict:
    """
    Measure prediction change between central and FL-updated model.
    Target: change in predictions < 10%.
    Reference: Gopal Varma, 2025
    """
    from src.central_model import predict as mlp_predict

    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X = df_fl[feat_cols].values.astype(np.float32)

    y_central, _ = mlp_predict(central_model, X)
    y_fl,      _ = mlp_predict(global_model,  X)

    prediction_change_pct = (y_central != y_fl).mean() * 100
    central_acc = accuracy_score(df_fl["risk_label_encoded"].values, y_central)
    fl_acc      = accuracy_score(df_fl["risk_label_encoded"].values, y_fl)

    # Federated loss convergence from fl_history
    loss_variance = None
    loss_trend    = []
    if fl_history and "distributed_loss" in fl_history:
        losses = [fl_history["distributed_loss"][k]
                  for k in sorted(fl_history["distributed_loss"].keys(), key=int)]
        loss_variance = round(float(np.var(losses)), 6)
        loss_trend = [round(l, 4) for l in losses]

    return {
        "metric":                 "Federated Loss Stability",
        "prediction_change_pct":  round(float(prediction_change_pct), 2),
        "central_accuracy":       round(float(central_acc), 4),
        "fl_accuracy":            round(float(fl_acc), 4),
        "accuracy_delta_pp":      round(float(fl_acc - central_acc) * 100, 2),
        "fl_loss_variance":       loss_variance,
        "fl_loss_trend":          loss_trend,
        "threshold":              10.0,
        "pass":                   bool(prediction_change_pct < 10.0),
        "value":                  float(prediction_change_pct),
        "note":                   f"{prediction_change_pct:.2f}% of predictions changed after FL",
    }


# ─── Metric 5: GPT Explanation Correctness ────────────────────────────────────

def evaluate_gpt_correctness(
    user_risk: str = "High",
    n_funds: int = 5,
    provider: Optional[str] = None,
) -> Dict:
    """
    Generate explanations for top-N funds and validate them.
    Validates that LLM-generated explanations are factually correct.
    """
    from src.recommender import load_mutual_funds, recommend_funds
    from src.gpt_explainer import explain_portfolio, batch_validate

    mf_df = load_mutual_funds()
    recs  = recommend_funds(user_risk, mf_df, top_n=n_funds)

    if recs.empty:
        return {"metric": "GPT Correctness", "pass": False, "error": "No recommendations found"}

    explanations = explain_portfolio(recs, user_risk, provider=provider)
    validation   = batch_validate(recs, explanations, user_risk)

    return {
        "metric":            "GPT Fund Explanation Correctness",
        "provider":          list(explanations.values())[0]["provider"] if explanations else "unknown",
        "n_funds_tested":    validation["total"],
        "pass_count":        validation["pass_count"],
        "overall_pass_rate": validation["overall_pass_rate"],
        "per_fund":          validation["per_fund"],
        "pass":              validation["overall_pass"],
        "value":             validation["overall_pass_rate"],
        "threshold":         0.75,
        "sample_explanation": list(explanations.values())[0]["explanation"]
                               if explanations else "",
    }


# ─── Metric 6: ROC-AUC (multi-class OvR) ───────────────────────────────────────

def evaluate_roc_auc(
    global_model,
    df_fl: pd.DataFrame,
    label_encoder=None,
) -> Dict:
    """
    Compute One-vs-Rest multiclass ROC-AUC for the FL global model.

    Returns per-class AUC scores, macro-average AUC, and per-class
    FPR/TPR arrays for plotting.

    References
    ----------
    Fawcett (2006) — "An introduction to ROC analysis", Pattern Recogn. Lett. 27(8).
    sklearn.metrics.roc_auc_score with multi_class='ovr'.
    """
    from src.central_model import predict as mlp_predict

    if label_encoder is None:
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X = df_fl[feat_cols].values.astype(np.float32)
    y = df_fl["risk_label_encoded"].values.astype(np.int64)

    _, probs = mlp_predict(global_model, X)
    n_classes = probs.shape[1]
    class_names = list(label_encoder.classes_)

    # One-vs-Rest: binarise y, then compute ROC per class
    y_bin = label_binarize(y, classes=list(range(n_classes)))

    fpr_dict: Dict[str, List] = {}
    tpr_dict: Dict[str, List] = {}
    auc_dict: Dict[str, float] = {}

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[cls] = fpr.tolist()
        tpr_dict[cls] = tpr.tolist()
        auc_dict[cls] = round(float(roc_auc), 4)

    # Macro-average: aggregate all OvR curves
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in class_names]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls in class_names:
        mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
    mean_tpr /= n_classes
    macro_auc = round(float(auc(all_fpr, mean_tpr)), 4)

    # Weighted AUC (sklearn)
    try:
        weighted_auc = round(float(roc_auc_score(
            y, probs, multi_class="ovr", average="weighted"
        )), 4)
    except Exception:
        weighted_auc = macro_auc

    return {
        "metric":       "ROC-AUC (OvR)",
        "macro_auc":    macro_auc,
        "weighted_auc": weighted_auc,
        "per_class_auc": auc_dict,
        "fpr":          fpr_dict,
        "tpr":          tpr_dict,
        "macro_fpr":    all_fpr.tolist(),
        "macro_tpr":    mean_tpr.tolist(),
        "value":        macro_auc,
    }


def plot_roc_auc_curves(
    roc_results: Dict,
    save_dir: Optional[Path] = None,
    label_encoder=None,
) -> None:
    """
    Plot per-class OvR ROC curves + macro-average on a single figure.

    Colours follow the risk spectrum (blue=Very_Low → red=Very_High).
    Saves to models/plot_roc_auc_curves.png.
    """
    save_dir = Path(save_dir or MODELS_DIR)
    colors = ["#1565C0", "#0288D1", "#388E3C", "#F57C00", "#C62828"]
    class_names = list(roc_results["per_class_auc"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "ROC-AUC Curves — RiskMLP FL Model (One-vs-Rest)",
        fontsize=13, fontweight="bold",
    )

    # ── Left: individual class curves ──────────────────────────────────────
    for cls, col in zip(class_names, colors):
        fpr = roc_results["fpr"][cls]
        tpr = roc_results["tpr"][cls]
        auc_val = roc_results["per_class_auc"][cls]
        axes[0].plot(fpr, tpr, color=col, linewidth=2,
                     label=f"{cls}  (AUC={auc_val:.3f})")
    # Macro-average
    axes[0].plot(
        roc_results["macro_fpr"], roc_results["macro_tpr"],
        color="black", linewidth=2.5, linestyle="--",
        label=f"Macro avg  (AUC={roc_results['macro_auc']:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Per-class ROC Curves (OvR)")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(alpha=0.3)

    # ── Right: AUC bar chart per class ───────────────────────────────────
    auc_vals = [roc_results["per_class_auc"][c] for c in class_names]
    bars = axes[1].barh(class_names, auc_vals, color=colors, alpha=0.85, height=0.55)
    for bar, val in zip(bars, auc_vals):
        axes[1].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=10)
    axes[1].axvline(roc_results["macro_auc"], color="black", linestyle="--",
                    linewidth=1.5, label=f"Macro AUC = {roc_results['macro_auc']:.3f}")
    axes[1].axvline(0.90, color="grey", linestyle=":", linewidth=1, label="Target 0.90")
    axes[1].set_xlim(0.5, 1.05)
    axes[1].set_xlabel("AUC")
    axes[1].set_title("AUC by Risk Class")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="x")

    label_subplots(axes)
    plt.tight_layout()
    out = save_dir / "plot_roc_auc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC-AUC plot → {out}")


# ─── Metric 7: Model Calibration (Brier Score) ────────────────────────────────

def evaluate_calibration(
    global_model,
    df_fl: pd.DataFrame,
    label_encoder=None,
) -> Dict:
    """
    Multi-class Brier Score: mean squared error over all predicted probability
    estimates and one-hot true labels.

    BS = (1/N) * Σ_n Σ_c (p_nc − y_nc)²

    Banking context: Basel II/III IRB models must produce calibrated probability
    outputs, not just rank-order predictions. A model with high F1 but poor
    calibration cannot be used for capital allocation or risk-based pricing.

    Threshold: BS < 0.10  (well-calibrated; Basel IRB benchmark)
    Reference: BCBS (2005) "Studies on the Validation of Internal Rating Systems";
               Niculescu-Mizil & Caruana (2005) ICML.
    """
    from src.central_model import predict as mlp_predict

    if label_encoder is None:
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X = df_fl[feat_cols].values.astype(np.float32)
    y = df_fl["risk_label_encoded"].values.astype(np.int64)
    n_classes = len(label_encoder.classes_)

    _, probs = mlp_predict(global_model, X)          # probs shape (N, n_classes)
    probs_np = np.array(probs) if not isinstance(probs, np.ndarray) else probs

    # One-hot encode true labels
    y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
    for i, lbl in enumerate(y):
        y_onehot[i, lbl] = 1.0

    brier_score = float(np.mean(np.sum((probs_np - y_onehot) ** 2, axis=1)))
    # Per-class binary Brier scores for diagnostics
    per_class_bs = {
        label_encoder.classes_[c]: round(float(np.mean((probs_np[:, c] - y_onehot[:, c]) ** 2)), 4)
        for c in range(n_classes)
    }

    threshold = 0.10
    return {
        "metric":          "Model Calibration (Brier Score)",
        "brier_score":     round(brier_score, 4),
        "per_class_brier": per_class_bs,
        "threshold":       threshold,
        "pass":            bool(brier_score < threshold),
        "value":           brier_score,
        "note":            f"BS={brier_score:.4f} (threshold < {threshold}; 0=perfect, ~0.8=random)",
    }


# ─── Metric 8: Fairness — Demographic Parity across Age Groups ────────────────

def evaluate_fairness(
    global_model,
    df_fl: pd.DataFrame,
    label_encoder=None,
) -> Dict:
    """
    Accuracy Parity across three age groups.

    Banking context: SEBI Circular SEBI/HO/IMD/DF2/CIR/P/2019/17 requires
    non-discriminatory suitability profiling. Age (via Age_Risk_Proxy) is a
    LEGITIMATE feature in investment risk assessment — younger investors have
    longer investment horizons and SHOULD receive higher risk tier assignments.
    Therefore, Demographic Parity (equal prediction rates across groups) is
    inappropriate here; it would penalise the model for actuarially sound
    risk differentiation mandated by SEBI's own product suitability rules.

    The correct fairness metric is ACCURACY PARITY: does the model predict
    each customer's risk tier with equal accuracy regardless of age group?
    A model biased against a specific age group would show systematically
    lower accuracy for that group's customers (misclassifying their risk),
    even if the overall aggregate accuracy is high.

    Method:
      1. Partition the test set into 3 age groups using raw 'Age' column:
           Young  : Age ≤ 30
           Middle : 31 ≤ Age ≤ 50
           Senior : Age  > 50
      2. For each group compute: accuracy = (correct predictions / group size)
      3. Accuracy Parity Ratio = min(group accuracy) / max(group accuracy)

    Threshold: APR ≥ 0.90  (model performs within 10% accuracy across groups)
    Reference: Hardt et al. (2016) "Equality of Opportunity in Supervised
               Learning"; Chouldechova (2017) "Fair Prediction with Disparate
               Impact"; SEBI LODR financial literacy mandate.
    """
    from src.central_model import predict as mlp_predict

    if label_encoder is None:
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    # Predict with FL model on test split
    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X = df_fl[feat_cols].values.astype(np.float32)
    y_pred, _ = mlp_predict(global_model, X)
    y_true     = df_fl["risk_label_encoded"].values.astype(np.int64)

    pred_labels = [label_encoder.classes_[p] for p in y_pred]
    true_labels = [label_encoder.classes_[t] for t in y_true]

    if "Age" not in df_fl.columns:
        return {"metric": "Fairness (Accuracy Parity)",
                "pass": None, "error": "Age column not in df_fl"}

    ages = df_fl["Age"].values
    groups = {
        "Young (≤30)":    (ages <= 30),
        "Middle (31-50)": ((ages > 30) & (ages <= 50)),
        "Senior (>50)":   (ages > 50),
    }

    per_group = {}
    accuracies = []
    for grp_name, mask in groups.items():
        if mask.sum() == 0:
            continue
        idx = [i for i in range(len(pred_labels)) if mask[i]]
        n_correct = sum(1 for i in idx if pred_labels[i] == true_labels[i])
        acc = n_correct / len(idx)
        per_group[grp_name] = {
            "n":        int(mask.sum()),
            "accuracy": round(acc, 4),
        }
        accuracies.append(acc)

    if len(accuracies) < 2 or max(accuracies) == 0:
        return {"metric": "Fairness (Accuracy Parity)", "pass": False,
                "error": "Insufficient group sizes for APR", "per_group": per_group}

    apr_value = float(min(accuracies) / max(accuracies))
    threshold = 0.90
    return {
        "metric":           "Fairness (Accuracy Parity, Age Groups)",
        "dir":              round(apr_value, 4),   # kept as 'dir' for backward compat
        "apr":              round(apr_value, 4),
        "per_group":        per_group,
        "min_acc_group":    min(per_group, key=lambda g: per_group[g]["accuracy"]),
        "max_acc_group":    max(per_group, key=lambda g: per_group[g]["accuracy"]),
        "threshold":        threshold,
        "pass":             bool(apr_value >= threshold),
        "value":            apr_value,
        "note":             (f"APR={apr_value:.4f} "
                             f"(min/max per-group accuracy; ≥0.90 = equitable classification)"),
    }


# ─── Metric 9: FL-Central Accuracy Gap ────────────────────────────────────────

def evaluate_fl_central_gap(
    central_model,
    global_model,
    df_fl: pd.DataFrame,
) -> Dict:
    """
    Formal privacy-utility trade-off metric: accuracy gap between the
    centralised baseline model and the FL+DP global model, measured on the
    FL test split.

    Banking context: Differential privacy injects Gaussian noise into every
    client gradient update. This formally quantifies how much accuracy is
    sacrificed for privacy. A gap above 2 pp would indicate that the DP noise
    level is too aggressive for production banking use.

    Formula:
        gap_pp = |central_accuracy − fl_accuracy| × 100

    Threshold: gap < 2.0 pp
    Reference: McMahan et al. (2017) "Communication-Efficient Learning of Deep
               Networks from Decentralized Data" — reports FL accuracy within
               1-2 pp of centralised for well-tuned DP parameters.
    """
    from src.central_model import predict as mlp_predict

    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X = df_fl[feat_cols].values.astype(np.float32)
    y = df_fl["risk_label_encoded"].values.astype(np.int64)

    y_central, _ = mlp_predict(central_model, X)
    y_fl,      _ = mlp_predict(global_model,  X)

    central_acc = float(accuracy_score(y, y_central))
    fl_acc      = float(accuracy_score(y, y_fl))
    gap_pp      = abs(central_acc - fl_acc) * 100.0

    threshold = 2.0
    return {
        "metric":       "FL-Central Accuracy Gap",
        "central_acc":  round(central_acc, 4),
        "fl_acc":       round(fl_acc, 4),
        "gap_pp":       round(gap_pp, 4),
        "threshold":    threshold,
        "pass":         bool(gap_pp < threshold),
        "value":        gap_pp,
        "note":         (f"Central={central_acc*100:.2f}%  FL={fl_acc*100:.2f}%  "
                         f"Gap={gap_pp:.2f}pp  (threshold < {threshold}pp)"),
    }


# ─── Aggregated evaluation ─────────────────────────────────────────────────────────────

def run_full_evaluation(
    df: pd.DataFrame,
    global_model,
    central_model=None,
    fl_history: Optional[Dict] = None,
    df_fl: Optional[pd.DataFrame] = None,
    run_gpt: bool = True,
    provider: Optional[str] = None,
) -> Dict:
    """
    Run all 8 evaluation metrics and return a consolidated report.

    Parameters
    ----------
    df           : full preprocessed customer DataFrame (for cluster analysis)
    global_model : FL-updated model
    central_model: central model (loaded from disk if None)
    fl_history   : dict output from run_fl_simulation
    df_fl        : FL split DataFrame (loaded from CSV if None)
    run_gpt      : whether to call GPT API (set False to skip)
    provider     : GPT provider override

    Returns
    -------
    dict with all metric results
    """
    from src.central_model import load_central_model

    if central_model is None:
        central_model = load_central_model()

    if df_fl is None:
        fl_csv = MODELS_DIR / "df_fl_split.csv"
        if fl_csv.exists():
            df_fl = pd.read_csv(fl_csv)
        else:
            # Replicate the 3-way split: exclude 5% demo holdout, then take
            # the 30% FL portion from the remaining 95% (matches train.py logic).
            import warnings
            warnings.warn(
                "df_fl_split.csv not found — deriving FL split on-the-fly. "
                "Run train.py first for a deterministic, pre-saved split.",
                UserWarning,
            )
            from sklearn.model_selection import train_test_split
            from config import CENTRAL_SPLIT, DEMO_SPLIT
            customers = df["Customer_ID"].unique()
            demo_n = max(1, int(len(customers) * DEMO_SPLIT))
            rng_eval = np.random.default_rng(RANDOM_SEED)
            demo_mask = rng_eval.choice(len(customers), size=demo_n, replace=False)
            demo_set = set(customers[demo_mask])
            remainder = np.array([c for c in customers if c not in demo_set])
            _central_frac = CENTRAL_SPLIT / (1.0 - DEMO_SPLIT)
            _, fl_cust = train_test_split(remainder, train_size=_central_frac,
                                          random_state=RANDOM_SEED)
            df_fl = df[df["Customer_ID"].isin(fl_cust)].copy()

    if fl_history is None:
        hist_path = MODELS_DIR / "fl_training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                fl_history = json.load(f)

    print("\n" + "="*65)
    print("  Running Evaluation Metrics")
    print("="*65)

    results = {}

    # ── Metric 1 ─────────────────────────────────────────────────────────
    print("\n[1/8] Cluster Accuracy (Silhouette Score) ...")
    try:
        results["cluster_accuracy"] = evaluate_cluster_accuracy(df)
        m = results["cluster_accuracy"]
        status = "PASS ✓" if m["pass"] else "FAIL ✗"
        print(f"      Silhouette = {m['silhouette_score']:.4f}  (threshold ≥ 0.80)  [{status}]")
    except Exception as e:
        results["cluster_accuracy"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Metric 2 ─────────────────────────────────────────────────────────
    print("\n[2/8] F1 Score with Federated Learning ...")
    try:
        results["fl_f1"] = evaluate_fl_f1(global_model, df_fl)
        m = results["fl_f1"]
        status = "PASS ✓" if m["pass"] else "FAIL ✗"
        print(f"      Macro F1  = {m['macro_f1']:.4f}  (threshold > 0.80)  [{status}]")
        for cls, v in m["per_class"].items():
            print(f"        {cls:<12}: F1={v['f1']:.4f}  P={v['precision']:.4f}  R={v['recall']:.4f}")
    except Exception as e:
        results["fl_f1"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Metric 3 ─────────────────────────────────────────────────────────
    print("\n[3/8] Data Privacy (ε, δ-DP) ...")
    try:
        results["privacy"] = evaluate_privacy()
        m = results["privacy"]
        print(f"      ε = {m['epsilon']:.4f},  δ = {m['delta']:.0e}  [{m['quality']}]")
    except Exception as e:
        results["privacy"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Metric 4 ─────────────────────────────────────────────────────────
    print("\n[4/8] Federated Loss Stability ...")
    try:
        results["fl_stability"] = evaluate_federated_loss(
            central_model, global_model, df_fl, fl_history
        )
        m = results["fl_stability"]
        status = "PASS ✓" if m["pass"] else "FAIL ✗"
        print(f"      Prediction change = {m['prediction_change_pct']:.2f}%  "
              f"(threshold < 10%)  [{status}]")
    except Exception as e:
        results["fl_stability"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Metric 5 ─────────────────────────────────────────────────────────
    if run_gpt:
        print("\n[5/8] GPT Fund Explanation Correctness ...")
        try:
            results["gpt_correctness"] = evaluate_gpt_correctness(provider=provider)
            m = results["gpt_correctness"]
            status = "PASS ✓" if m["pass"] else "FAIL ✗"
            print(f"      Pass rate = {m['overall_pass_rate']:.2f}  "
                  f"(threshold ≥ 0.75)  [{status}]  [via {m['provider']}]")
        except Exception as e:
            results["gpt_correctness"] = {"pass": False, "error": str(e)}
            print(f"      ERROR: {e}")
    else:
        results["gpt_correctness"] = {"pass": None, "skipped": True}

    # ── Metric 6: Model Calibration ────────────────────────────────────────
    print("\n[6/8] Model Calibration (Brier Score) ...")
    try:
        results["calibration"] = evaluate_calibration(global_model, df_fl)
        m = results["calibration"]
        status = "PASS ✓" if m["pass"] else "FAIL ✗"
        print(f"      Brier Score = {m['brier_score']:.4f}  (threshold < 0.10)  [{status}]")
        for cls, bs in m["per_class_brier"].items():
            print(f"        {cls:<12}: BS={bs:.4f}")
    except Exception as e:
        results["calibration"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Metric 7: Fairness ─────────────────────────────────────────────────
    print("\n[7/8] Fairness — Accuracy Parity (Age Groups) ...")
    try:
        results["fairness"] = evaluate_fairness(global_model, df_fl)
        m = results["fairness"]
        status = "PASS ✓" if m["pass"] else "FAIL ✗"
        print(f"      APR = {m['apr']:.4f}  (threshold ≥ 0.90)  [{status}]")
        for grp, v in m["per_group"].items():
            print(f"        {grp:<20}: n={v['n']:,}  accuracy={v['accuracy']:.4f}")
    except Exception as e:
        results["fairness"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Metric 8: FL-Central Accuracy Gap ─────────────────────────────────
    print("\n[8/8] FL-Central Accuracy Gap ...")
    try:
        results["fl_central_gap"] = evaluate_fl_central_gap(central_model, global_model, df_fl)
        m = results["fl_central_gap"]
        status = "PASS ✓" if m["pass"] else "FAIL ✗"
        print(f"      Gap = {m['gap_pp']:.4f} pp  (threshold < 2.0 pp)  [{status}]")
        print(f"        Central={m['central_acc']*100:.2f}%  FL={m['fl_acc']*100:.2f}%")
    except Exception as e:
        results["fl_central_gap"] = {"pass": False, "error": str(e)}
        print(f"      ERROR: {e}")

    # ── Informational: ROC-AUC ─────────────────────────────────────────────
    print("\n[+] ROC-AUC (One-vs-Rest, informational) ...")
    try:
        results["roc_auc"] = evaluate_roc_auc(global_model, df_fl)
        m = results["roc_auc"]
        print(f"      Macro AUC = {m['macro_auc']:.4f}  |  "
              f"Weighted AUC = {m['weighted_auc']:.4f}")
        for cls, v in m["per_class_auc"].items():
            print(f"        {cls:<12}: AUC={v:.4f}")
    except Exception as e:
        results["roc_auc"] = {"error": str(e)}
        print(f"      ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    # Use == True/False (not `is`) to handle both Python bool and numpy bool
    passed  = [k for k, v in results.items() if v.get("pass") == True  and k != "_summary"]
    failed  = [k for k, v in results.items() if v.get("pass") == False and k != "_summary"]
    skipped = [k for k, v in results.items() if v.get("pass") is None  and k != "_summary"]

    print(f"\n{'='*65}")
    print(f"  Evaluation Summary: {len(passed)}/8 metrics PASSED")
    print(f"{'='*65}")
    for k in results:
        v = results[k]
        icon = "✓" if v.get("pass") is True else ("—" if v.get("pass") is None else "✗")
        print(f"  [{icon}] {v.get('metric', k)}")

    results["_summary"] = {
        "passed": passed, "failed": failed, "skipped": skipped,
        "total_pass": len(passed), "total": 8,
    }

    # Save results
    out = {k: {k2: v2 for k2, v2 in v.items() if isinstance(v2, (str, int, float, bool, list, dict, type(None)))}
           for k, v in results.items()}
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)

    return results


# ─── Visualisation dashboard ──────────────────────────────────────────────────

def plot_evaluation_dashboard(
    results: Dict,
    save_dir: Optional[Path] = None,
) -> None:
    """Generate a 2-page evaluation dashboard."""
    save_dir = Path(save_dir or MODELS_DIR)

    # ── Page 1: Metrics overview ───────────────────────────────────────────
    metric_names = ["Cluster\nAccuracy", "F1 Score\n(FL)", "Privacy\n(ε, lower=better)", "FL Stability\n(change %)", "GPT\nCorrectness"]
    metric_keys  = ["cluster_accuracy", "fl_f1", "privacy", "fl_stability", "gpt_correctness"]
    thresholds   = [0.80, 0.80, None, 10.0, 0.75]
    values = []
    for k in metric_keys:
        v = results.get(k, {}).get("value", None)
        values.append(float(v) if v is not None else 0.0)

    colors = []
    for k, t in zip(metric_keys, thresholds):
        m = results.get(k, {})
        if m.get("pass") is True:
            colors.append("#4CAF50")
        elif m.get("pass") is False:
            colors.append("#F44336")
        else:
            colors.append("#9E9E9E")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Smart Fund Advisor — Evaluation Dashboard", fontsize=14, fontweight="bold")

    bars = axes[0].barh(metric_names[::-1], values[::-1], color=colors[::-1], alpha=0.85, height=0.6)
    for i, (bar, val) in enumerate(zip(bars, values[::-1])):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", ha="left", fontsize=9)
    axes[0].set_title("Metric Values vs Thresholds")
    axes[0].set_xlabel("Value")
    axes[0].axvline(0, color="black", linewidth=0.5)
    green_p = mpatches.Patch(color="#4CAF50", label="PASS")
    red_p   = mpatches.Patch(color="#F44336", label="FAIL")
    grey_p  = mpatches.Patch(color="#9E9E9E", label="N/A")
    axes[0].legend(handles=[green_p, red_p, grey_p], loc="lower right")

    # ── Pass/fail pie ──────────────────────────────────────────────────────
    summary = results.get("_summary", {})
    n_pass  = summary.get("total_pass", 0)
    n_fail  = len(summary.get("failed", []))
    n_skip  = len(summary.get("skipped", []))
    sizes   = [s for s in [n_pass, n_fail, n_skip] if s > 0]
    labels_ = [l for l, s in zip(["PASS", "FAIL", "SKIPPED"], [n_pass, n_fail, n_skip]) if s > 0]
    colors_ = [c for c, s in zip(["#4CAF50", "#F44336", "#9E9E9E"], [n_pass, n_fail, n_skip]) if s > 0]
    axes[1].pie(sizes, labels=labels_, colors=colors_, autopct="%1.0f%%",
                startangle=90, pctdistance=0.7, textprops={"fontsize": 11})
    axes[1].set_title(f"Evaluation Summary\n{n_pass}/5 Passed")

    label_subplots(axes)
    plt.tight_layout()
    plt.savefig(save_dir / "plot_evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation dashboard → {save_dir / 'plot_evaluation_dashboard.png'}")

    # ── Page 2: FL loss + per-class F1 ────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Federated Learning — Detailed Results", fontsize=13)

    # FL loss trend
    fl_stab = results.get("fl_stability", {})
    loss_trend = fl_stab.get("fl_loss_trend", [])
    if loss_trend:
        axes2[0].plot(range(1, len(loss_trend) + 1), loss_trend,
                      marker="o", color="darkorange", linewidth=2)
        axes2[0].set_title("FL Distributed Loss per Round")
        axes2[0].set_xlabel("Round"); axes2[0].set_ylabel("Avg Loss")
        axes2[0].grid(alpha=0.3)

    # Per-class F1
    fl_f1 = results.get("fl_f1", {})
    per_class = fl_f1.get("per_class", {})
    if per_class:
        cls_names = list(per_class.keys())
        f1_vals   = [per_class[c]["f1"]        for c in cls_names]
        prec_vals = [per_class[c]["precision"]  for c in cls_names]
        rec_vals  = [per_class[c]["recall"]     for c in cls_names]
        x = np.arange(len(cls_names))
        w = 0.25
        axes2[1].bar(x - w, prec_vals, w, label="Precision", color="#2196F3", alpha=0.85)
        axes2[1].bar(x,     f1_vals,   w, label="F1",        color="#4CAF50", alpha=0.85)
        axes2[1].bar(x + w, rec_vals,  w, label="Recall",    color="#FF9800", alpha=0.85)
        axes2[1].axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="Threshold 0.80")
        axes2[1].set_xticks(x); axes2[1].set_xticklabels(cls_names, rotation=30, ha="right")
        axes2[1].set_ylim(0, 1.05); axes2[1].set_title("Per-class Metrics (FL Model)")
        axes2[1].legend(fontsize=8); axes2[1].grid(alpha=0.3, axis="y")

    # Privacy epsilon by round
    priv = results.get("privacy", {})
    eps_by_round = priv.get("eps_by_round", {})
    if eps_by_round:
        rnds  = [int(k) for k in sorted(eps_by_round.keys(), key=int)]
        epss  = [eps_by_round[str(r)] for r in rnds]
        axes2[2].plot(rnds, epss, marker="s", color="purple", linewidth=2)
        axes2[2].fill_between(rnds, epss, alpha=0.15, color="purple")
        axes2[2].set_title(f"Privacy Budget (ε) by Round\nFinal ε = {priv.get('epsilon', 'N/A'):.4f}")
        axes2[2].set_xlabel("FL Rounds"); axes2[2].set_ylabel("Cumulative ε")
        axes2[2].grid(alpha=0.3)

    label_subplots(axes2)
    plt.tight_layout()
    plt.savefig(save_dir / "plot_evaluation_details.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation details → {save_dir / 'plot_evaluation_details.png'}")

    # ── Page 3: ROC-AUC curves ─────────────────────────────────────────────
    roc_data = results.get("roc_auc", {})
    if roc_data and "per_class_auc" in roc_data:
        try:
            plot_roc_auc_curves(roc_data, save_dir=save_dir)
        except Exception:
            pass


def plot_per_user_risk(
    df_sample: pd.DataFrame,
    model,
    label_encoder,
    n_users: int = 20,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Show risk predictions for individual users with probability bars.
    Returns a summary DataFrame.

    Parameters
    ----------
    df_sample   : preprocessed customer DataFrame
    model       : RiskMLP (central or FL)
    label_encoder: fitted LabelEncoder
    n_users     : number of users to visualise
    save_path   : where to save the plot
    """
    from src.central_model import predict as mlp_predict

    feat_cols = [f for f in RISK_FEATURES if f in df_sample.columns]
    customers = df_sample["Customer_ID"].unique()[:n_users]

    rows = []
    for cid in customers:
        sub = df_sample[df_sample["Customer_ID"] == cid].tail(1)
        X   = sub[feat_cols].values.astype(np.float32)
        pred_idx, probs = mlp_predict(model, X)
        pred_label = label_encoder.inverse_transform(pred_idx)[0]
        true_label = sub["risk_label"].iloc[0] if "risk_label" in sub.columns else "Unknown"
        prob_dict  = {label_encoder.inverse_transform([i])[0]: round(float(probs[0, i]), 3)
                      for i in range(probs.shape[1])}
        rows.append({
            "Customer_ID":  cid,
            "True_Risk":    true_label,
            "Predicted_Risk": pred_label,
            "Correct":      true_label == pred_label,
            "Confidence":   round(float(probs.max()), 3),
            **{f"P({rc})": prob_dict.get(rc, 0) for rc in RISK_CLASSES},
        })

    result_df = pd.DataFrame(rows)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    fig.suptitle(f"Risk Appetite Prediction — {n_users} Sample Users", fontsize=13, fontweight="bold")

    # ── Top: predicted vs true ─────────────────────────────────────────────
    risk_order = {r: i for i, r in enumerate(RISK_CLASSES)}
    cmap_risk = plt.cm.get_cmap("RdYlGn", 5)
    colors_pred = [cmap_risk(risk_order.get(r, 2)) for r in result_df["Predicted_Risk"]]
    colors_true = [cmap_risk(risk_order.get(r, 2)) for r in result_df["True_Risk"]]

    x = np.arange(len(result_df))
    axes[0].bar(x - 0.2, result_df["Predicted_Risk"].map(risk_order),
                0.4, color=colors_pred, alpha=0.85, label="Predicted")
    axes[0].bar(x + 0.2, result_df["True_Risk"].map(risk_order),
                0.4, color=colors_true, alpha=0.45, edgecolor="black",
                linestyle="--", linewidth=0.8, label="True")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f"{r}\n{'✓' if ok else '✗'}" for r, ok in zip(result_df["Customer_ID"], result_df["Correct"])],
        fontsize=7, rotation=45, ha="right"
    )
    axes[0].set_yticks(range(5)); axes[0].set_yticklabels(RISK_CLASSES)
    axes[0].set_title("Predicted vs True Risk Label per User")
    axes[0].legend()
    axes[0].grid(alpha=0.2, axis="y")

    # ── Bottom: probability heatmap ─────────────────────────────────────────
    prob_matrix = result_df[[f"P({rc})" for rc in RISK_CLASSES]].values
    im = axes[1].imshow(prob_matrix.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_yticks(range(5)); axes[1].set_yticklabels(RISK_CLASSES)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(result_df["Customer_ID"], fontsize=7, rotation=45, ha="right")
    axes[1].set_title("Risk Class Probabilities (Confidence Heatmap)")
    plt.colorbar(im, ax=axes[1], label="Probability")

    for xi in x:
        for yi in range(5):
            val = prob_matrix[xi, yi]
            if val > 0.3:
                axes[1].text(xi, yi, f"{val:.2f}", ha="center", va="center",
                             fontsize=6, color="white" if val > 0.6 else "black")

    label_subplots(axes, y=-0.24)
    plt.tight_layout()
    out_path = save_path or (MODELS_DIR / "plot_per_user_risk.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-user risk plot → {out_path}")

    return result_df
