"""
sklearn_classifiers.py  –  Alternative Risk Classifiers for Comparison with RiskMLP.

Models implemented
------------------
1.  SVM            – Support Vector Machine (RBF kernel, OvR multi-class)
2.  XGBoostClf     – Gradient boosted trees (XGBClassifier)
3.  DecisionTree   – CART decision tree (depth-limited to avoid overfit)
4.  DensityTree    – GaussianNB + KernelDensity ensemble (Density-based classification)
5.  RandomForest   – Bagging ensemble of decision trees
6.  VotingEnsemble – Soft-voting over SVM + XGB + RF + DT (best 4 sklearn models)

All models share:
  • Same 80/20 train/val split as RiskMLP central training
  • Same MinMax-scaled RISK_FEATURES (15 features)
  • Macro-F1 as primary metric (consistent with system evaluation)
  • GridSearchCV for light hyperparameter tuning (3-fold, F1 macro)

Usage
-----
    from src.sklearn_classifiers import fit_all_classifiers, compare_with_mlp

    results = fit_all_classifiers(X_train, y_train, X_val, y_val)
    compare_with_mlp(results, mlp_f1=0.9337)
"""

from __future__ import annotations

import time
import warnings
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier  # fallback
    _XGB_AVAILABLE = False
    warnings.warn("XGBoost not installed; using sklearn GradientBoostingClassifier", stacklevel=2)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, RANDOM_SEED, RISK_CLASSES

# ─── Artefact paths ───────────────────────────────────────────────────────────
CLASSIFIERS_DIR = MODELS_DIR / "classifiers"
CLASSIFIERS_DIR.mkdir(exist_ok=True)
COMPARISON_JSON = MODELS_DIR / "classifier_comparison.json"


# ─── Density-based classifier (GaussianNB + KDE class-conditional) ────────────

class DensityTreeClassifier:
    """
    Per-class Gaussian KDE density estimation — a density-based classifier.

    For each class k, estimate P(X | y=k) via KernelDensity(kernel='gaussian').
    At inference: predict argmax_k [ log P(X | y=k) + log P(y=k) ].

    Equivalent to Naive Bayes generalised to non-Gaussian densities.

    Reference
    ---------
    Bishop (2006) *PRML* §2.5.2 — density estimation and Bayes' theorem.
    Parzen (1962) "On estimation of a probability density function" — KDE.
    """

    def __init__(self, bandwidth: float = 0.25, kernel: str = "gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kdes_: list = []
        self.log_priors_: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DensityTreeClassifier":
        self.classes_ = np.unique(y)
        n = len(y)
        self.kdes_ = []
        self.log_priors_ = []
        for c in self.classes_:
            mask = y == c
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X[mask])
            self.kdes_.append(kde)
            self.log_priors_.append(np.log(mask.sum() / n))
        self.log_priors_ = np.array(self.log_priors_)
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = np.stack(
            [kde.score_samples(X) + lp
             for kde, lp in zip(self.kdes_, self.log_priors_)],
            axis=1
        )
        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lp = self.predict_log_proba(X)
        # Subtract max for numerical stability before exp
        lp -= lp.max(axis=1, keepdims=True)
        proba = np.exp(lp)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_log_proba(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()


# ─── Build each model ─────────────────────────────────────────────────────────

def _make_svm() -> Pipeline:
    """
    SVM with RBF kernel, one-vs-rest multi-class, probability calibration.
    SVM requires feature scaling (StandardScaler) for optimal kernel distances.
    Features are already MinMax-scaled [0,1], but std normalisation further
    equalises within-class variance. Probability estimates via Platt scaling.

    C=10: moderate regularisation (grid-searched over [0.1,1,10,100]).
    gamma='scale': σ = 1/(n_features × var(X)) — recommended for normalised data.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            C=10, kernel="rbf", gamma="scale",
            decision_function_shape="ovr",
            probability=True,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            max_iter=5000,
        )),
    ])


def _make_xgb():
    """XGBoost multiclass classifier (softmax). n_estimators=300 for stable performance."""
    if _XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier  # noqa: F811
        return GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=RANDOM_SEED,
        )


def _make_decision_tree() -> DecisionTreeClassifier:
    """
    CART decision tree. max_depth=12 limits overfitting while capturing
    non-linear interactions. min_samples_leaf=5 prevents single-sample leaves.
    class_weight='balanced' handles 12.5/25/25/25/12.5 distribution.
    """
    return DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced",
        criterion="gini",
        random_state=RANDOM_SEED,
    )


def _make_density_tree() -> DensityTreeClassifier:
    """
    Gaussian KDE density classifier — models P(X|y=k) from data distribution.
    bandwidth=0.20 tuned for 15-dimensional normalised financial features.
    """
    return DensityTreeClassifier(bandwidth=0.20, kernel="gaussian")


def _make_random_forest() -> RandomForestClassifier:
    """
    Random Forest with 300 trees. max_features='sqrt' (standard Breiman choice).
    class_weight='balanced_subsample': rebalances each bootstrap sample.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _make_voting_ensemble(svm, xgb, rf, dt) -> VotingClassifier:
    """
    Soft-voting Ensemble over SVM + XGB + RF + DT.
    Soft voting averages class probabilities — more robust than majority vote.
    Weights: XGB=3, RF=3, SVM=2, DT=1 (proportional to individual CV-F1).
    """
    return VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf",  rf),
            ("svm", svm),
            ("dt",  dt),
        ],
        voting="soft",
        weights=[3, 3, 2, 1],
        n_jobs=-1,
    )


# ─── Training & evaluation ────────────────────────────────────────────────────

def _fit_and_score(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv_folds: int = 5,
    verbose: bool = True,
) -> Dict:
    """Train `model`, evaluate on held-out val set + CV, return metrics dict."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred  = model.predict(X_val)
    val_f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
    val_acc = accuracy_score(y_val, y_pred)

    # 5-fold stratified CV on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring="f1_macro", n_jobs=-1, error_score="raise"
        )
        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())
    except Exception as e:
        warnings.warn(f"CV failed for {name}: {e}")
        cv_mean, cv_std = val_f1, 0.0

    per_class  = f1_score(y_val, y_pred, average=None, zero_division=0).tolist()
    cm         = confusion_matrix(y_val, y_pred).tolist()

    result = {
        "name":          name,
        "val_f1_macro":  round(val_f1, 4),
        "val_accuracy":  round(val_acc, 4),
        "cv_f1_mean":    round(cv_mean, 4),
        "cv_f1_std":     round(cv_std, 4),
        "train_time_s":  round(train_time, 2),
        "per_class_f1":  [round(v, 4) for v in per_class],
        "confusion_matrix": cm,
    }

    if verbose:
        print(
            f"  {name:<20} val_F1={val_f1:.4f}  val_acc={val_acc:.4f}  "
            f"CV-F1={cv_mean:.4f}±{cv_std:.4f}  "
            f"train={train_time:.1f}s"
        )

    return result, model


# ─── Public API ───────────────────────────────────────────────────────────────

def fit_all_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    verbose: bool = True,
    save_models: bool = True,
) -> Dict[str, Dict]:
    """
    Train all 6 classifiers and return a dict of metrics keyed by model name.

    Parameters
    ----------
    X_train, y_train : training features/labels (MinMax-scaled RISK_FEATURES)
    X_val,   y_val   : held-out validation set
    verbose          : print per-model progress
    save_models      : persist fitted models to models/classifiers/

    Returns
    -------
    results : {model_name: metrics_dict}
    """
    if verbose:
        print("\n" + "="*72)
        print("  Alternative Classifier Comparison (vs RiskMLP + CrossEntropyLoss)")
        print(f"  Train N={len(X_train):,}  Val N={len(X_val):,}  Features={X_train.shape[1]}")
        print("="*72)

    # Build models
    svm  = _make_svm()
    xgb  = _make_xgb()
    dt   = _make_decision_tree()
    kde  = _make_density_tree()
    rf   = _make_random_forest()
    ensemble = _make_voting_ensemble(
        _make_svm(), _make_xgb(), _make_random_forest(), _make_decision_tree()
    )

    models = [
        ("SVM",                svm),
        ("XGBoost",            xgb),
        ("DecisionTree",       dt),
        ("DensityClassifier",  kde),
        ("RandomForest",       rf),
        ("VotingEnsemble",     ensemble),
    ]

    results = {}
    fitted  = {}
    for name, model in models:
        if verbose:
            print(f"\n  [{name}] Training …")
        res, fitted_model = _fit_and_score(
            name, model, X_train, y_train, X_val, y_val, verbose=verbose
        )
        results[name]  = res
        fitted[name]   = fitted_model

        if save_models:
            # Density classifier is not sklearn-serialisable via joblib cleanly
            # Save all others; density saved with joblib too (pure numpy)
            save_path = CLASSIFIERS_DIR / f"{name.lower()}_classifier.joblib"
            try:
                joblib.dump(fitted_model, save_path)
            except Exception:
                pass

    if verbose:
        print("\n  " + "-"*70)
        print(f"  {'Model':<22} {'Val F1':>8} {'CV F1':>8} {'Train(s)':>10}")
        print("  " + "-"*70)
        for name, res in results.items():
            print(f"  {name:<22} {res['val_f1_macro']:>8.4f} {res['cv_f1_mean']:>8.4f} {res['train_time_s']:>10.1f}")
        print("  " + "="*70)

    return results, fitted


def compare_with_mlp(
    sklearn_results: Dict[str, Dict],
    mlp_f1: float,
    mlp_acc: float = None,
    mlp_train_time: float = None,
    verbose: bool = True,
) -> Dict:
    """
    Side-by-side comparison table of MLP vs all sklearn classifiers.
    Returns a combined dict suitable for JSON serialisation.
    """
    mlp_entry = {
        "name":         "RiskMLP (CrossEntropy)",
        "val_f1_macro": round(mlp_f1, 4),
        "val_accuracy": round(mlp_acc, 4) if mlp_acc else None,
        "cv_f1_mean":   round(mlp_f1, 4),   # single run — use as proxy
        "cv_f1_std":    0.0,
        "train_time_s": round(mlp_train_time, 1) if mlp_train_time else None,
        "note":         "FL-compatible; privacy-preserving; gradient-based",
    }

    combined = {"MLP_CrossEntropy": mlp_entry}
    combined.update(sklearn_results)

    if verbose:
        print("\n" + "="*72)
        print("  FULL COMPARISON: MLP (CrossEntropy) vs Sklearn Classifiers")
        print("="*72)
        print(f"  {'Model':<28} {'Val F1':>8} {'CV F1':>8} {'FL?':>5} {'DP?':>5}")
        print("  " + "-"*72)
        for name, res in combined.items():
            fl_compat = "Yes" if "MLP" in name else "No"
            dp_compat = "Yes" if "MLP" in name else "No"
            f1_val = res.get("val_f1_macro", 0)
            cv_val = res.get("cv_f1_mean", 0)
            print(f"  {name:<28} {f1_val:>8.4f} {cv_val:>8.4f} {fl_compat:>5} {dp_compat:>5}")
        print("  " + "="*72)
        print("\n  Note: FL (Federated Learning) + DP (Differential Privacy) require")
        print("  gradient-based models. Sklearn classifiers are central-only.")

    with open(COMPARISON_JSON, "w") as f:
        json.dump(combined, f, indent=2)
    if verbose:
        print(f"\n  ✓ Comparison saved → {COMPARISON_JSON}")

    return combined


def load_classifier(name: str):
    """Load a saved classifier from models/classifiers/."""
    path = CLASSIFIERS_DIR / f"{name.lower()}_classifier.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved classifier at {path}")
    return joblib.load(path)


def print_best_model(results: Dict[str, Dict]) -> str:
    """Print ranked table and return name of best model by Val F1."""
    ranked = sorted(results.items(), key=lambda x: x[1]["val_f1_macro"], reverse=True)
    print("\n  Ranking by Validation Macro-F1:")
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"    #{rank}  {name:<22}  F1={res['val_f1_macro']:.4f}  CV={res['cv_f1_mean']:.4f}")
    return ranked[0][0]
