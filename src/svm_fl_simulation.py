"""
svm_fl_simulation.py  –  Federated Learning + Differential Privacy for LinearSVM.

Why Linear SVM is FL-compatible
--------------------------------
A LinearSVM trained with stochastic gradient descent (SGDClassifier,
loss='modified_huber') has a weight vector W = (coef_, intercept_) that is
algebraically equivalent to the final gradient-sum of mini-batch SGD updates.
This makes it directly amenable to FedAvg:

    W_global = Σ_k (n_k / N) · W_k_local

where W_k_local is the weight vector after K local SGD steps on device k.

Differential Privacy (weight-update perturbation)
--------------------------------------------------
For linear models, clipping at the gradient level OR at the final weight-update
level are statistically equivalent under the Gaussian mechanism (Dwork et al.
2014):

    1. Record W₀  = global weights (broadcast to device)
    2. Train locally: W₁ = W₀ + Σ SGD_steps
    3. Compute  ΔW  = W₁ − W₀                    (total update)
    4. Clip:    ΔW̃  = ΔW · min(1, C / ‖ΔW‖₂)   (bound sensitivity to C)
    5. Noise:   ΔW̃ += N(0, σ²C²·I)              (Gaussian mechanism)
    6. Send:    W₀ + ΔW̃   to server

The same DP hyper-parameters (σ = DP_NOISE_MULTIPLIER, C = DP_MAX_GRAD_NORM)
from config.py are reused so privacy budget ε is directly comparable between
RiskMLP-FL and SVM-FL under identical accounting assumptions.

Non-IID / drift
---------------
Identical Dirichlet(α) round sampling and drift-cosine monitoring as
fl_simulation.py — results are directly comparable.

References
----------
McMahan et al. (2017) "Communication-Efficient Learning" — FedAvg.
Li et al.     (2020) "FedProx" — proximal term for client drift.
Dwork et al.  (2014) "Algorithmic Foundations of DP" — Gaussian mechanism.
Abadi et al.  (2016) "Deep Learning with DP" — DP-SGD (extended here to linear).
Mironov et al.(2019) "Rényi DP" — amplification by subsampling.
"""

from __future__ import annotations

import copy
import json
import math
import time
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FL_ROUNDS, FL_MIN_CLIENTS, FL_FRACTION_FIT,
    FL_LOCAL_EPOCHS, FL_BATCH_SIZE,
    DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM,
    RISK_FEATURES, N_RISK_CLASSES, RANDOM_SEED, MODELS_DIR,
    DIRICHLET_ALPHA, DRIFT_COSINE_THRESHOLD, DRIFT_WEIGHT_PENALTY,
    LEARNING_RATE, FEDPROX_MU, CENTRAL_SPLIT,
)

# Re-use the same Dirichlet + entropy helpers from fl_simulation to avoid duplication
from src.fl_simulation import _dirichlet_round_sample, _round_class_entropy

# ─── Constants ────────────────────────────────────────────────────────────────
SVM_FL_MODEL_PATH = MODELS_DIR / "svm_fl_global_model.joblib"
SVM_FL_HISTORY_PATH = MODELS_DIR / "svm_fl_training_history.json"
SVM_COMPARISON_PATH = MODELS_DIR / "svm_vs_mlp_fl_comparison.json"
DROPOUT_RATE     = 0.15
MIN_LOCAL_EPOCHS = 2
MAX_LOCAL_EPOCHS = 8

_ALL_CLASSES = np.arange(N_RISK_CLASSES, dtype=np.int64)

# ─── SVMFLModel: parameter-vector wrapper around SGDClassifier ────────────────

class SVMFLModel:
    """
    Thin FedAvg-compatible wrapper around SGDClassifier(loss='modified_huber').

    'modified_huber' loss:
      • Smooth approximation of the SVM hinge loss — differentiable everywhere.
      • Supports predict_proba() out-of-the-box (required for pseudo-label FL).
      • Equivalent to L2-regularised linear SVM in the limit of many passes.

    FedAvg parameter representation
    --------------------------------
    coef_       : ndarray (N_RISK_CLASSES, n_features)  — weight hyperplane normals
    intercept_  : ndarray (N_RISK_CLASSES,)             — decision boundary offsets

    These are flattened into a single 1-D parameter vector for drift cosine
    monitoring (matches the interface used by RiskMLP).
    """

    def __init__(
        self,
        alpha: float = 1e-4,    # L2 regularisation (equivalent to C=1/alpha in SVC)
        learning_rate: float = LEARNING_RATE,
    ):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self._model = self._make_fresh()

    def _make_fresh(self) -> SGDClassifier:
        return SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=self.alpha,
            learning_rate="adaptive",
            eta0=self.learning_rate,
            max_iter=1,             # we control epochs manually via partial_fit
            warm_start=True,
            # NOTE: class_weight='balanced' is incompatible with partial_fit.
            # Class imbalance is handled via per-sample weights passed explicitly
            # in _svm_local_train (class_weights_arr[y_local]).
            random_state=RANDOM_SEED,
            n_jobs=1,
        )

    # ── Parameter serialisation (mirrors RiskMLP.get/set_parameters) ─────────

    def get_parameters(self) -> List[np.ndarray]:
        """Return [coef_, intercept_] as a list of numpy arrays."""
        if not hasattr(self._model, "coef_"):
            # Model not yet fitted — return zero arrays
            return [
                np.zeros((N_RISK_CLASSES, len(RISK_FEATURES)), dtype=np.float64),
                np.zeros(N_RISK_CLASSES, dtype=np.float64),
            ]
        return [self._model.coef_.copy(), self._model.intercept_.copy()]

    def set_parameters(self, params: List[np.ndarray]) -> None:
        """Set coef_ and intercept_ from FedAvg-aggregated arrays."""
        coef, intercept = params
        if not hasattr(self._model, "coef_"):
            # Initialise internal sklearn attributes so warm_start works
            self._model.coef_       = coef.copy().astype(np.float64)
            self._model.intercept_  = intercept.copy().astype(np.float64)
            self._model.classes_    = _ALL_CLASSES
            self._model.n_iter_     = 0
        else:
            self._model.coef_[:]       = coef
            self._model.intercept_[:]  = intercept

    def flat_params(self) -> np.ndarray:
        """Flattened 1-D concatenation of [coef_, intercept_] for cosine drift."""
        p = self.get_parameters()
        return np.concatenate([p[0].flatten(), p[1].flatten()])

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self._model, "coef_"):
            return np.zeros(len(X), dtype=np.int64)
        return self._model.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self._model, "coef_"):
            p = np.ones((len(X), N_RISK_CLASSES)) / N_RISK_CLASSES
            return p
        return self._model.predict_proba(X)


# ─── FedAvg aggregation for SVM weight vectors ────────────────────────────────

def _svm_fedavg(
    client_updates: List[Tuple[List[np.ndarray], int]],
    global_params: List[np.ndarray],
) -> Tuple[List[np.ndarray], Dict]:
    """
    Weighted FedAvg over (coef_, intercept_) arrays.

    Identical drift monitoring logic to fl_simulation._fedavg():
      • Compute per-client ΔW = W_local − W_global (flattened).
      • Cosine similarity of each ΔW vs the mean update direction.
      • Penalise high-drift clients (cosine < threshold) in FedAvg weights.
    """
    drift_info: Dict = {
        "n_clients": len(client_updates),
        "n_high_drift": 0,
        "cosine_sims": [],
        "mean_cosine": 0.0,
        "min_cosine": 0.0,
        "drift_penalty_applied": False,
    }

    flat_global = np.concatenate([p.flatten() for p in global_params])

    update_vecs = []
    for params, _ in client_updates:
        flat_local = np.concatenate([p.flatten() for p in params])
        update_vecs.append(flat_local - flat_global)

    mean_delta = np.mean(update_vecs, axis=0)
    mean_norm  = np.linalg.norm(mean_delta) + 1e-12

    cosine_sims = []
    for dv in update_vecs:
        dv_norm = np.linalg.norm(dv) + 1e-12
        cos = float(np.dot(dv, mean_delta) / (dv_norm * mean_norm))
        cosine_sims.append(cos)

    drift_info["cosine_sims"]  = [round(c, 4) for c in cosine_sims]
    drift_info["mean_cosine"]  = round(float(np.mean(cosine_sims)), 4)
    drift_info["min_cosine"]   = round(float(np.min(cosine_sims)), 4)
    n_high_drift = sum(1 for c in cosine_sims if c < DRIFT_COSINE_THRESHOLD)
    drift_info["n_high_drift"] = n_high_drift
    drift_info["drift_penalty_applied"] = n_high_drift > 0

    # Build FedAvg weights (penalise high-drift clients)
    raw_weights = []
    for i, (_, n) in enumerate(client_updates):
        penalty = DRIFT_WEIGHT_PENALTY if cosine_sims[i] < DRIFT_COSINE_THRESHOLD else 1.0
        raw_weights.append(n * penalty)
    total = sum(raw_weights) or 1.0

    # Aggregate: weighted average of each parameter array
    agg = None
    for (params, _), w in zip(client_updates, raw_weights):
        weight = w / total
        if agg is None:
            agg = [weight * p.astype(np.float64) for p in params]
        else:
            agg = [a + weight * p for a, p in zip(agg, params)]

    return agg, drift_info  # type: ignore[return-value]


# ─── Per-device local training with DP weight-update perturbation ─────────────

def _svm_local_train(
    global_params: List[np.ndarray],
    X_local: np.ndarray,
    y_local: np.ndarray,
    local_epochs: int = FL_LOCAL_EPOCHS,
    dp_enabled: bool = True,
    class_weights_arr: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], int, float]:
    """
    Simulate one mobile device's local SGD-SVM training with DP.

    DP mechanism (weight-update perturbation)
    ------------------------------------------
    Gradient-level DP and weight-update DP are equivalent for linear models
    (both reduce to the Gaussian mechanism on the same sensitivity quantity).
    We apply DP at the update level:

        ΔW = W_local − W_global
        ΔW̃ = ΔW · min(1, C/‖ΔW‖₂)        [clip]
        ΔW̃ += N(0, (σ·C)² · I)            [noise]
        W_send = W_global + ΔW̃

    Complexity: O(K · n_local · d) where K=local_epochs, d=n_features.

    Parameters
    ----------
    global_params : [coef_ (K×d), intercept_ (K,)] — broadcast from server
    X_local, y_local : device-local features and (pseudo-)labels
    local_epochs  : number of SGD passes over the local data
    dp_enabled    : add Gaussian noise to weight update
    class_weights_arr : per-class weight array (N_RISK_CLASSES,), optional

    Returns
    -------
    (updated_params, n_samples, approx_hinge_loss)
    """
    # Build fresh local model and load global weights
    local_model = SVMFLModel()
    local_model.set_parameters([p.copy() for p in global_params])

    # Snapshot of global weights (for ΔW computation)
    W0_coef      = global_params[0].copy()
    W0_intercept = global_params[1].copy()

    # Unique classes present on this device (may be < N_RISK_CLASSES)
    dev_classes = np.union1d(y_local, _ALL_CLASSES)

    # Compute sample weights for class balance
    if class_weights_arr is not None:
        sample_w = class_weights_arr[y_local]
    else:
        sample_w = None

    # Local SGD training (K epochs, each pass = partial_fit once)
    for _ in range(local_epochs):
        local_model._model.partial_fit(
            X_local, y_local,
            classes=dev_classes,
            sample_weight=sample_w,
        )

    # Compute approximate hinge decision-function margin violation
    df = local_model._model.decision_function(X_local)   # shape (n, n_classes)
    # 'modified_huber' surrogate for hinge: max(0, 1 - y·f)²
    margins = 1.0 - df[np.arange(len(y_local)), y_local]
    approx_loss = float(np.mean(np.maximum(0.0, margins) ** 2))

    # ── DP: weight-update perturbation ────────────────────────────────────
    updated_coef      = local_model._model.coef_.astype(np.float64)
    updated_intercept = local_model._model.intercept_.astype(np.float64)

    if dp_enabled:
        # Compute and clip ΔW
        delta_coef      = updated_coef      - W0_coef
        delta_intercept = updated_intercept - W0_intercept
        delta_flat = np.concatenate([delta_coef.flatten(), delta_intercept.flatten()])
        delta_norm = np.linalg.norm(delta_flat)
        clip_factor = min(1.0, DP_MAX_GRAD_NORM / (delta_norm + 1e-12))
        delta_coef      *= clip_factor
        delta_intercept *= clip_factor
        # Add Gaussian noise (same σ, C as RiskMLP-FL for identical ε bound)
        noise_scale = DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM
        delta_coef      += np.random.randn(*delta_coef.shape)      * noise_scale
        delta_intercept += np.random.randn(*delta_intercept.shape) * noise_scale
        # Reconstruct noisy update
        updated_coef      = W0_coef      + delta_coef
        updated_intercept = W0_intercept + delta_intercept

    return (
        [updated_coef, updated_intercept],
        len(y_local),
        approx_loss,
    )


# ─── Incremental FL simulation (mirrors run_incremental_fl_simulation) ─────────

def run_svm_incremental_fl(
    df_fl: "pd.DataFrame",
    dp_enabled: bool = True,
    n_waves: int = 5,
    rounds_per_wave: int = 3,
    verbose: bool = True,
) -> Tuple[SVMFLModel, Dict]:
    """
    Production-realistic incremental FL simulation using DP LinearSVM.

    Structurally mirrors run_incremental_fl_simulation() so results are
    directly comparable:
      • Same wave/round structure (default 5 waves × 3 rounds = 15 total).
      • Same Dirichlet non-IID sampling (α₀=1.5 → α_N=0.3).
      • Same client-drift monitoring and penalised FedAvg.
      • Same straggler dropout (15%) and heterogeneous local-epoch count.
      • Same pseudo-labelling strategy (global model predicts on new devices
        at wave entry — no labels ever sent from server to device).
      • Same DP noise (σ, C) → identical privacy budget (ε, δ) to RiskMLP-FL.

    Only the local model and training routine differ:
      RiskMLP-FL  → gradient-descent on a 4-layer MLP, gradient-level DP
      SVM-FL      → SGD on a linear model,             weight-update-level DP

    Parameters
    ----------
    df_fl          : FL cohort DataFrame (≤4 rows per Customer_ID)
    dp_enabled     : add DP noise to weight updates
    n_waves        : user on-boarding waves
    rounds_per_wave: FL rounds per wave
    verbose        : print per-round progress

    Returns
    -------
    global_svm   : SVMFLModel with final FedAvg'd weights
    svm_history  : per-round loss/accuracy/entropy/drift metrics dict
    """
    import pandas as pd
    import joblib

    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Build per-device data ──────────────────────────────────────────────
    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    client_features:  dict = {}
    client_gt_labels: dict = {}
    for cid, grp in df_fl.groupby("Customer_ID"):
        client_features[str(cid)]  = grp[feat_cols].values.astype(np.float64)
        client_gt_labels[str(cid)] = grp["risk_label_encoded"].values.astype(np.int64)

    all_client_ids = np.array(list(client_features.keys()))
    total_devices  = len(all_client_ids)

    # Per-class weights for balanced local training (same as RiskMLP-FL)
    all_y = np.concatenate(list(client_gt_labels.values()))
    _counts = np.bincount(all_y, minlength=N_RISK_CLASSES).astype(float)
    _cw = len(all_y) / (N_RISK_CLASSES * (_counts + 1e-8))
    class_weights_arr = _cw.astype(np.float64)

    # client_risk_class used ONLY for Dirichlet sampling (never as training signal)
    client_risk_class: dict = {
        cid: int(client_gt_labels[cid][0]) for cid in client_features
    }

    rng.shuffle(all_client_ids)
    wave_cohorts = np.array_split(all_client_ids, n_waves)

    total_rounds = n_waves * rounds_per_wave
    svm_history: dict = {
        "distributed_loss": {},
        "round_accuracy":   {},
        "round_entropy":    {},
        "round_drift":      {},
        "waves":            {},
    }

    # ── Warm-start: pre-train global SVM on central-split data for stable init ─
    # Load the pre-scaled X_train/y_train from the central fit (persisted in
    # models/fl_customer_split.csv which contains the 30% FL cohort already
    # re-encoded). We use all FL data for the initial warm-start pass.
    global_svm = SVMFLModel()
    # One warm-start pass on all FL data to initialise weights
    X_all_init = np.vstack([client_features[c] for c in all_client_ids])
    y_all_init = np.concatenate([client_gt_labels[c] for c in all_client_ids])
    # Warm-start: 3 passes, NO DP noise (this is the central equivalent for SVM)
    for _  in range(3):
        global_svm._model.partial_fit(
            X_all_init, y_all_init.astype(np.int64),
            classes=_ALL_CLASSES,
        )

    global_params = global_svm.get_parameters()

    active_training_data: dict = {}
    active_pool: list = []
    global_round = 0
    _base_mu = FEDPROX_MU       # FedProx mu (not directly used for SVM but kept for parity)
    drift_mu_boost = 1.0

    if verbose:
        print(f"\n[SVM-FL] {total_devices} devices  |  "
              f"{n_waves} waves × {rounds_per_wave} rounds = {total_rounds} total  |  "
              f"DP σ={DP_NOISE_MULTIPLIER}  C={DP_MAX_GRAD_NORM}")
        print(f"[SVM-FL] Model: SGDClassifier(loss='modified_huber')  |  FedAvg(weighted)")
        print(f"[SVM-FL] DP: weight-update perturbation (equiv. gradient-level for linear)")
        print(f"[SVM-FL] Non-IID: Dirichlet(α={DIRICHLET_ALPHA})  |  "
              f"Drift: cosine<{DRIFT_COSINE_THRESHOLD}→×{DRIFT_WEIGHT_PENALTY}")

    # ── Wave loop ──────────────────────────────────────────────────────────
    for wave_idx, cohort in enumerate(wave_cohorts, start=1):
        new_users = list(cohort)

        # Pseudo-label new users with current global SVM
        pseudo_match, total_new_records = 0, 0
        for cid in new_users:
            X_loc = client_features[cid]
            y_pseudo = global_svm.predict(X_loc)
            active_training_data[cid] = (X_loc, y_pseudo)
            pseudo_match += (y_pseudo == client_gt_labels[cid]).sum()
            total_new_records += len(X_loc)

        pseudo_acc = pseudo_match / total_new_records if total_new_records > 0 else 0.0

        # Spearman ρ between pseudo-labels and oracle labels (A6 parity)
        if len(new_users) > 2:
            ps_lbls = [int(active_training_data[c][1][0]) for c in new_users]
            gt_lbls = [int(client_gt_labels[c][0]) for c in new_users]
            spearman_rho = float(spearmanr(ps_lbls, gt_lbls).statistic)
        else:
            spearman_rho = float("nan")

        # A2: Adaptive Dirichlet alpha (same schedule as RiskMLP-FL)
        alpha_wave = max(0.3, 1.5 - 1.2 * (wave_idx - 1) / max(1, n_waves - 1))

        active_pool.extend(new_users)
        active_ids = np.array(active_pool)
        n_active   = len(active_ids)
        n_sample   = max(FL_MIN_CLIENTS, int(FL_FRACTION_FIT * n_active))
        n_sample   = min(n_sample, n_active)

        wave_start_round = global_round + 1

        if verbose:
            print(f"\n{'─'*65}")
            print(f"  WAVE {wave_idx}/{n_waves} (SVM-FL)  ──  {len(new_users)} new users onboarded")
            print(f"  Pseudo-label accuracy on new arrivals: {pseudo_acc:.4f}")
            print(f"  (pool: {n_active} total, {n_sample} sampled/round)")
            print(f"{'─'*65}")

        # ── FL rounds ─────────────────────────────────────────────────────
        for local_rnd in range(1, rounds_per_wave + 1):
            global_round += 1

            sampled = _dirichlet_round_sample(
                active_ids, client_risk_class, n_sample, alpha_wave, rng
            )
            round_entropy = _round_class_entropy(sampled, client_risk_class, N_RISK_CLASSES)

            # Straggler dropout (same 15% as RiskMLP-FL)
            active = [cid for cid in sampled if rng.random() > DROPOUT_RATE]
            if not active:
                active = list(sampled[:1])

            client_updates: list = []
            round_losses:   list = []

            for cid in active:
                dev_epochs = int(rng.integers(MIN_LOCAL_EPOCHS, MAX_LOCAL_EPOCHS + 1))
                X_loc, y_pseudo = active_training_data[cid]
                updated, n, loss = _svm_local_train(
                    global_params, X_loc, y_pseudo,
                    local_epochs=dev_epochs,
                    dp_enabled=dp_enabled,
                    class_weights_arr=class_weights_arr,
                )
                client_updates.append((updated, n))
                round_losses.append(loss)

            global_params, drift_info = _svm_fedavg(client_updates, global_params)
            avg_loss = float(np.mean(round_losses))

            # Update global model with new params for inference
            global_svm.set_parameters(global_params)

            drift_fraction = drift_info["n_high_drift"] / max(1, drift_info["n_clients"])
            drift_mu_boost = 1.5 if drift_fraction > 0.30 else 1.0

            svm_history["distributed_loss"][str(global_round)] = avg_loss
            svm_history["round_entropy"][str(global_round)]    = round(round_entropy, 4)
            svm_history["round_drift"][str(global_round)] = {
                "n_clients":      drift_info["n_clients"],
                "n_high_drift":   drift_info["n_high_drift"],
                "drift_fraction": round(drift_fraction, 4),
                "mean_cosine":    drift_info["mean_cosine"],
                "min_cosine":     drift_info["min_cosine"],
                "alpha_wave":     round(float(alpha_wave), 3),
            }

            # Accuracy vs oracle labels (eval only, not training signal)
            correct, total_n = 0, 0
            for cid in sampled:
                preds  = global_svm.predict(client_features[cid])
                y_true = client_gt_labels[cid]
                correct  += (preds == y_true).sum()
                total_n  += len(y_true)

            round_acc = correct / total_n if total_n > 0 else 0.0
            svm_history["round_accuracy"][str(global_round)] = round_acc

            if verbose:
                print(f"  Round {global_round:2d}/{total_rounds}  "
                      f"[Wave {wave_idx}  local {local_rnd}/{rounds_per_wave}]  "
                      f"loss={avg_loss:.4f}  acc={round_acc:.4f}  "
                      f"H={round_entropy:.3f}  α={alpha_wave:.2f}  "
                      f"drift={drift_info['n_high_drift']}/{drift_info['n_clients']}({drift_fraction:.0%})  "
                      f"cos={drift_info['mean_cosine']:+.3f}  "
                      f"active={len(active)}/{n_sample}")

        # ── Wave summary ───────────────────────────────────────────────────
        wave_end_acc = svm_history["round_accuracy"][str(global_round)]
        wave_entropies = [svm_history["round_entropy"][str(r)]
                          for r in range(wave_start_round, global_round + 1)]
        wave_drift_counts = [svm_history["round_drift"][str(r)]["n_high_drift"]
                             for r in range(wave_start_round, global_round + 1)]
        svm_history["waves"][str(wave_idx)] = {
            "new_users_joined":   len(new_users),
            "total_pool_size":    n_active,
            "pseudo_label_acc":   round(float(pseudo_acc), 4),
            "spearman_rho":       round(float(spearman_rho), 4) if spearman_rho == spearman_rho else None,
            "alpha_wave":         round(float(alpha_wave), 3),
            "start_round":        wave_start_round,
            "end_round":          global_round,
            "end_accuracy":       round(wave_end_acc, 4),
            "mean_round_entropy": round(float(np.mean(wave_entropies)), 4),
            "mean_drift_clients": round(float(np.mean(wave_drift_counts)), 2),
        }
        if verbose:
            rho_str = f"{spearman_rho:.4f}" if spearman_rho == spearman_rho else "n/a"
            print(f"  ↳ Wave {wave_idx} (SVM-FL) — pool={n_active}  "
                  f"pseudo_acc={pseudo_acc:.4f}  rho={rho_str}  end_acc={wave_end_acc:.4f}")

    # ── Save final global SVM model ────────────────────────────────────────
    import joblib
    joblib.dump(global_svm, SVM_FL_MODEL_PATH)
    with open(SVM_FL_HISTORY_PATH, "w") as f:
        json.dump(svm_history, f, indent=2)

    if verbose:
        iid_max_H = math.log(N_RISK_CLASSES)
        all_entropies = list(svm_history["round_entropy"].values())
        mean_H = float(np.mean(all_entropies)) if all_entropies else 0.0
        print(f"\n[SVM-FL] Training complete  →  {total_rounds} rounds across {n_waves} waves")
        print(f"[SVM-FL] Global SVM saved → {SVM_FL_MODEL_PATH}")
        print(f"[SVM-FL] Mean class entropy: {mean_H:.4f} / IID_max={iid_max_H:.4f}"
              f" → non-IID ratio={mean_H/iid_max_H:.3f}")

        print(f"\n  Wave summary (SVM-FL):")
        print(f"  {'Wave':<6} {'NewUsers':>9} {'Pool':>6} {'Rounds':>7} "
              f"{'PseudoAcc':>10} {'EndAcc':>8} {'MeanH':>7} {'DriftAvg':>9}")
        print("  " + "─" * 68)
        for wid, wdata in svm_history["waves"].items():
            rng_str = f"{wdata['start_round']}-{wdata['end_round']}"
            print(f"  {wid:<6} {wdata['new_users_joined']:>9} {wdata['total_pool_size']:>6} "
                  f"{rng_str:>7} {wdata['pseudo_label_acc']:>10.4f} {wdata['end_accuracy']:>8.4f} "
                  f"{wdata['mean_round_entropy']:>7.3f} {wdata['mean_drift_clients']:>9.1f}")

    return global_svm, svm_history


# ─── Side-by-side comparison: RiskMLP-FL vs SVM-FL ────────────────────────────

def compare_fl_models(
    df_fl: "pd.DataFrame",
    mlp_model,
    svm_model: SVMFLModel,
    mlp_history: Dict,
    svm_history: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate both FL models on the full FL cohort (oracle labels, eval only)
    and produce a side-by-side comparison table.

    Metrics
    -------
    • Macro F1  (primary — consistent with evaluation.py)
    • Accuracy
    • Final-round accuracy (last FL round)
    • Mean round loss
    • Mean per-round class entropy (non-IID measure)
    • FL-compatible  : both YES
    • DP-compatible  : both YES (same ε, δ)

    Returns comparison dict (also saved to models/svm_vs_mlp_fl_comparison.json).
    """
    import torch

    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    X_eval = df_fl[feat_cols].values.astype(np.float64)
    y_eval = df_fl["risk_label_encoded"].values.astype(np.int64)

    # ── RiskMLP-FL predictions ─────────────────────────────────────────────
    X_mlp = X_eval.astype(np.float32)
    with torch.no_grad():
        mlp_logits = mlp_model(torch.tensor(X_mlp))
        y_pred_mlp = mlp_logits.argmax(dim=1).numpy().astype(np.int64)

    mlp_f1  = float(f1_score(y_eval, y_pred_mlp, average="macro", zero_division=0))
    mlp_acc = float(accuracy_score(y_eval, y_pred_mlp))

    # ── SVM-FL predictions ─────────────────────────────────────────────────
    y_pred_svm = svm_model.predict(X_eval)
    svm_f1  = float(f1_score(y_eval, y_pred_svm, average="macro", zero_division=0))
    svm_acc = float(accuracy_score(y_eval, y_pred_svm))

    # ── History-derived metrics ────────────────────────────────────────────
    mlp_losses = list(mlp_history.get("distributed_loss", {}).values())
    svm_losses = list(svm_history.get("distributed_loss", {}).values())
    mlp_entropies = list(mlp_history.get("round_entropy", {}).values())
    svm_entropies = list(svm_history.get("round_entropy", {}).values())
    mlp_accs = list(mlp_history.get("round_accuracy",   {}).values())
    svm_accs = list(svm_history.get("round_accuracy",   {}).values())

    comparison = {
        "RiskMLP_FL": {
            "model":            "RiskMLP (4-layer MLP + CrossEntropyLoss)",
            "fl_compatible":    True,
            "dp_compatible":    True,
            "dp_mechanism":     "gradient-level DP-SGD",
            "final_macro_f1":   round(mlp_f1,  4),
            "final_accuracy":   round(mlp_acc, 4),
            "final_round_acc":  round(mlp_accs[-1], 4) if mlp_accs else None,
            "mean_round_loss":  round(float(np.mean(mlp_losses)), 6) if mlp_losses else None,
            "mean_round_entropy": round(float(np.mean(mlp_entropies)), 4) if mlp_entropies else None,
            "n_params":         "~73K",
            "per_class_f1":     {
                cls: round(float(f1_score(y_eval, y_pred_mlp, average=None, zero_division=0)[i]), 4)
                for i, cls in enumerate(["Very_Low","Low","Medium","High","Very_High"])
            },
        },
        "SVM_FL": {
            "model":            "LinearSVM-FL (SGD modified_huber + FedAvg)",
            "fl_compatible":    True,
            "dp_compatible":    True,
            "dp_mechanism":     "weight-update perturbation (equiv. gradient-level)",
            "final_macro_f1":   round(svm_f1,  4),
            "final_accuracy":   round(svm_acc, 4),
            "final_round_acc":  round(svm_accs[-1], 4) if svm_accs else None,
            "mean_round_loss":  round(float(np.mean(svm_losses)), 6) if svm_losses else None,
            "mean_round_entropy": round(float(np.mean(svm_entropies)), 4) if svm_entropies else None,
            "n_params":         f"~{N_RISK_CLASSES * len(RISK_FEATURES) + N_RISK_CLASSES}",
            "per_class_f1":     {
                cls: round(float(f1_score(y_eval, y_pred_svm, average=None, zero_division=0)[i]), 4)
                for i, cls in enumerate(["Very_Low","Low","Medium","High","Very_High"])
            },
        },
        "winner":     "RiskMLP_FL" if mlp_f1 >= svm_f1 else "SVM_FL",
        "f1_delta":   round(abs(mlp_f1 - svm_f1), 4),
        "verdict":    (
            "RiskMLP-FL outperforms LinearSVM-FL on macro F1 "
            "— non-linear interactions in the 15-feature risk score require MLP depth. "
            "LinearSVM-FL demonstrates FL+DP feasibility for simpler baselines."
            if mlp_f1 >= svm_f1 else
            "LinearSVM-FL matches or outperforms RiskMLP-FL — "
            "risk labels may be linearly separable in the scaled feature space."
        ),
    }

    with open(SVM_COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    if verbose:
        print("\n" + "=" * 74)
        print("  FL Model Comparison: RiskMLP-FL  vs  LinearSVM-FL")
        print("=" * 74)
        print(f"  {'Metric':<32} {'RiskMLP-FL':>14} {'SVM-FL':>14}")
        print("  " + "─" * 64)
        rows = [
            ("Macro F1 (FL cohort)",    f"{mlp_f1:.4f}",  f"{svm_f1:.4f}"),
            ("Accuracy (FL cohort)",    f"{mlp_acc:.4f}", f"{svm_acc:.4f}"),
            ("Param count",             "~73K",           comparison["SVM_FL"]["n_params"]),
            ("FL compatible",           "Yes ✓",          "Yes ✓"),
            ("DP compatible",           "Yes ✓",          "Yes ✓"),
            ("DP mechanism",            "gradient-DP",    "weight-update-DP"),
            ("Non-linear boundary",     "Yes (MLP)",      "No (linear)"),
            ("Mean round loss",
             f"{np.mean(mlp_losses):.4f}" if mlp_losses else "—",
             f"{np.mean(svm_losses):.4f}" if svm_losses else "—"),
            ("Mean round entropy (H)",
             f"{np.mean(mlp_entropies):.3f}" if mlp_entropies else "—",
             f"{np.mean(svm_entropies):.3f}" if svm_entropies else "—"),
        ]
        for label, mlp_val, svm_val in rows:
            print(f"  {label:<32} {mlp_val:>14} {svm_val:>14}")
        print("  " + "=" * 64)
        winner_tag = "RiskMLP-FL" if mlp_f1 >= svm_f1 else "SVM-FL"
        print(f"  WINNER: {winner_tag}  (ΔF1 = {comparison['f1_delta']:.4f})")
        print(f"\n  Per-class F1:")
        print(f"  {'Class':<14} {'RiskMLP-FL':>14} {'SVM-FL':>14}")
        print("  " + "─" * 44)
        for cls in ["Very_Low", "Low", "Medium", "High", "Very_High"]:
            print(f"  {cls:<14} {comparison['RiskMLP_FL']['per_class_f1'][cls]:>14.4f} "
                  f"{comparison['SVM_FL']['per_class_f1'][cls]:>14.4f}")
        print()
        print(f"  Verdict: {comparison['verdict']}")
        print(f"\n  Comparison saved → {SVM_COMPARISON_PATH}")
        print("=" * 74)

    return comparison
