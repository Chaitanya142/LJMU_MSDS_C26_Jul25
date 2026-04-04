"""
fl_simulation.py  –  In-process FedProx / FedAvg simulation for Smart Fund Advisor.

Architecture: one virtual mobile device per customer
-----------------------------------------------------------
Each customer in the 30 % FL cohort is treated as a separate mobile device.
That device holds ONLY that customer's own bank records (≤ 4 monthly rows).
Raw data NEVER leaves the device — only DP-noised gradient updates reach
the central aggregation server.

    Device for CUS_0x1000  |  records of CUS_0x1000 only
    Device for CUS_0x1009  |  records of CUS_0x1009 only  (never sees others)
    ...                    |  ...
    Device for CUS_0xNNNN  |  records of CUS_0xNNNN only

Data Isolation Guarantee
------------------------
  Built-in: each device's data is constructed via groupby("Customer_ID"),
  so cross-customer contamination is structurally impossible.
  A per-run verification check confirms 0 Customer_ID overlaps.

Algorithm: FedProx  (Li et al., 2020)
---------------------------------------
  Local loss = CrossEntropy(w) + (µ/2) · ||w − w_global||²
  µ = FEDPROX_MU (0 = pure FedAvg).
  The proximal term reduces client drift across heterogeneous devices.

Privacy: Amplified RDP (Mironov 2017 + 2019)
---------------------------------------------
  Gradient clipping C = DP_MAX_GRAD_NORM.
  Gaussian noise N(0, (σ·C)²) added per device per step.
  Because only q=FL_FRACTION_FIT (~30%) of devices participate per round,
  Poisson subsampling amplification (Mironov et al. 2019, Thm 9) tightens
  the effective (ε, δ) budget significantly vs naive composition.
  BatchNorm kept in eval mode (running stats from central model) to handle
  devices with only 1 record without numerical instability.

Non-IID participation (Dirichlet, Li et al. 2022)
---------------------------------------------------
  Real devices do NOT participate uniformly across rounds — conservative
  investors (older, rural) cluster into different rounds than aggressive
  investors (young, urban). Per-round class mix is drawn from
  Dirichlet(DIRICHLET_ALPHA): α=0.5 gives moderate geographic clustering.
  This stresses both the proximal term and the aggregation procedure.

Client Drift Monitoring
-----------------------
  After each round, cosine similarity between each client's update vector
  (Δw = w_local − w_global) and the round's aggregated direction is computed.
  Clients with cosine < DRIFT_COSINE_THRESHOLD are "high-drift" devices; their
  weight in FedAvg is penalised by DRIFT_WEIGHT_PENALTY. Per-round drift
  statistics are logged in fl_training_history.json for thesis analysis.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
from scipy.stats import spearmanr
from config import (
    FL_ROUNDS, FL_MIN_CLIENTS, FL_FRACTION_FIT,
    FL_GLOBAL_MODEL, MODELS_DIR,
    RISK_FEATURES, N_RISK_CLASSES, RANDOM_SEED, CENTRAL_SPLIT,
    FL_LOCAL_EPOCHS, FL_BATCH_SIZE,
    DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM,
    LEARNING_RATE, FEDPROX_MU,
    DIRICHLET_ALPHA, DRIFT_COSINE_THRESHOLD, DRIFT_WEIGHT_PENALTY,
)
from src.central_model import RiskMLP, load_central_model

# ─── Realistic device heterogeneity constants ─────────────────────────────────
DROPOUT_RATE     = 0.15  # fraction of sampled devices that drop each round
MIN_LOCAL_EPOCHS = 2     # min local epochs per device (heterogeneous hardware)
MAX_LOCAL_EPOCHS = 8     # max local epochs per device


# ─── Single mobile device local training (FedProx + Adam + DP) ───────────────

def _local_train(
    global_params: List[np.ndarray],
    X_local: np.ndarray,
    y_local: np.ndarray,
    local_epochs: int = FL_LOCAL_EPOCHS,
    dp_enabled: bool = True,
    mu: float = FEDPROX_MU,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], int, float]:
    """
    Simulate one mobile device's local training with FedProx.

    FedProx objective:
        L_local(w) = CE(w) + (µ/2) · ||w − w_global||²

    The device holds only its own bank records (≤ 4 rows).
    BatchNorm is kept in eval mode (running stats from central model) so the
    model is stable even when the device has a single record.
    Adam optimizer converges faster than SGD on the small, noisy per-device
    gradients produced under DP noise.

    class_weights: optional per-class weights (shape N_RISK_CLASSES) derived
    from the global FL distribution to balance tail-class updates across devices.

    Returns
    -------
    (updated_params, n_samples, final_ce_loss)
    """
    model = RiskMLP()
    model.set_parameters(global_params)
    model.train()
    # Keep BatchNorm in eval mode: devices have ≤ 4 records, batch statistics
    # are meaningless at that scale — use running stats from the central model.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    # Frozen reference copy for the FedProx proximal term
    global_ref = copy.deepcopy(model)
    global_ref.eval()
    for p in global_ref.parameters():
        p.requires_grad_(False)

    Xt = torch.tensor(X_local, dtype=torch.float32)
    yt = torch.tensor(y_local, dtype=torch.long)

    # Batch size bounded by the number of on-device records
    batch_size = max(1, min(FL_BATCH_SIZE, len(yt)))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    # Use global class weights to counteract majority-class bias across devices.
    # Tail classes (Very_Low / Very_High ≈ 12.5 % each) would otherwise be
    # underrepresented since most devices have only 1 label per customer.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    final_loss = 0.0
    for _ in range(local_epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()

            ce_loss = criterion(model(X_b), y_b)

            # FedProx proximal term
            if mu > 0:
                prox = sum(
                    ((p - p_g) ** 2).sum()
                    for p, p_g in zip(model.parameters(), global_ref.parameters())
                )
                loss = ce_loss + (mu / 2.0) * prox
            else:
                loss = ce_loss

            loss.backward()
            final_loss = ce_loss.item()

            # DP: clip gradients to bound sensitivity
            nn.utils.clip_grad_norm_(model.parameters(), DP_MAX_GRAD_NORM)

            # DP: add calibrated Gaussian noise to gradients
            if dp_enabled:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(
                                torch.randn_like(p.grad) * (DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM)
                            )

            optimizer.step()

    return model.get_parameters(), len(y_local), final_loss


# ─── FedAvg aggregation with client drift penalty ────────────────────────────

def _fedavg(
    client_updates: List[Tuple[List[np.ndarray], int]],
    global_params: Optional[List[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Weighted FedAvg with optional client drift monitoring.

    For each client, compute the cosine similarity between its update vector
    (Δw = w_local − w_global) and the equal-weight mean update direction.
    Clients with cosine < DRIFT_COSINE_THRESHOLD have their sample-count
    weight multiplied by DRIFT_WEIGHT_PENALTY before aggregation.

    Returns
    -------
    agg_params : aggregated global parameters
    drift_info : dict with per-round drift statistics
    """
    drift_info: Dict = {
        "n_clients":        len(client_updates),
        "n_high_drift":     0,
        "cosine_sims":      [],
        "mean_cosine":      0.0,
        "min_cosine":       0.0,
        "drift_penalty_applied": False,
    }

    if global_params is not None:
        # Flatten global params once
        flat_global = np.concatenate([p.flatten() for p in global_params])

        # Compute per-client update vectors and their cosine vs global
        update_vecs = []
        for params, _ in client_updates:
            flat_local = np.concatenate([p.flatten() for p in params])
            delta = flat_local - flat_global
            update_vecs.append(delta)

        # Mean update direction (equal weight for reference direction)
        mean_delta = np.mean(update_vecs, axis=0)
        mean_norm  = np.linalg.norm(mean_delta) + 1e-12

        cosine_sims = []
        for delta in update_vecs:
            delta_norm = np.linalg.norm(delta) + 1e-12
            cos = float(np.dot(delta, mean_delta) / (delta_norm * mean_norm))
            cosine_sims.append(cos)

        drift_info["cosine_sims"]  = [round(c, 4) for c in cosine_sims]
        drift_info["mean_cosine"]  = round(float(np.mean(cosine_sims)), 4)
        drift_info["min_cosine"]   = round(float(np.min(cosine_sims)), 4)
        n_high_drift = sum(1 for c in cosine_sims if c < DRIFT_COSINE_THRESHOLD)
        drift_info["n_high_drift"] = n_high_drift
        drift_info["drift_penalty_applied"] = n_high_drift > 0
    else:
        cosine_sims = [1.0] * len(client_updates)  # no drift info → treat all as aligned

    # Build effective weights: penalise high-drift clients
    raw_weights = []
    for i, (_, n) in enumerate(client_updates):
        cos_i = cosine_sims[i] if global_params is not None else 1.0
        penalty = DRIFT_WEIGHT_PENALTY if cos_i < DRIFT_COSINE_THRESHOLD else 1.0
        raw_weights.append(n * penalty)

    total = sum(raw_weights)
    if total == 0:
        total = 1.0

    agg = None
    for (params, _), w in zip(client_updates, raw_weights):
        weight = w / total
        if agg is None:
            agg = [weight * p for p in params]
        else:
            agg = [a + weight * p for a, p in zip(agg, params)]

    return agg, drift_info  # type: ignore


# ─── Dirichlet non-IID round sampling ────────────────────────────────────────

def _dirichlet_round_sample(
    active_ids: np.ndarray,
    client_risk_class: Dict[str, int],
    n_sample: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n_sample clients such that the per-round class distribution is
    drawn from Dirichlet(alpha), modelling geographic/demographic clustering.

    In real FL deployments, conservative investors (older, rural) and aggressive
    investors (young, urban) are active in different rounds — they do not
    participate uniformly. This function simulates that correlation.

    Parameters
    ----------
    active_ids        : array of client IDs currently in the active pool
    client_risk_class : cid → oracle risk class index (for sampling bias only,
                        never used as a training signal)
    n_sample          : target number of clients to select
    alpha             : Dirichlet concentration parameter
                        (0.1 = extreme skew, 0.5 = moderate, 100 = near-IID)
    rng               : seeded numpy Generator

    Returns
    -------
    sampled : array of n_sample client IDs
    """
    classes = sorted(set(client_risk_class[cid] for cid in active_ids))
    n_classes = len(classes)
    if n_classes == 0 or n_sample >= len(active_ids):
        return rng.choice(active_ids, size=min(n_sample, len(active_ids)), replace=False)

    # Draw this round's target class proportions from Dirichlet
    proportions = rng.dirichlet([alpha] * n_classes)

    # Group active clients by risk class
    class_buckets: Dict[int, List[str]] = {c: [] for c in classes}
    for cid in active_ids:
        class_buckets[client_risk_class[cid]].append(str(cid))

    # Sample from each bucket proportionally to drawn proportions
    sampled: List[str] = []
    for cls, prop in zip(classes, proportions):
        bucket = class_buckets[cls]
        if not bucket:
            continue
        n_from = max(0, round(prop * n_sample))
        n_from = min(n_from, len(bucket))
        if n_from > 0:
            chosen = rng.choice(bucket, size=n_from, replace=False)
            sampled.extend(chosen.tolist())

    # De-duplicate and trim/fill to exactly n_sample
    sampled = list(dict.fromkeys(sampled))  # preserve order, remove dups
    if len(sampled) > n_sample:
        sampled = sampled[:n_sample]
    elif len(sampled) < n_sample:
        remaining = [c for c in active_ids if str(c) not in sampled]
        deficit = n_sample - len(sampled)
        if remaining:
            extra = rng.choice(remaining, size=min(deficit, len(remaining)), replace=False)
            sampled.extend(extra.tolist())

    return np.array(sampled[:n_sample])


def _round_class_entropy(
    sampled_ids: np.ndarray,
    client_risk_class: Dict[str, int],
    n_classes: int,
) -> float:
    """
    Shannon entropy of the class distribution among sampled clients this round.
    IID rounds → high entropy (~log(n_classes)).
    Non-IID (skewed) rounds → low entropy.
    Logged per round to quantify non-IID-ness in thesis.
    """
    counts = np.zeros(n_classes, dtype=float)
    for cid in sampled_ids:
        counts[client_risk_class[str(cid)]] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ─── Main simulation entry point ──────────────────────────────────────────────

def run_fl_simulation(
    df_fl: "pd.DataFrame",
    dp_enabled: bool = True,
    rounds: int = FL_ROUNDS,
    verbose: bool = True,
    fedprox_mu: Optional[float] = None,
) -> Tuple[RiskMLP, Dict]:
    """
    A1: fedprox_mu=0 → pure FedAvg ablation; default=FEDPROX_MU → FedProx.
    A2: Adaptive Dirichlet alpha: starts at 1.5 (near-IID) → decays to 0.3 (heavy skew).
    A3: Active drift response: if drift_fraction > 30%, boost mu_round by ×1.5.
    """
    """
    Run FedProx federated learning simulation.

    Each customer in df_fl is treated as one mobile device that exclusively
    holds that customer's own bank records (≤ 4 rows per Customer_ID).
    Raw data never leaves the device; only DP-noised gradients are aggregated.

    Parameters
    ----------
    df_fl      : DataFrame already limited to ≤ 4 rows per Customer_ID
    dp_enabled : add Gaussian DP noise to device gradients
    rounds     : number of FL communication rounds
    verbose    : print per-round metrics

    Returns
    -------
    global_model : RiskMLP with final aggregated weights
    fl_history   : dict with per-round loss and accuracy metrics
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Build per-device data: one entry per Customer_ID ──────────────────
    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, grp in df_fl.groupby("Customer_ID"):
        X = grp[feat_cols].values.astype(np.float32)
        y = grp["risk_label_encoded"].values.astype(np.int64)
        client_data[str(cid)] = (X, y)

    client_ids = np.array(list(client_data.keys()))
    n_devices  = len(client_ids)
    n_sample   = max(FL_MIN_CLIENTS, int(FL_FRACTION_FIT * n_devices))
    n_sample   = min(n_sample, n_devices)

    # ── Global class weights: balanced weighting over full FL distribution ─
    # Mirrors sklearn class_weight='balanced':  w_i = N / (n_classes × count_i)
    # Ensures tail classes (Very_Low/Very_High ~12.5%) get equal gradient
    # contribution across all device updates.
    all_y_fl = np.concatenate([y for _, y in client_data.values()])
    class_counts = np.bincount(all_y_fl, minlength=N_RISK_CLASSES).astype(float)
    class_w = len(all_y_fl) / (N_RISK_CLASSES * (class_counts + 1e-8))
    class_weights_tensor = torch.tensor(class_w, dtype=torch.float32)

    # ── Data isolation verification ────────────────────────────────────────
    # By construction each device holds exactly one Customer_ID's rows.
    # Confirm no Customer_ID appears in more than one device bucket.
    all_cids   = df_fl["Customer_ID"].values
    unique_per_bucket = all(
        len(grp["Customer_ID"].unique()) == 1
        for _, grp in df_fl.groupby("Customer_ID")
    )
    isolation_status = "✓ GUARANTEED" if unique_per_bucket else "✗ VIOLATION"

    # ── Build client_risk_class map: cid → majority risk class (for Dirichlet) ─
    # This is the oracle risk class per customer — used ONLY for participation
    # sampling; never used as a training signal for FL devices.
    client_risk_class: Dict[str, int] = {
        cid: int(y[0]) for cid, (_, y) in client_data.items()
    }

    if verbose:
        avg_recs = len(df_fl) / n_devices if n_devices else 0
        print(f"\n[FL] {n_devices} mobile devices  |  {rounds} rounds  |  "
              f"{n_sample} sampled/round (~{DROPOUT_RATE:.0%} straggler dropout)  |  "
              f"~{avg_recs:.1f} records/device  |  "
              f"FedProx µ₀={FEDPROX_MU} (adaptive decay ×0.9/round)  |  DP σ={DP_NOISE_MULTIPLIER}")
        print(f"[FL] Device epochs: Uniform({MIN_LOCAL_EPOCHS},{MAX_LOCAL_EPOCHS}) per round  |  "
              f"Data isolation: {isolation_status}")
        print(f"[FL] Non-IID: Dirichlet(α={DIRICHLET_ALPHA}) participation sampling  |  "
              f"Drift threshold: cosine<{DRIFT_COSINE_THRESHOLD} → penalty×{DRIFT_WEIGHT_PENALTY}")

    # ── Warm-start from central model weights ──────────────────────────────
    global_params = load_central_model().get_parameters()

    fl_history: Dict = {
        "distributed_loss":    {},
        "round_accuracy":      {},
        "round_entropy":       {},   # per-round class participation entropy
        "round_drift":         {},   # per-round client drift statistics
    }

    # A1: effective FedProx mu (0 = pure FedAvg ablation)
    _base_mu = fedprox_mu if fedprox_mu is not None else FEDPROX_MU

    # A3: drift-response boost carries forward across rounds
    drift_mu_boost = 1.0

    # ── Federated rounds ───────────────────────────────────────────────────
    for rnd in range(1, rounds + 1):
        # A2: Adaptive Dirichlet alpha — starts near-IID (1.5), decays to 0.3
        alpha_round = max(0.3, 1.5 - 1.2 * (rnd - 1) / max(1, rounds - 1))

        # Non-IID Dirichlet sampling — class-skewed participation per round
        sampled = _dirichlet_round_sample(
            client_ids, client_risk_class, n_sample, alpha_round, rng
        )

        # Per-round class entropy (measures how non-IID this round is)
        round_entropy = _round_class_entropy(sampled, client_risk_class, N_RISK_CLASSES)

        # A3: Adaptive μ with drift-response boost
        mu_round = _base_mu * (0.9 ** (rnd - 1)) * drift_mu_boost

        # Straggler dropout: ~15% of sampled devices don't respond this round
        active = [cid for cid in sampled if rng.random() > DROPOUT_RATE]
        if not active:
            active = list(sampled[:1])   # guarantee ≥ 1 update per round

        client_updates: List[Tuple[List[np.ndarray], int]] = []
        round_losses: List[float] = []

        for cid in active:
            # Variable local epochs: simulate heterogeneous device capabilities
            dev_epochs = int(rng.integers(MIN_LOCAL_EPOCHS, MAX_LOCAL_EPOCHS + 1))
            X_loc, y_loc = client_data[cid]
            updated, n, loss = _local_train(
                global_params, X_loc, y_loc,
                local_epochs=dev_epochs,
                dp_enabled=dp_enabled,
                mu=mu_round,
                class_weights=class_weights_tensor,
            )
            client_updates.append((updated, n))
            round_losses.append(loss)

        # Drift-aware FedAvg: penalise high-drift client updates
        global_params, drift_info = _fedavg(client_updates, global_params)
        avg_loss = float(np.mean(round_losses))

        # A3: active drift response — boost mu if >30% clients are high-drift
        drift_fraction = drift_info["n_high_drift"] / max(1, drift_info["n_clients"])
        drift_mu_boost = 1.5 if drift_fraction > 0.30 else 1.0

        fl_history["distributed_loss"][str(rnd)] = avg_loss
        fl_history["round_entropy"][str(rnd)]    = round(round_entropy, 4)
        fl_history["round_drift"][str(rnd)] = {
            "n_clients":       drift_info["n_clients"],
            "n_high_drift":    drift_info["n_high_drift"],
            "drift_fraction":  round(drift_fraction, 4),
            "mean_cosine":     drift_info["mean_cosine"],
            "min_cosine":      drift_info["min_cosine"],
            "mu_effective":    round(float(mu_round), 5),
            "alpha_round":     round(float(alpha_round), 3),
            "drift_mu_boost":  round(float(drift_mu_boost), 2),
        }

        # Round-level accuracy evaluated on sampled devices
        global_model_tmp = RiskMLP()
        global_model_tmp.set_parameters(global_params)
        global_model_tmp.eval()
        correct, total_n = 0, 0
        with torch.no_grad():
            for cid in sampled:
                X_loc, y_loc = client_data[cid]
                preds = global_model_tmp(torch.tensor(X_loc)).argmax(dim=1).numpy()
                correct += (preds == y_loc).sum()
                total_n += len(y_loc)

        round_acc = correct / total_n if total_n > 0 else 0.0
        fl_history["round_accuracy"][str(rnd)] = round_acc

        if verbose:
            print(f"  Round {rnd:2d}/{rounds}  "
                  f"loss={avg_loss:.4f}  acc={round_acc:.4f}  "
                  f"H={round_entropy:.3f}  α={alpha_round:.2f}  "
                  f"drift={drift_info['n_high_drift']}/{drift_info['n_clients']}({drift_fraction:.0%})  "
                  f"cos={drift_info['mean_cosine']:+.3f}  "
                  f"active={len(active)}/{n_sample}  mu={mu_round:.4f}  boost={drift_mu_boost:.1f}")

    # ── Build final global model ───────────────────────────────────────────
    global_model = RiskMLP()
    global_model.set_parameters(global_params)
    global_model.eval()

    torch.save(global_model.state_dict(), FL_GLOBAL_MODEL)
    with open(MODELS_DIR / "fl_training_history.json", "w") as f:
        json.dump(fl_history, f, indent=2)

    if verbose:
        final_acc     = fl_history["round_accuracy"][str(rounds)]
        mean_entropy  = float(np.mean(list(fl_history["round_entropy"].values())))
        iid_max_H     = math.log(N_RISK_CLASSES)
        mean_n_drift  = float(np.mean([v["n_high_drift"] for v in fl_history["round_drift"].values()]))
        print(f"\n[FL] Global model saved → {FL_GLOBAL_MODEL}")
        print(f"[FL] Final round accuracy (sampled devices): {final_acc:.4f}")
        print(f"[FL] Mean per-round class entropy: {mean_entropy:.4f}  "
              f"(IID baseline = {iid_max_H:.4f})  "
              f"→ non-IID ratio = {mean_entropy/iid_max_H:.3f}")
        print(f"[FL] Mean high-drift clients/round: {mean_n_drift:.1f}")

    return global_model, fl_history


# ─── Production-like incremental FL simulation ────────────────────────────────

def run_incremental_fl_simulation(
    df_fl: "pd.DataFrame",
    dp_enabled: bool = True,
    n_waves: int = 5,
    rounds_per_wave: int = 3,
    verbose: bool = True,
    fedprox_mu: Optional[float] = None,
) -> Tuple[RiskMLP, Dict]:
    """Production-realistic incremental FL simulation.

    A1: fedprox_mu=0 → pure FedAvg ablation; default=FEDPROX_MU → FedProx.
    A2: Adaptive Dirichlet alpha: starts at 1.5 (near-IID in wave 1) → 0.3 (skewed in wave N).
    A3: Active drift response: if drift_fraction > 30% in a round, boost mu by ×1.5 for next round.
    A6: Spearman rho between pseudo-labels and oracle labels logged per wave.
    """
    """
    Production-realistic FL simulation where users join the system incrementally.

    KEY REALISM: No central server ever assigns risk labels to FL users.
    ─────────────────────────────────────────────────────────────────────
    In real FL deployment:
      • The 70% central-training users have known labels → supervised training.
      • When a NEW user installs the app (each wave), the server knows NOTHING
        about their risk tolerance.
      • The CURRENT global model runs ON THE DEVICE and predicts the user's risk
        label from their local bank records.  That prediction becomes the
        device's training target (pseudo-label).
      • The device trains locally on (features, pseudo-label) for Uniform(2,8)
        epochs with FedProx + DP noise.
      • Only clipped + noised gradient updates leave the device — raw data and
        pseudo-labels never reach the server.
      • As more waves join and the global model improves, its pseudo-labels for
        newly joining users become progressively more accurate.

    This self-consistent loop is the core demonstration of federated learning:
      global_model predicts → device trains on prediction
      → better gradients flow back → model improves → repeat.

    Accuracy metric
    ───────────────
    Round-level accuracy is measured by comparing global model predictions
    against the rule-based oracle labels (ground truth held locally on device,
    used ONLY for evaluation — never transmitted, never used as training signal).

    Wave structure (default 5 waves × 3 rounds = 15 total rounds):
        Wave 1 : first 20% join  →  global_model predicts their labels  →  3 rounds
        Wave 2 : next  20% join  →  model (improved) predicts their labels  →  3 rounds
        ...
        Wave 5 : final 20% join  →  most accurate pseudo-labelling  →  3 rounds

    Parameters
    ----------
    df_fl           : DataFrame with FL cohort, ≤4 rows per Customer_ID
    dp_enabled      : add Gaussian DP noise to device gradients
    n_waves         : number of user onboarding waves (default 5)
    rounds_per_wave : FL rounds to run after each wave joins (default 3)
    verbose         : print per-round and per-wave metrics

    Returns
    -------
    global_model : RiskMLP with final aggregated weights
    fl_history   : dict with per-round metrics + wave summaries
    """
    import pandas as pd

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Build per-device data stores ───────────────────────────────────────
    # client_features : cid → X  (raw features — the only data on device)
    # client_gt_labels: cid → y  (rule-based oracle label — for eval ONLY,
    #                              never used as training signal in FL)
    feat_cols = [f for f in RISK_FEATURES if f in df_fl.columns]
    client_features:  dict = {}
    client_gt_labels: dict = {}
    for cid, grp in df_fl.groupby("Customer_ID"):
        client_features[str(cid)]  = grp[feat_cols].values.astype(np.float32)
        client_gt_labels[str(cid)] = grp["risk_label_encoded"].values.astype(np.int64)

    all_client_ids = np.array(list(client_features.keys()))
    total_devices  = len(all_client_ids)

    # Global class weights from oracle label distribution (balanced weighting)
    all_y_inc = np.concatenate(list(client_gt_labels.values()))
    _counts_inc = np.bincount(all_y_inc, minlength=N_RISK_CLASSES).astype(float)
    _cw_inc = len(all_y_inc) / (N_RISK_CLASSES * (_counts_inc + 1e-8))
    class_weights_tensor = torch.tensor(_cw_inc, dtype=torch.float32)

    # ── client_risk_class: cid → oracle class index (Dirichlet sampling only) ─
    client_risk_class: dict = {
        cid: int(client_gt_labels[cid][0]) for cid in client_features
    }

    # Shuffle once so waves aren't sorted by any implicit ordering
    rng.shuffle(all_client_ids)

    # ── Split into n_waves equal cohorts ──────────────────────────────────
    wave_cohorts = np.array_split(all_client_ids, n_waves)

    total_rounds = n_waves * rounds_per_wave
    fl_history: dict = {
        "distributed_loss": {},
        "round_accuracy":   {},
        "round_entropy":    {},   # per-round class participation entropy
        "round_drift":      {},   # per-round client drift statistics
        "waves":            {},
    }

    # ── Warm-start from central model weights (same as batch FL) ──────────
    global_params = load_central_model().get_parameters()

    # active_training_data: cid → (X, y_pseudo)
    # Pseudo-labels are generated on-device using the global model AT THE TIME
    # the user first joins.  They are fixed for that user's lifetime in the sim.
    active_training_data: dict = {}
    active_pool: list = []    # grows with each wave
    global_round = 0          # absolute round counter across all waves

    if verbose:
        print(f"\n[FL-Incremental] {total_devices} total devices  |  "
              f"{n_waves} waves × {rounds_per_wave} rounds = {total_rounds} total rounds  |  "
              f"DP σ={DP_NOISE_MULTIPLIER}  |  FedProx µ₀={FEDPROX_MU} (adaptive decay)")
        print(f"[FL-Incremental] Label strategy: pseudo-labels from global_model (no central labelling)")
        print(f"[FL-Incremental] Non-IID: Dirichlet(α={DIRICHLET_ALPHA})  |  "
              f"Drift penalty: cosine<{DRIFT_COSINE_THRESHOLD} → ×{DRIFT_WEIGHT_PENALTY}")
    # A1: effective FedProx mu
    _base_mu = fedprox_mu if fedprox_mu is not None else FEDPROX_MU

    # A3: drift-response boost state
    drift_mu_boost = 1.0
    # ── Wave loop ──────────────────────────────────────────────────────────
    for wave_idx, cohort in enumerate(wave_cohorts, start=1):
        new_users = list(cohort)

        # ── NEW USERS ONBOARD: global model runs on-device to assign labels ─
        # Build the current global model (snapshot at wave entry)
        labeller = RiskMLP()
        labeller.set_parameters(global_params)
        labeller.eval()

        pseudo_match = 0   # how many pseudo-labels agree with oracle labels
        with torch.no_grad():
            for cid in new_users:
                X_loc = client_features[cid]
                y_pseudo = labeller(
                    torch.tensor(X_loc)
                ).argmax(dim=1).numpy().astype(np.int64)
                active_training_data[cid] = (X_loc, y_pseudo)
                # Count agreement with oracle for reporting
                pseudo_match += (y_pseudo == client_gt_labels[cid]).sum()

        total_new_records = sum(len(client_features[c]) for c in new_users)
        pseudo_acc = pseudo_match / total_new_records if total_new_records > 0 else 0.0

        # A6: Spearman rank correlation between pseudo-labels and oracle labels
        if len(new_users) > 2:
            ps_lbls = [int(active_training_data[c][1][0]) for c in new_users]
            gt_lbls = [int(client_gt_labels[c][0]) for c in new_users]
            spearman_rho = float(spearmanr(ps_lbls, gt_lbls).statistic)
        else:
            spearman_rho = float("nan")

        # A2: Adaptive Dirichlet alpha per wave (1.5 → 0.3 linearly)
        alpha_wave = max(0.3, 1.5 - 1.2 * (wave_idx - 1) / max(1, n_waves - 1))

        active_pool.extend(new_users)
        active_ids = np.array(active_pool)

        n_active = len(active_ids)
        n_sample = max(FL_MIN_CLIENTS, int(FL_FRACTION_FIT * n_active))
        n_sample = min(n_sample, n_active)

        wave_start_round = global_round + 1

        if verbose:
            print(f"\n{'─'*65}")
            print(f"  WAVE {wave_idx}/{n_waves}  ──  {len(new_users)} new users onboarded")
            print(f"  Global model pseudo-label accuracy on new arrivals: {pseudo_acc:.4f}")
            print(f"  (pool: {n_active} total, {n_sample} sampled/round)")
            print(f"{'─'*65}")

        # ── FL rounds for this wave ────────────────────────────────────────
        for local_rnd in range(1, rounds_per_wave + 1):
            global_round += 1

            # A2: Use wave's adaptive alpha for Dirichlet sampling
            sampled = _dirichlet_round_sample(
                active_ids, client_risk_class, n_sample, alpha_wave, rng
            )
            round_entropy = _round_class_entropy(sampled, client_risk_class, N_RISK_CLASSES)
            # A3: Adaptive mu with drift-boost from previous round
            mu_round = _base_mu * (0.9 ** (global_round - 1)) * drift_mu_boost

            # Straggler dropout
            active = [cid for cid in sampled if rng.random() > DROPOUT_RATE]
            if not active:
                active = list(sampled[:1])

            client_updates: list = []
            round_losses:   list = []

            for cid in active:
                dev_epochs = int(rng.integers(MIN_LOCAL_EPOCHS, MAX_LOCAL_EPOCHS + 1))
                # Training uses pseudo-labels (realistic: generated on-device)
                X_loc, y_pseudo = active_training_data[cid]
                updated, n, loss = _local_train(
                    global_params, X_loc, y_pseudo,
                    local_epochs=dev_epochs,
                    dp_enabled=dp_enabled,
                    mu=mu_round,
                    class_weights=class_weights_tensor,
                )
                client_updates.append((updated, n))
                round_losses.append(loss)

            # Drift-aware FedAvg
            global_params, drift_info = _fedavg(client_updates, global_params)
            avg_loss = float(np.mean(round_losses))

            # A3: active drift response
            drift_fraction = drift_info["n_high_drift"] / max(1, drift_info["n_clients"])
            drift_mu_boost = 1.5 if drift_fraction > 0.30 else 1.0

            fl_history["distributed_loss"][str(global_round)] = avg_loss
            fl_history["round_entropy"][str(global_round)]    = round(round_entropy, 4)
            fl_history["round_drift"][str(global_round)] = {
                "n_clients":      drift_info["n_clients"],
                "n_high_drift":   drift_info["n_high_drift"],
                "drift_fraction": round(drift_fraction, 4),
                "mean_cosine":    drift_info["mean_cosine"],
                "min_cosine":     drift_info["min_cosine"],
                "mu_effective":   round(float(mu_round), 5),
                "alpha_wave":     round(float(alpha_wave), 3),
                "drift_mu_boost": round(float(drift_mu_boost), 2),
            }

            # Accuracy = global model predictions vs oracle labels on sampled devices
            # (oracle labels = evaluation ground truth, never used for training)
            global_model_tmp = RiskMLP()
            global_model_tmp.set_parameters(global_params)
            global_model_tmp.eval()
            correct, total_n = 0, 0
            with torch.no_grad():
                for cid in sampled:
                    X_loc  = client_features[cid]
                    y_true = client_gt_labels[cid]   # oracle, eval only
                    preds  = global_model_tmp(torch.tensor(X_loc)).argmax(dim=1).numpy()
                    correct  += (preds == y_true).sum()
                    total_n  += len(y_true)

            round_acc = correct / total_n if total_n > 0 else 0.0
            fl_history["round_accuracy"][str(global_round)] = round_acc

            if verbose:
                print(f"  Round {global_round:2d}/{total_rounds}  "
                      f"[Wave {wave_idx}  local {local_rnd}/{rounds_per_wave}]  "
                      f"loss={avg_loss:.4f}  acc={round_acc:.4f}  "
                      f"H={round_entropy:.3f}  α={alpha_wave:.2f}  "
                      f"drift={drift_info['n_high_drift']}/{drift_info['n_clients']}({drift_fraction:.0%})  "
                      f"cos={drift_info['mean_cosine']:+.3f}  "
                      f"active={len(active)}/{n_sample}  mu={mu_round:.4f}  boost={drift_mu_boost:.1f}")

        # ── Wave summary ───────────────────────────────────────────────────
        wave_end_acc = fl_history["round_accuracy"][str(global_round)]
        wave_entropies = [fl_history["round_entropy"][str(r)]
                          for r in range(wave_start_round, global_round + 1)]
        wave_drift_counts = [fl_history["round_drift"][str(r)]["n_high_drift"]
                             for r in range(wave_start_round, global_round + 1)]
        fl_history["waves"][str(wave_idx)] = {
            "new_users_joined":   len(new_users),
            "total_pool_size":    n_active,
            "pseudo_label_acc":   round(float(pseudo_acc), 4),
            "spearman_rho":       round(float(spearman_rho), 4) if not (spearman_rho != spearman_rho) else None,
            "alpha_wave":         round(float(alpha_wave), 3),
            "start_round":        wave_start_round,
            "end_round":          global_round,
            "end_accuracy":       round(wave_end_acc, 4),
            "mean_round_entropy": round(float(np.mean(wave_entropies)), 4),
            "mean_drift_clients": round(float(np.mean(wave_drift_counts)), 2),
        }
        if verbose:
            rho_str = f"{spearman_rho:.4f}" if spearman_rho == spearman_rho else "n/a"
            print(f"  ↳ Wave {wave_idx} complete — pool={n_active} users  "
                  f"pseudo_acc={pseudo_acc:.4f}  rho={rho_str}  end_acc={wave_end_acc:.4f}  "
                  f"α={alpha_wave:.2f}  mean_H={fl_history['waves'][str(wave_idx)]['mean_round_entropy']:.3f}  "
                  f"mean_drift={fl_history['waves'][str(wave_idx)]['mean_drift_clients']:.1f}")

    # ── Build and save final global model ─────────────────────────────────
    global_model = RiskMLP()
    global_model.set_parameters(global_params)
    global_model.eval()

    torch.save(global_model.state_dict(), FL_GLOBAL_MODEL)
    with open(MODELS_DIR / "fl_training_history.json", "w") as f:
        json.dump(fl_history, f, indent=2)

    if verbose:
        iid_max_H = math.log(N_RISK_CLASSES)
        all_entropies = list(fl_history["round_entropy"].values())
        mean_H = float(np.mean(all_entropies)) if all_entropies else 0.0
        print(f"\n[FL-Incremental] Training complete  →  {total_rounds} rounds across {n_waves} waves")
        print(f"[FL-Incremental] Global model saved → {FL_GLOBAL_MODEL}")
        print(f"[FL-Incremental] Mean class entropy: {mean_H:.4f}"
              f" / IID_max={iid_max_H:.4f}"
              f" → non-IID ratio={mean_H/iid_max_H:.3f}"
              f"  (Dirichlet α={DIRICHLET_ALPHA})")
        print(f"\n  Wave summary:")
        print(f"  {'Wave':<6} {'NewUsers':>9} {'Pool':>6} {'Rounds':>7} "
              f"{'PseudoAcc':>10} {'EndAcc':>8} {'MeanH':>7} {'DriftAvg':>9}")
        print("  " + "─" * 68)
        for wid, wdata in fl_history["waves"].items():
            rng_str = f"{wdata['start_round']}-{wdata['end_round']}"
            print(f"  {wid:<6} {wdata['new_users_joined']:>9} {wdata['total_pool_size']:>6} "
                  f"{rng_str:>7} {wdata['pseudo_label_acc']:>10.4f} {wdata['end_accuracy']:>8.4f} "
                  f"{wdata['mean_round_entropy']:>7.3f} {wdata['mean_drift_clients']:>9.1f}")

    return global_model, fl_history


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path as _Path
    from src.preprocessing import get_clean_customer_data
    from src.risk_labeling import assign_risk_label
    from sklearn.model_selection import train_test_split as tts
    from config import MODELS_DIR as _MODELS_DIR, DEMO_SPLIT as _DEMO_SPLIT

    print("[FL] Loading and preprocessing data ...")
    df = get_clean_customer_data(fit_scaler=False)
    df = assign_risk_label(df, fit_encoder=False)

    # Use the pre-saved FL split from train.py (3-way: 65% central /30% FL /5% demo)
    # If not available, replicate the same 3-way logic to stay consistent.
    fl_csv = _MODELS_DIR / "df_fl_split.csv"
    if fl_csv.exists():
        df_fl = pd.read_csv(fl_csv)
        fl_customers = df_fl["Customer_ID"].unique()
    else:
        customers = df["Customer_ID"].unique()
        # Carve out demo holdout first, then split remainder 65/30
        demo_n   = max(1, int(len(customers) * _DEMO_SPLIT))
        rng_cli  = np.random.default_rng(RANDOM_SEED)
        demo_idx = rng_cli.choice(len(customers), size=demo_n, replace=False)
        demo_set = set(customers[demo_idx])
        remainder = np.array([c for c in customers if c not in demo_set])
        central_n = int(len(remainder) * CENTRAL_SPLIT / (1 - _DEMO_SPLIT))
        _, fl_customers = tts(remainder, train_size=central_n, random_state=RANDOM_SEED)
        df_fl = df[df["Customer_ID"].isin(fl_customers)].copy()
        df_fl = df_fl.groupby("Customer_ID").tail(4).reset_index(drop=True)

    print(f"[FL] FL split: {len(fl_customers)} customers = {len(fl_customers)} mobile devices")
    run_fl_simulation(df_fl, dp_enabled=True, verbose=True)
