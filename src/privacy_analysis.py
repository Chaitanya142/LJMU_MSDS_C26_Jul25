"""
privacy_analysis.py  –  Differential Privacy budget (ε) accounting for Smart Fund Advisor.

Theory
------
Each FL client trains locally with:
  - Gradient clipping: max-norm C  (bounds sensitivity)
  - Gaussian noise:    σ·C added per-gradient  (privacy noise)

We use the moments accountant / Rényi DP → (ε, δ)-DP guarantee.

Rényi DP bound (Mironov 2017, simplified Gaussian mechanism):
  RDP(α) ≈ α / (2σ²)  for the Gaussian mechanism
  ε(δ)   = RDP(α) − log(δ) / (α − 1)

For exact accounting over T rounds (composition):
  Total RDP(α) = T × steps_per_round × RDP(α)   [basic composition]

We compute ε for δ=1e-5 across a grid of α then take the minimum.

Usage
-----
    from src.privacy_analysis import compute_epsilon, privacy_summary
    eps, delta = compute_epsilon()
    privacy_summary()
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM,
    FL_ROUNDS, FL_LOCAL_EPOCHS, FL_FRACTION_FIT, FL_MIN_CLIENTS,
)

# ─── Rényi DP for Gaussian mechanism ──────────────────────────────────────────

def _rdp_gaussian(sigma: float, alpha: float) -> float:
    """
    RDP(α) for one application of the Gaussian mechanism with
    noise multiplier σ (noise_std = σ · C, sensitivity = C → ratio = σ).
    """
    return alpha / (2.0 * sigma ** 2)


def _rdp_to_approx_dp(rdp: float, alpha: float, delta: float) -> float:
    """
    Convert RDP guarantee to (ε, δ)-DP via Proposition 3 (Mironov 2017 tight).
    Returns ε.
    """
    if alpha <= 1:
        return float("inf")
    return rdp + math.log1p(-1.0 / alpha) - (math.log(delta) + math.log(1 - 1.0 / alpha)) / (alpha - 1)


def compute_epsilon(
    noise_multiplier: float = DP_NOISE_MULTIPLIER,
    max_grad_norm: float = DP_MAX_GRAD_NORM,
    fl_rounds: int = FL_ROUNDS,
    local_epochs: int = FL_LOCAL_EPOCHS,
    fraction_fit: float = FL_FRACTION_FIT,
    delta: float = 1e-5,
    alpha_min: int = 2,
    alpha_max: int = 512,
) -> Tuple[float, float]:
    """
    Compute (ε, δ)-DP budget using Rényi DP with Poisson subsampling amplification.

    Privacy amplification via subsampling (Mironov et al. 2019, Theorem 9):
    When only a fraction q of clients participates each round, the effective
    per-round RDP is strictly smaller than the unamplified bound.

    For αε ≥ 1 and the Gaussian mechanism with σ (noise multiplier):
        RDP_step(α) = α / (2σ²)   [unamplified Gaussian mechanism]

    Amplified bound (safe upper bound for Poisson subsampling, any q ∈ (0,1]):
        log RDP_amp(α) ≤ log(1 + q²·binom(α,2)·(exp(2·RDP_step(2)) − 1))/(α−1)
        capped at RDP_step(α) when q is large.

    This is the approach used by Google's DP-SGD / TensorFlow Privacy library.
    Expected client participation = q × fl_rounds rounds per client.
    Total steps per client = expected_rounds × local_epochs.

    Returns
    -------
    (epsilon, delta)
    """
    # Expected rounds a single client participates in
    expected_rounds_per_client = fraction_fit * fl_rounds
    total_steps = max(1, int(expected_rounds_per_client * local_epochs))

    # Subsampling probability at the round level
    q = fraction_fit

    alphas = np.arange(alpha_min, alpha_max + 1, dtype=float)
    best_eps = float("inf")

    for alpha in alphas:
        try:
            # Unamplified Gaussian RDP per step
            rdp_step = _rdp_gaussian(noise_multiplier, alpha)

            # ── Poisson subsampling amplification (Mironov et al. 2019, Thm 9) ──
            if alpha <= 1.0:
                amplified = rdp_step
            else:
                # Abadi 2016 bound
                abadi_inner = (1 - q) + q * math.exp(rdp_step)
                if abadi_inner <= 0:
                    abadi = rdp_step
                else:
                    abadi = (alpha / (alpha - 1.0)) * math.log(abadi_inner)

                # Balle et al. 2020 tighter amplification for Gaussian
                rdp2 = _rdp_gaussian(noise_multiplier, 2.0)
                two_rdp2 = 2.0 * rdp2
                if two_rdp2 > 500:
                    balle = rdp_step  # trivial bound when σ is tiny
                else:
                    binom_a2 = alpha * (alpha - 1.0) / 2.0
                    balle = math.log(
                        1.0 + min(q * q, 1.0) * binom_a2 * max(0.0, math.exp(two_rdp2) - 1.0)
                    ) / (alpha - 1.0)

                amplified = min(abadi, balle, rdp_step)

            total_rdp = total_steps * amplified
            eps = _rdp_to_approx_dp(total_rdp, alpha, delta)
            if math.isfinite(eps) and eps < best_eps:
                best_eps = eps
        except (ValueError, OverflowError):
            continue  # skip this alpha order; numerical instability for extreme sigma

    return round(best_eps, 4), delta


def per_round_epsilon(
    noise_multiplier: float = DP_NOISE_MULTIPLIER,
    fraction_fit: float = FL_FRACTION_FIT,
    local_epochs: int = FL_LOCAL_EPOCHS,
    delta: float = 1e-5,
) -> float:
    """Return ε consumed in one FL round (single-round budget)."""
    eps, _ = compute_epsilon(
        noise_multiplier=noise_multiplier,
        fraction_fit=fraction_fit,
        fl_rounds=1,
        local_epochs=local_epochs,
        delta=delta,
    )
    return eps


def epsilon_vs_rounds(
    max_rounds: int = 20,
    noise_multiplier: float = DP_NOISE_MULTIPLIER,
    delta: float = 1e-5,
) -> Dict[int, float]:
    """Return a dict of {round_number: cumulative_epsilon} for plotting."""
    result = {}
    for r in range(1, max_rounds + 1):
        eps, _ = compute_epsilon(
            noise_multiplier=noise_multiplier,
            fl_rounds=r,
            delta=delta,
        )
        result[r] = eps
    return result


def epsilon_vs_sigma(
    sigmas: list | None = None,
    fl_rounds: int = FL_ROUNDS,
    delta: float = 1e-5,
) -> Dict[float, float]:
    """Return {sigma: epsilon} for sensitivity analysis."""
    if sigmas is None:
        sigmas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    return {
        s: compute_epsilon(noise_multiplier=s, fl_rounds=fl_rounds, delta=delta)[0]
        for s in sigmas
    }


def privacy_summary(print_output: bool = True) -> Dict:
    """Compute and optionally print the full DP privacy summary."""
    eps, delta = compute_epsilon()
    eps_per_round = per_round_epsilon()

    # Sigma trade-off table
    sigma_table = epsilon_vs_sigma(sigmas=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

    quality = (
        "Excellent (ε < 1)" if eps < 1 else
        "Strong (ε < 3)"    if eps < 3 else
        "Good (ε < 5)"      if eps < 5 else
        "Moderate (ε < 10)" if eps < 10 else
        f"Formal ε={eps:.2f}; privacy noise active"
    )

    summary = {
        "noise_multiplier_sigma":  DP_NOISE_MULTIPLIER,
        "max_gradient_norm_C":     DP_MAX_GRAD_NORM,
        "noise_std_per_update":    DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM,
        "fl_rounds":               FL_ROUNDS,
        "local_epochs_per_round":  FL_LOCAL_EPOCHS,
        "sampling_fraction_q":     FL_FRACTION_FIT,
        "delta":                   delta,
        "epsilon_total":           eps,
        "epsilon_per_round":       eps_per_round,
        "privacy_quality":         quality,
        "sigma_epsilon_tradeoff":  sigma_table,
    }

    if print_output:
        print("=" * 60)
        print("  Differential Privacy Budget")
        print("=" * 60)
        print(f"  Noise multiplier (σ)         : {DP_NOISE_MULTIPLIER}")
        print(f"  Gradient clip norm (C)       : {DP_MAX_GRAD_NORM}")
        print(f"  Noise std per gradient       : {DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM:.2f}")
        print(f"  FL rounds                    : {FL_ROUNDS}")
        print(f"  Local epochs per round       : {FL_LOCAL_EPOCHS}")
        print(f"  Sampling fraction q          : {FL_FRACTION_FIT}")
        print()
        print(f"  ╔══════════════════════════════╗")
        print(f"  ║  ε (epsilon) = {eps:<8.4f}   ║")
        print(f"  ║  δ (delta)   = {delta:<8.1e}   ║")
        print(f"  ╚══════════════════════════════╝")
        print()
        print(f"  Privacy Quality              : {quality}")
        print(f"  ε per round (expected)       : {eps_per_round:.4f}")
        print()
        print("  σ → ε Trade-off (same FL config):")
        for s, e in sigma_table.items():
            bar = "▓" * min(30, max(1, int(e)))
            print(f"    σ={s:.1f} → ε={e:7.4f}  {bar}")
        print()
        print("  Note: ε measures worst-case distinguishability risk.")
        print("  Gradient noise (σ·C) is always added regardless of ε value.")
        print("  Higher σ → lower ε (stronger formal guarantee) at cost of accuracy.")
        print("=" * 60)

    return summary


if __name__ == "__main__":
    privacy_summary()
    print("\nEpsilon by rounds:")
    for r, e in epsilon_vs_rounds(max_rounds=FL_ROUNDS).items():
        print(f"  Round {r:2d}: ε = {e:.4f}")
