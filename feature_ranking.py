"""
feature_ranking.py — Feature importance ranking for the bank user dataset
=========================================================================

Three complementary lenses are applied to the 15 engineered risk features:

    1. PCA         — variance-weighted absolute loadings on the first K components
    2. NMF         — Non-negative Matrix Factorization; each feature's mean
                     participation across latent factors (H-matrix row norms)
    3. Statistical — Mutual Information + |Spearman ρ| with the risk score

A Hybrid score = weighted ensemble of all four metrics after [0,1] normalisation.

Run
---
    conda run -n smart_fund_advisor python feature_ranking.py
    conda run -n smart_fund_advisor python feature_ranking.py --top 10 --plot

Output
------
    • Console : ranked table (all 15 features)
    • File    : models/feature_ranking.csv
    • Plot    : models/feature_ranking.png  (if --plot flag used)
"""

from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── project root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import get_clean_customer_data
from src.risk_labeling import compute_risk_score, assign_risk_label
from config import RISK_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Weights for the hybrid ensemble (must sum to 1.0)
# PCA gets the highest weight: empirically captures the most variance structure.
# NMF slightly lower (assumes non-negativity, loses signed information).
# MI and Spearman are complementary statistical anchors.
# ─────────────────────────────────────────────────────────────────────────────
HYBRID_WEIGHTS = {
    "pca":       0.35,   # variance-explained weighted loadings
    "nmf":       0.25,   # non-negative factorisation participation
    "mi":        0.25,   # mutual information with risk score
    "spearman":  0.15,   # |Spearman ρ| with risk score
}
assert abs(sum(HYBRID_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

PCA_N_COMPONENTS = 8   # capture ≥ 95% variance across 15 features
NMF_N_COMPONENTS = 5   # mirror 5 risk classes (latent risk dimensions)
NMF_MAX_ITER     = 500


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, pd.Series]:
    """
    Run preprocessing pipeline and return (X, risk_score).

    X           : DataFrame[RISK_FEATURES], shape (N, 15), scaled to [0,1]
    risk_score  : continuous composite score (Σ weight × feature)
    """
    print("[1/5] Running preprocessing pipeline …")
    df = get_clean_customer_data(fit_scaler=True)

    # Compute continuous risk score (used as the regression target for MI/Spearman)
    df["risk_score"] = compute_risk_score(df)

    # Keep only the 15 canonical risk features
    available = [f for f in RISK_FEATURES if f in df.columns]
    missing   = set(RISK_FEATURES) - set(available)
    if missing:
        print(f"  ⚠ Missing features (will be excluded): {missing}")

    X = df[available].fillna(0.0)
    y = df["risk_score"]

    print(f"  ✓ {X.shape[0]:,} customers × {X.shape[1]} features")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. PCA ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_pca(X: pd.DataFrame, n_components: int = PCA_N_COMPONENTS) -> pd.Series:
    """
    Rank features by their contribution to the top `n_components` principal
    components, weighted by the explained variance ratio of each component.

    Score_i = Σ_k  explained_variance_ratio_k × |loading_ik|

    This captures how much of the total data variance is "carried" by each
    feature — a high score means the feature co-varies strongly with the
    directions that explain the most customer variance.

    Reference
    ---------
    Jolliffe (2002) *Principal Component Analysis* (2nd ed.) — Section 10.2:
    "Variable selection by PCA loadings."
    """
    print("[2/5] PCA-based ranking …")
    n_components = min(n_components, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X.values)

    # Shape: (n_components, n_features)
    loadings = np.abs(pca.components_)
    # Weight each component's loadings by its explained variance share
    evr = pca.explained_variance_ratio_             # shape (n_components,)
    weighted_loadings = (evr[:, None] * loadings)   # broadcast → (n_comp, n_feat)
    scores = weighted_loadings.sum(axis=0)           # aggregate → (n_feat,)

    # ---- diagnostics
    cumvar = evr.cumsum()[-1] * 100
    print(f"  PCA: {n_components} components explain {cumvar:.1f}% of variance")
    top3 = pd.Series(scores, index=X.columns).nlargest(3)
    print(f"  Top-3 by PCA: {top3.to_dict()}")

    return pd.Series(scores, index=X.columns, name="pca_raw")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NMF ranking (Matrix Factorisation)
# ─────────────────────────────────────────────────────────────────────────────

def rank_nmf(X: pd.DataFrame, n_components: int = NMF_N_COMPONENTS) -> pd.Series:
    """
    Rank features using Non-negative Matrix Factorisation (NMF).

    NMF decomposes  X ≈ W · H  where:
        W : (n_customers, n_components)  — customer factor loadings
        H : (n_components, n_features)   — feature participation per factor

    Feature importance = L2-norm of each column of H (a feature's total
    participation across all latent factors).

    Why NMF over SVD here?
    ----------------------
    • Features are already in [0,1] (non-negative) after MinMax scaling.
    • NMF forces a parts-based, additive representation — interpretable as
      "which features build up each risk prototype".
    • Aligns naturally with 5 risk classes (n_components = 5).

    Reference
    ---------
    Lee & Seung (1999) "Learning the parts of objects by non-negative matrix
    factorization," *Nature* 401, 788–791.
    Cai et al. (2011) "Graph regularized NMF," IEEE TPAMI.
    """
    print("[3/5] NMF-based ranking (Matrix Factorisation) …")
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",          # deterministic init; better than random for tabular
        max_iter=NMF_MAX_ITER,
        random_state=42,
        l1_ratio=0.0,            # pure Frobenius loss (no sparsity on H)
    )
    nmf.fit(X.values)

    # H shape: (n_components, n_features)
    H = nmf.components_
    # Column-wise L2 norm → scalar importance per feature
    scores = np.linalg.norm(H, axis=0)

    # ---- diagnostics: reconstruction error
    W   = nmf.transform(X.values)
    rec = np.linalg.norm(X.values - W @ H, "fro") / np.linalg.norm(X.values, "fro")
    print(f"  NMF: {n_components} components, relative reconstruction error = {rec:.4f}")
    top3 = pd.Series(scores, index=X.columns).nlargest(3)
    print(f"  Top-3 by NMF: {top3.to_dict()}")

    return pd.Series(scores, index=X.columns, name="nmf_raw")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical ranking : Mutual Information + |Spearman ρ|
# ─────────────────────────────────────────────────────────────────────────────

def rank_statistical(X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Two supervised measures against the continuous risk score:

    MI (Mutual Information)
    -----------------------
    Estimates how much knowing feature X_i reduces uncertainty in risk_score.
    Non-parametric (k-NN based estimator), captures non-linear dependencies.
    Ref: Kraskov et al. (2004) "Estimating mutual information," PRE 69.

    |Spearman ρ|
    ------------
    Rank-order correlation; robust to outliers (no normality assumption).
    Captures monotonic (possibly non-linear) feature–risk relationship.
    Ref: Spearman (1904) "The proof and measurement of association between
    two things," *American Journal of Psychology* 15.
    """
    print("[4/5] Statistical ranking (MI + Spearman) …")
    mi_scores = mutual_info_regression(
        X.values, y.values, random_state=42, n_neighbors=5
    )

    sp_scores = []
    for col in X.columns:
        rho, _ = spearmanr(X[col], y)
        sp_scores.append(abs(rho))

    mi_series = pd.Series(mi_scores, index=X.columns, name="mi_raw")
    sp_series = pd.Series(sp_scores, index=X.columns, name="spearman_raw")

    top3_mi = mi_series.nlargest(3)
    top3_sp = sp_series.nlargest(3)
    print(f"  Top-3 by MI:       {top3_mi.to_dict()}")
    print(f"  Top-3 by Spearman: {top3_sp.to_dict()}")

    return mi_series, sp_series


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hybrid filtering
# ─────────────────────────────────────────────────────────────────────────────

def _minmax_norm(s: pd.Series) -> pd.Series:
    """Normalise a series to [0, 1]; handle constant series safely."""
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def hybrid_rank(
    pca_raw:  pd.Series,
    nmf_raw:  pd.Series,
    mi_raw:   pd.Series,
    sp_raw:   pd.Series,
) -> pd.DataFrame:
    """
    Hybrid Filter = weighted normalised ensemble.

    Each raw score is normalised to [0,1] before combination so that
    differences in scale (PCA loadings vs MI in nats vs ρ in [-1,1]) do not
    bias the ensemble.

    Hybrid_i = w_pca  × norm(PCA_i)
             + w_nmf  × norm(NMF_i)
             + w_mi   × norm(MI_i)
             + w_sp   × norm(|ρ|_i)

    The weights in HYBRID_WEIGHTS reflect:
      • PCA (0.35)  — captures unsupervised multivariate structure
      • NMF (0.25)  — additive parts-based factorisation
      • MI  (0.25)  — non-linear supervised signal
      • ρ   (0.15)  — monotonic supervised signal (complementary to MI)
    """
    print("[5/5] Computing hybrid score …")

    pca_n = _minmax_norm(pca_raw)
    nmf_n = _minmax_norm(nmf_raw)
    mi_n  = _minmax_norm(mi_raw)
    sp_n  = _minmax_norm(sp_raw)

    hybrid = (
        HYBRID_WEIGHTS["pca"]      * pca_n
        + HYBRID_WEIGHTS["nmf"]    * nmf_n
        + HYBRID_WEIGHTS["mi"]     * mi_n
        + HYBRID_WEIGHTS["spearman"] * sp_n
    )

    df = pd.DataFrame({
        "Feature":          hybrid.index,
        "PCA_score":        pca_n.values,
        "NMF_score":        nmf_n.values,
        "MI_score":         mi_n.values,
        "Spearman_score":   sp_n.values,
        "Hybrid_score":     hybrid.values,
        # Raw scores (unscaled) for reference
        "PCA_raw":          pca_raw.values,
        "NMF_raw":          nmf_raw.values,
        "MI_raw":           mi_raw.values,
        "Spearman_raw":     sp_raw.values,
    })

    df = df.sort_values("Hybrid_score", ascending=False).reset_index(drop=True)
    df.index += 1                              # 1-based rank
    df.index.name = "Rank"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Display & persist
# ─────────────────────────────────────────────────────────────────────────────

def display_table(df: pd.DataFrame, top: int | None = None) -> None:
    view = df.head(top) if top else df
    # Format for console
    formatted = view[[
        "Feature", "Hybrid_score", "PCA_score", "NMF_score",
        "MI_score", "Spearman_score"
    ]].copy()
    for col in ["Hybrid_score", "PCA_score", "NMF_score", "MI_score", "Spearman_score"]:
        formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "=" * 78)
    print("  Feature Ranking — Smart Fund Advisor (bank_user_dataset)")
    print("=" * 78)
    header = f"{'Rank':>4}  {'Feature':<34} {'Hybrid':>8} {'PCA':>8} {'NMF':>8} {'MI':>8} {'Spear':>8}"
    print(header)
    print("-" * 78)
    for rank, row in formatted.iterrows():
        print(
            f"{rank:>4}  {row['Feature']:<34} "
            f"{row['Hybrid_score']:>8} {row['PCA_score']:>8} "
            f"{row['NMF_score']:>8} {row['MI_score']:>8} {row['Spearman_score']:>8}"
        )
    print("=" * 78)
    print(
        f"\n  Hybrid = "
        f"{HYBRID_WEIGHTS['pca']}×PCA + {HYBRID_WEIGHTS['nmf']}×NMF + "
        f"{HYBRID_WEIGHTS['mi']}×MI + {HYBRID_WEIGHTS['spearman']}×|Spearman|\n"
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"  ✓ Saved ranking CSV → {path}")


def plot_ranking(df: pd.DataFrame, path: Path, top: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    view = df.head(top).iloc[::-1]   # reverse so rank-1 is at top of horizontal bar
    features = view["Feature"].tolist()
    y_pos    = np.arange(len(features))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Feature Ranking — Smart Fund Advisor\n"
        "(PCA · NMF Matrix Factorisation · MI · Spearman — Hybrid Ensemble)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Left panel: stacked contribution bar chart ──────────────────────────
    ax = axes[0]
    w_pca = HYBRID_WEIGHTS["pca"]
    w_nmf = HYBRID_WEIGHTS["nmf"]
    w_mi  = HYBRID_WEIGHTS["mi"]
    w_sp  = HYBRID_WEIGHTS["spearman"]

    pca_contrib = view["PCA_score"].values * w_pca
    nmf_contrib = view["NMF_score"].values * w_nmf
    mi_contrib  = view["MI_score"].values  * w_mi
    sp_contrib  = view["Spearman_score"].values * w_sp

    colors = ["#2563EB", "#7C3AED", "#059669", "#D97706"]
    ax.barh(y_pos, pca_contrib,                                 color=colors[0], label="PCA")
    ax.barh(y_pos, nmf_contrib, left=pca_contrib,               color=colors[1], label="NMF")
    ax.barh(y_pos, mi_contrib,  left=pca_contrib+nmf_contrib,   color=colors[2], label="MI")
    ax.barh(y_pos, sp_contrib,  left=pca_contrib+nmf_contrib+mi_contrib, color=colors[3], label="Spearman")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Contribution to Hybrid Score")
    ax.set_title("(A) Stacked Contribution per Method", fontsize=10, fontweight="bold", y=-0.17)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # ── Right panel: dot-plot comparing all 4 normalised scores ─────────────
    ax2 = axes[1]
    markers   = ["o", "s", "D", "^"]
    score_cols = ["PCA_score", "NMF_score", "MI_score", "Spearman_score"]
    labels     = ["PCA", "NMF", "MI", "Spearman"]
    for mc, col, lab, clr in zip(markers, score_cols, labels, colors):
        ax2.scatter(view[col].values, y_pos, marker=mc, color=clr, label=lab,
                    s=55, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features, fontsize=9)
    ax2.set_xlabel("Normalised Score [0, 1]")
    ax2.set_title("(B) Per-Method Normalised Score", fontsize=10, fontweight="bold", y=-0.17)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlim(-0.05, 1.10)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.grid(axis="y", linestyle=":", alpha=0.2)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved ranking plot → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(top: int | None = None, plot: bool = False) -> pd.DataFrame:
    X, y = load_features()

    pca_raw          = rank_pca(X)
    nmf_raw          = rank_nmf(X)
    mi_raw, sp_raw   = rank_statistical(X, y)

    ranking = hybrid_rank(pca_raw, nmf_raw, mi_raw, sp_raw)

    display_table(ranking, top=top)

    out_csv = ROOT / "models" / "feature_ranking.csv"
    save_csv(ranking, out_csv)

    if plot:
        out_png = ROOT / "models" / "feature_ranking.png"
        plot_ranking(ranking, out_png)

    return ranking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature ranking via PCA, NMF Matrix Factorisation, and Hybrid Filter"
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only top-N features in console (default: all 15)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a ranking visualisation to models/feature_ranking.png"
    )
    args = parser.parse_args()
    main(top=args.top, plot=args.plot)
