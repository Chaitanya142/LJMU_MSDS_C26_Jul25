"""
weight_validation.py
====================
Section 4.x — Validation of the Domain‑Driven Risk Weights

Runs three complementary checks:
  1. Logistic Regression  : coefficients should have the same sign as the
                            hand‑crafted FEATURE_WEIGHTS.
  2. XGBoost classification: gain‑based importances + SHAP values should rank
                             features consistently with the scorecard.
  3. Sensitivity analysis : perturbing every weight by ±20 % while preserving
                            sign and relative order should leave the 5‑class
                            distribution close to 12.5/25/25/25/12.5.

All artefacts (numeric tables + publication figures) are saved to
models/weight_validation/.

Usage
-----
  conda run -n smart_fund_advisor python -m src.weight_validation
  # or
  conda run -n smart_fund_advisor python src/weight_validation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                        # non‑interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import classification_report
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
import xgboost as xgb

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    RISK_CLASSES, RISK_FEATURES, RANDOM_SEED, CENTRAL_SPLIT, DEMO_SPLIT,
)
from src.preprocessing  import get_clean_customer_data
from src.risk_labeling  import assign_risk_label, FEATURE_WEIGHTS
from src.utils          import label_subplots

warnings.filterwarnings("ignore")

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = ROOT / "models" / "weight_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "figure.facecolor":  "white",
})

PALETTE_POS  = "#2196F3"   # blue  – positive weight
PALETTE_NEG  = "#F44336"   # red   – negative weight
PALETTE_RISK = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]  # V_L→V_H


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Data preparation (mirrors STEP 1 + STEP 2 of train.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _load_central_split() -> tuple[pd.DataFrame, list[str]]:
    """Return (df_central, feat_cols) using the same seed as train.py."""
    print("  [data]  Loading and preprocessing …")
    df = get_clean_customer_data(fit_scaler=True)
    df = assign_risk_label(df, fit_encoder=True)

    all_cust  = df["Customer_ID"].unique()
    rest, _   = train_test_split(all_cust,  test_size=DEMO_SPLIT,
                                 random_state=RANDOM_SEED)
    c_frac    = CENTRAL_SPLIT / (1.0 - DEMO_SPLIT)
    central_c, _ = train_test_split(rest, train_size=c_frac,
                                    random_state=RANDOM_SEED)

    df_central = df[df["Customer_ID"].isin(central_c)].copy()
    feat_cols  = [f for f in RISK_FEATURES if f in df_central.columns]
    print(f"  [data]  Central split: {df_central['Customer_ID'].nunique()} "
          f"customers, {len(feat_cols)} features")
    return df_central, feat_cols


def _get_XY(df: pd.DataFrame, feat_cols: list[str], seed: int = RANDOM_SEED):
    """Return (X_tr, X_val, y_tr, y_val) — same 80/20 stratified split."""
    le    = joblib.load(ROOT / "models" / "label_encoder.joblib")
    X_all = df[feat_cols].fillna(0).values.astype(np.float32)
    y_all = le.transform(df["risk_label"].values)

    return train_test_split(X_all, y_all,
                            test_size=0.20, stratify=y_all,
                            random_state=seed)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Logistic Regression — coefficient sign alignment
# ══════════════════════════════════════════════════════════════════════════════

def run_logistic_regression(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """
    Train multinomial LR, compare coefficient sign with FEATURE_WEIGHTS,
    produce a heatmap of per‑class coefficients.

    Returns a dict with alignment_pct, match_details, model, scaler.
    """
    print("\n" + "─" * 60)
    print("  CHECK 1 — Multinomial Logistic Regression")
    print("─" * 60)

    X_tr, X_val, y_tr, y_val = _get_XY(df, feat_cols)

    # Scale independently (LR is sensitive to feature magnitude)
    sc = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_tr)
    X_val_sc = sc.transform(X_val)

    lr = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=2000, C=1.0, random_state=RANDOM_SEED,
    )
    lr.fit(X_tr_sc, y_tr)
    val_acc = (lr.predict(X_val_sc) == y_val).mean()
    print(f"  Validation accuracy : {val_acc:.4f}")

    # Map encoded class indices back to RISK_CLASSES order
    le   = joblib.load(ROOT / "models" / "label_encoder.joblib")
    # lr.classes_ are the encoded integers (0..4) in the order the LR saw them
    # le.classes_ are the RISK_CLASS strings in alphabetical order
    # We need coef_ rows re-ordered by the RISK_CLASSES list (Very_Low→Low→…→Very_High)
    risk_to_enc  = {name: enc for enc, name in enumerate(le.classes_)}
    ordered_encs = [risk_to_enc[c] for c in RISK_CLASSES]
    # lr.classes_ maps position in coef_ to encoded int
    lr_class_to_row = {cls: idx for idx, cls in enumerate(lr.classes_)}
    row_order = [lr_class_to_row[enc] for enc in ordered_encs]
    coef_ordered = lr.coef_[row_order, :]

    print(classification_report(y_val, lr.predict(X_val_sc),
                                 target_names=le.classes_, zero_division=0))

    # coef_ shape: (n_classes, n_features) — rows in RISK_CLASSES order
    coef = pd.DataFrame(coef_ordered, index=RISK_CLASSES, columns=feat_cols)

    # ── sign alignment ───────────────────────────────────────────────────────
    # Strategy: for each feature, compute the "ordered-class coefficient trend"
    # (slope of a linear regression over class_index vs coefficient value).
    # A positive FEATURE_WEIGHT should produce an upward trend (Higher class →
    # larger coefficient), because Higher risk class = higher score.
    class_idx    = np.arange(len(RISK_CLASSES), dtype=float)
    trend_sign   = {}
    for feat in feat_cols:
        c_vals    = coef[feat].values
        slope     = np.polyfit(class_idx, c_vals, deg=1)[0]
        trend_sign[feat] = np.sign(slope)

    match_details = {}
    correct = 0
    print(f"\n  {'Feature':<32}  {'Design sign':>11}  {'LR trend sign':>13}  {'Match':>6}")
    print("  " + "─" * 66)
    for feat in feat_cols:
        design_sign = np.sign(FEATURE_WEIGHTS.get(feat, 0))
        lr_sign     = trend_sign[feat]
        match       = (design_sign == lr_sign)
        correct     += int(match)
        tick = "✓" if match else "✗"
        print(f"  {feat:<32}  {int(design_sign):>+11}  {int(lr_sign):>+13}  {tick:>6}")
        match_details[feat] = {
            "design_sign": int(design_sign),
            "lr_sign":     int(lr_sign),
            "match":       match,
        }

    alignment_pct = 100 * correct / len(feat_cols)
    print(f"\n  Sign alignment: {correct}/{len(feat_cols)}  ({alignment_pct:.1f} %)")

    # ── heatmap ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    import matplotlib.colors as mcolors
    vmax = np.abs(coef.values).max()
    im = ax.imshow(coef.values, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(RISK_CLASSES)))
    ax.set_yticklabels(RISK_CLASSES)
    ax.set_title("Multinomial Logistic Regression Coefficients\n"
                 "(red = positive  →  higher risk; blue = negative  →  lower risk)")
    plt.colorbar(im, ax=ax, pad=0.01)
    # Annotate each cell
    for i, rc in enumerate(RISK_CLASSES):
        for j, fc in enumerate(feat_cols):
            ax.text(j, i, f"{coef.loc[rc, fc]:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if abs(coef.loc[rc, fc]) > 0.5 * vmax else "black")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plot_lr_coef_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → models/weight_validation/plot_lr_coef_heatmap.png")

    # ── trend-sign comparison bar ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    design_vals = [FEATURE_WEIGHTS.get(f, 0) for f in feat_cols]
    lr_slopes   = []
    for feat in feat_cols:
        c_vals = coef[feat].values
        slope  = np.polyfit(class_idx, c_vals, deg=1)[0]
        lr_slopes.append(slope)

    colors_d = [PALETTE_POS if v >= 0 else PALETTE_NEG for v in design_vals]
    colors_s = [PALETTE_POS if v >= 0 else PALETTE_NEG for v in lr_slopes]

    axes[0].barh(feat_cols, design_vals, color=colors_d)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Domain‑Driven Scorecard Weights")
    axes[0].set_xlabel("Weight value")

    axes[1].barh(feat_cols, lr_slopes, color=colors_s)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("LR Coefficient Trend (slope across risk classes)")
    axes[1].set_xlabel("Slope (OvR logit per class‑index unit)")

    for ax in axes:
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Sign Alignment: Scorecard Weights vs. LR Coefficient Trends",
                 fontsize=13, fontweight="bold")
    label_subplots(axes)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plot_lr_sign_alignment.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → models/weight_validation/plot_lr_sign_alignment.png")

    return {
        "model":          lr,
        "scaler":         sc,
        "val_acc":        val_acc,
        "coef":           coef,
        "alignment_pct":  alignment_pct,
        "correct":        correct,
        "match_details":  match_details,
        "lr_slopes":      dict(zip(feat_cols, lr_slopes)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  XGBoost — feature importance (gain) + SHAP
# ══════════════════════════════════════════════════════════════════════════════

def run_xgboost_importance(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """
    Train XGBClassifier, compute gain‑based importances and SHAP values,
    compare ranking with FEATURE_WEIGHTS absolute magnitude ranking.
    """
    print("\n" + "─" * 60)
    print("  CHECK 2 — XGBoost Feature Importances + SHAP")
    print("─" * 60)

    X_tr, X_val, y_tr, y_val = _get_XY(df, feat_cols)

    # Pass DataFrames so XGBoost retains feature names
    X_tr_df  = pd.DataFrame(X_tr,  columns=feat_cols)
    X_val_df = pd.DataFrame(X_val, columns=feat_cols)

    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss",
        objective="multi:softprob", num_class=len(RISK_CLASSES),
        random_state=RANDOM_SEED, verbosity=0,
    )
    xgb_model.fit(X_tr_df, y_tr,
                  eval_set=[(X_val_df, y_val)],
                  verbose=False)
    val_acc = (xgb_model.predict(X_val_df) == y_val).mean()
    print(f"  Validation accuracy : {val_acc:.4f}")

    # ── gain importances ────────────────────────────────────────────────────
    gain_imp = xgb_model.get_booster().get_score(importance_type="gain")
    gain_series = pd.Series(gain_imp).reindex(feat_cols).fillna(0)
    gain_series = gain_series / gain_series.sum()   # normalise to [0,1]

    # ── SHAP ────────────────────────────────────────────────────────────────
    try:
        import shap
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X_val_df)       # list of arrays (one per class)
        # Mean absolute SHAP across classes and validation samples
        mean_shap  = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        shap_series = pd.Series(mean_shap, index=feat_cols)
        shap_series = shap_series / shap_series.sum()
        shap_available = True
    except ImportError:
        shap_series     = gain_series.copy()   # fallback = gain
        shap_available  = False
        print("  [SHAP] shap not installed — using gain importances as proxy")

    # ── Spearman rank correlation between importance and |design weight| ─────
    abs_weights = pd.Series({f: abs(FEATURE_WEIGHTS.get(f, 0)) for f in feat_cols})
    abs_weights = abs_weights / abs_weights.sum()

    from scipy.stats import spearmanr
    rho_gain,  p_gain  = spearmanr(abs_weights.loc[feat_cols],
                                    gain_series.loc[feat_cols])
    rho_shap,  p_shap  = spearmanr(abs_weights.loc[feat_cols],
                                    shap_series.loc[feat_cols])

    print(f"\n  Spearman ρ (|weight| vs gain importance) : {rho_gain:.3f}  (p={p_gain:.4f})")
    print(f"  Spearman ρ (|weight| vs SHAP importance) : {rho_shap:.3f}  (p={p_shap:.4f})")

    # ── printed ranking table ────────────────────────────────────────────────
    rank_tbl = pd.DataFrame({
        "Gain_rank":   gain_series.rank(ascending=False).fillna(len(feat_cols)),
        "SHAP_rank":   shap_series.rank(ascending=False).fillna(len(feat_cols)),
        "Design_rank": abs_weights.rank(ascending=False).fillna(len(feat_cols)),
        "Gain_imp":    gain_series.round(4),
        "SHAP_imp":    shap_series.round(4),
        "Design_|w|":  abs_weights.round(4),
    }).sort_values("SHAP_rank")
    rank_tbl["Gain_rank"]   = rank_tbl["Gain_rank"].astype(int)
    rank_tbl["SHAP_rank"]   = rank_tbl["SHAP_rank"].astype(int)
    rank_tbl["Design_rank"] = rank_tbl["Design_rank"].astype(int)
    print("\n" + rank_tbl.to_string())

    # ── save numeric table ───────────────────────────────────────────────────
    rank_tbl.to_csv(OUT_DIR / "xgb_importance_rank_table.csv")

    # ── combined bar chart ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ordered = shap_series.sort_values(ascending=False).index.tolist()

    def _bar(ax, series, title, palette_fn):
        vals  = series.loc[ordered].values
        cols  = [palette_fn(v, f) for v, f in zip(vals, ordered)]
        ax.barh(ordered, vals, color=cols)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    def _color_gain(v, f):   return "#1565C0"
    def _color_shap(v, f):   return "#4CAF50"
    def _color_design(v, f): return PALETTE_POS if FEATURE_WEIGHTS.get(f, 0) >= 0 else PALETTE_NEG

    _bar(axes[0], gain_series,  "XGBoost Gain Importance",      _color_gain)
    _bar(axes[1], shap_series,  "Mean |SHAP| (normalised)",     _color_shap)
    _bar(axes[2], abs_weights,  "Scorecard |Weight| (normalised)", _color_design)

    fig.suptitle("Feature Importance Ranking ─ XGBoost vs. Domain Scorecard",
                 fontsize=13, fontweight="bold")
    label_subplots(axes)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plot_xgb_importance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → models/weight_validation/plot_xgb_importance.png")

    # ── SHAP summary (dot) plot ──────────────────────────────────────────────
    if shap_available:
        try:
            import shap
            # Use class‑averaged absolute SHAP for a single beeswarm‑style summary
            fig, ax = plt.subplots(figsize=(9, 6))
            mean_abs = pd.Series(
                np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0),
                index=feat_cols,
            ).sort_values(ascending=True)
            colors = [PALETTE_POS if FEATURE_WEIGHTS.get(f, 0) >= 0 else PALETTE_NEG
                      for f in mean_abs.index]
            ax.barh(mean_abs.index, mean_abs.values, color=colors)
            ax.set_xlabel("Mean |SHAP value| (average across 5 risk classes)")
            ax.set_title("Global Feature Importance via SHAP\n"
                         "(blue = positive scorecard weight, red = negative)")
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()
            fig.savefig(OUT_DIR / "plot_shap_summary.png", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → models/weight_validation/plot_shap_summary.png")
        except Exception as e:
            print(f"  [SHAP] summary plot failed: {e}")

    return {
        "model":           xgb_model,
        "val_acc":         val_acc,
        "gain_series":     gain_series,
        "shap_series":     shap_series,
        "rank_tbl":        rank_tbl,
        "rho_gain":        rho_gain,
        "p_gain":          p_gain,
        "rho_shap":        rho_shap,
        "p_shap":          p_shap,
        "shap_available":  shap_available,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Sensitivity analysis — ±20 % weight perturbation
# ══════════════════════════════════════════════════════════════════════════════

TARGET_DIST = np.array([0.125, 0.25, 0.25, 0.25, 0.125])   # 12.5/25/25/25/12.5

def _compute_baseline_bins(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the score quantile boundary values under the original FEATURE_WEIGHTS.
    These fixed cut-points are then applied to all perturbed score vectors so
    that deviations from the 12.5/25/25/25/12.5 target are meaningful.
    """
    from src.risk_labeling import RISK_QUANTILES

    score = pd.Series(0.0, index=df.index)
    for feat, w in FEATURE_WEIGHTS.items():
        if feat in df.columns:
            score += w * df[feat].fillna(0)

    # These are the actual boundary values in score-space
    boundaries = score.quantile(RISK_QUANTILES[1:-1]).values
    return boundaries


def _perturbed_distribution(df: pd.DataFrame, feat_cols: list[str],
                             weights: dict[str, float],
                             fixed_bins: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the 5-class risk distribution under a perturbed weight vector.

    If fixed_bins is provided, use those score-space boundaries (derived from
    the baseline weights) so that any shift in the label distribution is
    attributable purely to the weight perturbation — not to quantile re-fitting.

    Returns fraction per class (ordered Very_Low … Very_High).
    """
    score = pd.Series(0.0, index=df.index)
    for feat, w in weights.items():
        if feat in df.columns:
            score += w * df[feat].fillna(0)

    if fixed_bins is not None:
        bins   = ([-np.inf] + list(fixed_bins) + [np.inf])
        labels = pd.cut(score, bins=bins, labels=RISK_CLASSES,
                        include_lowest=True).astype(str)
    else:
        from src.risk_labeling import RISK_QUANTILES
        labels = pd.qcut(score, q=RISK_QUANTILES, labels=RISK_CLASSES,
                         duplicates="drop").astype(str)

    counts = labels.value_counts()
    return np.array([counts.get(c, 0) for c in RISK_CLASSES]) / len(labels)


def run_sensitivity_analysis(df: pd.DataFrame, feat_cols: list[str],
                              n_trials: int = 500) -> dict:
    """
    Perturb every weight independently by U(−0.20, +0.20) × |weight|
    (sign-preserving) across n_trials random draws.
    Record the resulting 5-class distribution per trial.

    Also runs the "single-feature ±20 %" sweep to identify which feature
    has the largest distributional influence.
    """
    print("\n" + "─" * 60)
    print("  CHECK 3 — Sensitivity Analysis (±20 % weight perturbation)")
    print("─" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── A: Monte Carlo — all weights simultaneously ──────────────────────────
    print(f"  Monte Carlo: {n_trials} random perturbations (each weight ∈ [−20%, +20%]) …")
    # Fixed score-space thresholds from the ORIGINAL weights — applying these
    # to perturbed scores reveals genuine distributional shifts.
    fixed_bins = _compute_baseline_bins(df)
    base_dist = _perturbed_distribution(df, feat_cols, FEATURE_WEIGHTS, fixed_bins)
    print(f"  Baseline distribution : {np.round(base_dist * 100, 2)} %")

    trial_dists = []
    for _ in range(n_trials):
        perturbed = {}
        for feat, w in FEATURE_WEIGHTS.items():
            delta = rng.uniform(-0.20, 0.20) * abs(w)
            perturbed[feat] = w + (delta if w >= 0 else -delta)
        trial_dists.append(_perturbed_distribution(df, feat_cols, perturbed, fixed_bins))

    trial_arr = np.vstack(trial_dists)  # (n_trials, 5)

    max_deviation = np.abs(trial_arr - TARGET_DIST).max(axis=1)   # per trial
    mean_max_dev  = max_deviation.mean()
    p95_max_dev   = np.percentile(max_deviation, 95)

    print(f"  Mean max class‑fraction deviation from target : {mean_max_dev*100:.2f} pp")
    print(f"  95th‑pct max deviation                        : {p95_max_dev*100:.2f} pp")

    # ── B: single-feature ±20 % sweep ────────────────────────────────────────
    print("\n  Single-feature ±20 % sweep:")
    single_results = {}
    header = f"  {'Feature':<32}  {'−20% ΔMax':>10}  {'+20% ΔMax':>10}"
    print(header); print("  " + "─" * (len(header) - 2))
    for feat in feat_cols:
        w   = FEATURE_WEIGHTS.get(feat, 0)
        row = {}
        for sign_dir, multiplier in [("minus", -0.20), ("plus", +0.20)]:
            w_pert = dict(FEATURE_WEIGHTS)
            w_pert[feat] = w + multiplier * abs(w)
            d = _perturbed_distribution(df, feat_cols, w_pert, fixed_bins)
            delta_max = float(np.abs(d - base_dist).max())
            row[sign_dir] = delta_max
        single_results[feat] = row
        print(f"  {feat:<32}  {row['minus']*100:>9.2f} pp  {row['plus']*100:>9.2f} pp")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1) Distribution of max deviations
    ax = axes[0]
    ax.hist(max_deviation * 100, bins=30, color="#42A5F5", edgecolor="white")
    ax.axvline(mean_max_dev * 100, color="red",    linestyle="--",
               label=f"mean {mean_max_dev*100:.2f} pp")
    ax.axvline(p95_max_dev  * 100, color="orange", linestyle="--",
               label=f"p95  {p95_max_dev*100:.2f} pp")
    ax.set_xlabel("Max class‑fraction deviation from target (pp)")
    ax.set_ylabel("Count (out of 500 trials)")
    ax.set_title("Distributional Stability\nunder ±20 % Weight Perturbation")
    ax.legend(fontsize=9)

    # 2) Boxplot of class fractions across trials
    ax = axes[1]
    bp = ax.boxplot([trial_arr[:, i] * 100 for i in range(5)],
                    labels=RISK_CLASSES, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], PALETTE_RISK):
        patch.set_facecolor(c)
    ax.axhline(12.5, color="gray", linestyle=":", linewidth=1, label="12.5 % target")
    ax.axhline(25.0, color="gray", linestyle=":", linewidth=1, label="25.0 % target")
    ax.set_ylabel("Class fraction (%)")
    ax.set_title("Per-Class Distribution\nacross 500 Perturbation Trials")
    ax.legend(fontsize=8, loc="upper right")

    # 3) Single-feature sensitivity
    ax = axes[2]
    feat_sorted = sorted(single_results,
                         key=lambda f: max(single_results[f].values()), reverse=True)
    minus_vals = [single_results[f]["minus"] * 100 for f in feat_sorted]
    plus_vals  = [single_results[f]["plus"]  * 100 for f in feat_sorted]
    y_pos      = np.arange(len(feat_sorted))
    ax.barh(y_pos - 0.2, minus_vals, 0.4, label="−20 %", color="#EF5350")
    ax.barh(y_pos + 0.2, plus_vals,  0.4, label="+20 %", color="#42A5F5")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Max distribution shift (pp)")
    ax.set_title("Single-Feature Sensitivity\n(max shift from ±20 % change)")
    ax.legend(fontsize=9)

    fig.suptitle("Sensitivity Analysis — Risk Weight Perturbation Study",
                 fontsize=13, fontweight="bold")
    label_subplots(axes)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "plot_sensitivity_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → models/weight_validation/plot_sensitivity_analysis.png")

    return {
        "base_dist":       base_dist,
        "trial_arr":       trial_arr,
        "mean_max_dev":    mean_max_dev,
        "p95_max_dev":     p95_max_dev,
        "single_results":  single_results,
        "n_trials":        n_trials,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Print consolidated results table + save JSON summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(lr_res: dict, xgb_res: dict, sens_res: dict,
                   feat_cols: list[str]) -> None:
    import json

    print("\n" + "═" * 65)
    print("  VALIDATION SUMMARY — Section 4.x")
    print("═" * 65)

    print(f"\n  Check 1 — Logistic Regression")
    print(f"    Validation accuracy          : {lr_res['val_acc']:.4f}")
    print(f"    Sign alignment (all features): "
          f"{lr_res['correct']}/{len(feat_cols)}  "
          f"({lr_res['alignment_pct']:.1f} %)")
    mismatches = [f for f, d in lr_res["match_details"].items() if not d["match"]]
    if mismatches:
        print(f"    Mismatched features          : {', '.join(mismatches)}")
    else:
        print("    Mismatched features          : none")

    print(f"\n  Check 2 — XGBoost Feature Importance")
    print(f"    Validation accuracy          : {xgb_res['val_acc']:.4f}")
    print(f"    Spearman ρ (|w| vs gain imp) : {xgb_res['rho_gain']:.3f}  "
          f"p={xgb_res['p_gain']:.4f}")
    print(f"    Spearman ρ (|w| vs SHAP)     : {xgb_res['rho_shap']:.3f}  "
          f"p={xgb_res['p_shap']:.4f}")
    print(f"    SHAP available               : {xgb_res['shap_available']}")

    print(f"\n  Check 3 — Sensitivity Analysis ({sens_res['n_trials']} trials, ±20 %)")
    print(f"    Mean max class deviation     : {sens_res['mean_max_dev']*100:.2f} pp")
    print(f"    95th‑pct max deviation       : {sens_res['p95_max_dev']*100:.2f} pp")
    print(f"    Baseline distribution        : "
          + "  ".join(f"{c}={v*100:.1f}%" for c, v in
                      zip(RISK_CLASSES, sens_res["base_dist"])))

    # Save JSON
    summary = {
        "lr_val_acc":         round(lr_res["val_acc"], 4),
        "lr_sign_alignment":  f"{lr_res['correct']}/{len(feat_cols)}",
        "lr_alignment_pct":   round(lr_res["alignment_pct"], 1),
        "lr_mismatches":      mismatches,
        "xgb_val_acc":        round(xgb_res["val_acc"], 4),
        "rho_gain":           round(xgb_res["rho_gain"], 3),
        "p_gain":             round(xgb_res["p_gain"], 4),
        "rho_shap":           round(xgb_res["rho_shap"], 3),
        "p_shap":             round(xgb_res["p_shap"], 4),
        "shap_available":     xgb_res["shap_available"],
        "sens_n_trials":      sens_res["n_trials"],
        "sens_mean_max_dev_pp": round(sens_res["mean_max_dev"] * 100, 2),
        "sens_p95_max_dev_pp":  round(sens_res["p95_max_dev"]  * 100, 2),
        "baseline_dist_pct": {
            c: round(v * 100, 2)
            for c, v in zip(RISK_CLASSES, sens_res["base_dist"])
        },
    }
    out_json = OUT_DIR / "weight_validation_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  JSON summary → models/weight_validation/weight_validation_summary.json")

    print("\n" + "═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 65)
    print("  Section 4.x — Domain‑Driven Weight Validation")
    print("═" * 65)

    df_central, feat_cols = _load_central_split()

    lr_res   = run_logistic_regression(df_central, feat_cols)
    xgb_res  = run_xgboost_importance(df_central, feat_cols)
    sens_res = run_sensitivity_analysis(df_central, feat_cols, n_trials=500)

    _print_summary(lr_res, xgb_res, sens_res, feat_cols)


if __name__ == "__main__":
    main()
