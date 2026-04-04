"""
utils.py  –  Shared utilities: plotting, metrics, reproducibility.
"""

from __future__ import annotations

import random
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import string
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
from config import RISK_CLASSES, MODELS_DIR, RANDOM_SEED, LABEL_ENCODER_PATH


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── Subplot labelling helper ─────────────────────────────────────────────────

def label_subplots(axes, labels=None, fontsize=12, x=0.5, y=-0.20,
                   fontweight="bold", color="black") -> None:
    """Add (A), (B), (C)… labels below the x-axis tick labels of each panel.

    Parameters
    ----------
    axes      : single Axes, list/array of Axes, or 2-D array (nrows×ncols)
    labels    : optional custom list of label strings; defaults to A, B, C …
    fontsize  : font size for the label
    x, y      : axes-fraction position; y=-0.20 clears x-tick labels (~-0.08
                to -0.14) and places the panel letter in the whitespace below.
    """
    import numpy as _np
    ax_flat = _np.array(axes).flatten()
    n = len(ax_flat)
    if labels is None:
        labels = [f"({string.ascii_uppercase[i]})" for i in range(n)]
    for ax, lbl in zip(ax_flat, labels):
        ax.text(x, y, lbl,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=fontsize, fontweight=fontweight, color=color,
                clip_on=False)   # clip_on=False: render even outside axes bbox
    # Reserve bottom padding so panel letters clear the x-axis tick labels.
    try:
        fig = ax_flat[0].get_figure()
        if fig.subplotpars.bottom < 0.15:
            fig.subplots_adjust(bottom=0.15)
    except Exception:
        pass




def plot_training_history(history: Dict, save_path: Path | str | None = None) -> None:
    """Plot train/val loss and val accuracy side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(history["val_loss"],   label="Val Loss",   color="coral")
    ax.set_title("Loss per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(history["val_acc"], label="Val Accuracy", color="seagreen")
    ax.set_title("Validation Accuracy per Epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.axhline(max(history["val_acc"]), linestyle="--", color="gray", alpha=0.5,
               label=f"Best: {max(history['val_acc']):.3f}")
    ax.legend(); ax.grid(alpha=0.3)

    label_subplots(axes)

    labels = ['(A) Loss per Epoch','(B) Validation Accuracy per Epoch']
    line1 = " | ".join(labels[:2])   # first 3 panels

    plt.suptitle("Training Convergence MLP\n" + line1 ,
             fontsize=13, y=1.02, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history plot → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: Path | str | None = None) -> None:
    # Load le.classes_ so axis labels match encoded integer order (alphabetical):
    # 0=High, 1=Low, 2=Medium, 3=Very_High, 4=Very_Low
    le = joblib.load(LABEL_ENCODER_PATH)
    class_labels = list(le.classes_)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title("Confusion Matrix — Risk Appetite")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix → {save_path}")
    plt.show()


def plot_risk_distribution(labels: np.ndarray,
                           title: str = "Risk Appetite Distribution",
                           save_path: Path | str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(labels, return_counts=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    bars = ax.bar(unique.astype(str), counts, color=colors[:len(unique)])
    ax.set_title(title)
    ax.set_xlabel("Risk Class"); ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved distribution plot → {save_path}")
    plt.show()


def plot_fl_history(fl_history: Dict,
                    save_path: Path | str | None = None) -> None:
    """Plot distributed loss (and accuracy if available) across FL rounds."""
    rounds = sorted(fl_history["distributed_loss"].keys(), key=int)
    losses = [fl_history["distributed_loss"][r] for r in rounds]
    round_ints = [int(r) for r in rounds]

    has_acc = "round_accuracy" in fl_history and fl_history["round_accuracy"]

    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14 if has_acc else 8, 4))
    ax_loss = axes[0] if has_acc else axes

    ax_loss.plot(round_ints, losses, marker="o", color="darkorange", linewidth=2)
    ax_loss.set_title("FL — Avg Distributed Loss per Round")
    ax_loss.set_xlabel("FL Round"); ax_loss.set_ylabel("Avg. Loss")
    ax_loss.grid(alpha=0.3)

    if has_acc:
        accs = [fl_history["round_accuracy"][r] for r in rounds]
        axes[1].plot(round_ints, [a * 100 for a in accs],
                     marker="s", color="steelblue", linewidth=2)
        axes[1].set_title("FL — Round Accuracy (sampled clients)")
        axes[1].set_xlabel("FL Round"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(80, 100)
        axes[1].grid(alpha=0.3)


    labels = ['(A) FL — Avg Distributed Loss per Round','(B) FL — Round Accuracy (sampled clients)']
    line1 = " | ".join(labels[:2])   # first 3 panels

    fig.suptitle("FL Loss Stability\n" + line1 ,
             fontsize=13, y=1.08, fontweight='bold')
    
    if has_acc:
        label_subplots(axes)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved FL history plot → {save_path}")
    plt.show()


def plot_feature_importance(weights_dict: Dict[str, float],
                            save_path: Path | str | None = None) -> None:
    """Bar chart of domain-weight magnitude (proxy for feature importance)."""
    features = list(weights_dict.keys())
    values   = [abs(v) for v in weights_dict.values()]
    colors   = ["#4CAF50" if weights_dict[f] > 0 else "#F44336" for f in features]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(features, values, color=colors)
    ax.set_title("Feature Importance (Domain Weight Magnitude)\n"
                 "Green = raises risk appetite | Red = lowers risk appetite")
    ax.set_xlabel("Absolute Weight")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance plot → {save_path}")
    plt.show()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def print_full_report(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      label_names: List[str] | None = None) -> None:
    print("\n" + "="*65)
    print(" CLASSIFICATION REPORT — Risk Appetite (5 Classes)")
    print("="*65)
    # Use le.classes_ by default: LabelEncoder sorts alphabetically so
    # le.classes_[i] correctly maps encoded integer i → class name.
    # Passing RISK_CLASSES (semantic order) here was a bug.
    if label_names is None:
        label_names = list(joblib.load(LABEL_ENCODER_PATH).classes_)
    # Map integer labels to names if needed
    if y_true.dtype in [np.int32, np.int64, int]:
        target_names = [label_names[i] for i in sorted(set(y_true) | set(y_pred))]
    else:
        target_names = label_names
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def load_json(path: Path | str) -> Dict:
    with open(path) as f:
        return json.load(f)
