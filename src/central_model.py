"""
central_model.py  –  Multi-Layer Perceptron for 5-class risk appetite classification.

Architecture (v2)
-----------------
Input → FC(256)→BN→GELU→Dropout → Residual[FC(128)→BN→GELU→Dropout]
      → FC(64)→BN→GELU → FC(32)→BN→GELU → FC(5)

Improvements over v1
--------------------
 • Focal Loss for hard-example mining: down-weights easy samples, boosts
   learning on confused minority classes (Very_Low / Very_High).
 • Label smoothing (α=0.05): regularises overconfident predictions.
 • Early stopping with patience: prevents overfitting, stops when val_acc
   plateaus — more production-realistic than fixed epoch count.
 • GELU activation: smoother gradient landscape than ReLU.
 • Residual connection on 2nd block (256→128): stable gradient flow.
 • OneCycleLR scheduler: super-convergence for faster training.

Trained on 65 % of unique customers (central server data; 30 % FL, 5 % demo holdout).
The saved weights become the starting point for federated learning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from pathlib import Path
from typing import List, Tuple, Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    HIDDEN_DIMS, DROPOUT, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    N_RISK_CLASSES, RISK_FEATURES, RISK_CLASSES,
    CENTRAL_MODEL_PATH, MODELS_DIR, RANDOM_SEED, LABEL_ENCODER_PATH,
    LABEL_SMOOTHING, FOCAL_LOSS_GAMMA, EARLY_STOP_PATIENCE,
)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — down-weights well-classified examples.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    When γ=0 this reduces to standard CrossEntropy.
    When γ>0 the loss for confident predictions is reduced,
    forcing the model to focus on hard/misclassified samples.
    This is critical for the tail classes (Very_Low / Very_High ≈ 12.5% each).
    """

    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)          # p_t = probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ─── Model definition ─────────────────────────────────────────────────────────

class RiskMLP(nn.Module):
    """
    Feed-forward MLP with optional residual connection and GELU activation.
    Architecture: Input → [FC→BN→GELU→Dropout] × L → FC(n_classes)
    A skip-connection from input is added at the 2nd hidden block for
    stable gradient flow in deeper configurations ([256,128,64,32]).
    """

    def __init__(self,
                 input_dim: int = len(RISK_FEATURES),
                 hidden_dims: List[int] = HIDDEN_DIMS,
                 n_classes: int = N_RISK_CLASSES,
                 dropout: float = DROPOUT):
        super().__init__()
        self.use_residual = len(hidden_dims) >= 2

        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

        # Residual projection: input → hidden_dims[1] (skip first block)
        if self.use_residual:
            self._res_proj = nn.Linear(input_dim, hidden_dims[1])
        else:
            self._res_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual and self._res_proj is not None:
            # Forward through first block (layers 0-3: Linear+BN+GELU+Dropout)
            h = self.net[:4](x)
            # Forward through second block (layers 4-7)
            h = self.net[4:8](h)
            # Add residual from input (projected to match dimensionality)
            h = h + self._res_proj(x)
            # Forward through remaining layers
            h = self.net[8:](h)
            return h
        return self.net(x)

    def get_parameters(self) -> List[np.ndarray]:
        """Return full state_dict (weights + BatchNorm buffers) as numpy arrays."""
        return [v.detach().cpu().numpy() for v in self.state_dict().values()]

    def set_parameters(self, params: List[np.ndarray]) -> None:
        """Restore full state_dict from numpy arrays."""
        state_dict = self.state_dict()
        new_state = {
            k: torch.tensor(arr, dtype=v.dtype)
            for (k, v), arr in zip(state_dict.items(), params)
        }
        self.load_state_dict(new_state, strict=True)


# ─── Training utilities ───────────────────────────────────────────────────────

def build_tensors(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def train_one_epoch(model: RiskMLP,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    scheduler=None) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: RiskMLP,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ─── Central training pipeline ────────────────────────────────────────────────

def train_central_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    verbose: bool = True,
) -> Tuple[RiskMLP, Dict]:
    """
    Train RiskMLP on central server data (65 % of customers, 3-way split).

    Improvements over v1:
      • Focal Loss (γ=2.0) for hard-example mining on tail classes
      • Label smoothing (α=0.05) to regularise overconfident predictions
      • Early stopping (patience=8) to prevent overfitting
      • OneCycleLR for super-convergence
      • Gradient clipping (max_norm=5) for training stability

    Returns
    -------
    model   : trained RiskMLP (in eval mode, moved to CPU)
    history : dict with lists for train_loss, val_loss, val_acc
    """
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[Central] Training on device: {device}")
        print(f"[Central] Features: {X_train.shape[1]} | "
              f"Loss: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING}) | "
              f"Early stop patience={EARLY_STOP_PATIENCE}")

    # Class-weighted CrossEntropyLoss with label smoothing.
    # Previously FocalLoss(γ=2.0); changed to CrossEntropyLoss because:
    #  - PCA-updated weights already encode class importance at the data level
    #  - CrossEntropy is the canonical multi-class loss; fits bell-curve label
    #    distribution (12.5/25/25/25/12.5) cleanly with class weights
    #  - Simpler gradient dynamics improve FL convergence stability
    #  - label_smoothing=0.05 retained for calibration regularisation
    class_counts = np.bincount(y_train, minlength=N_RISK_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = RiskMLP().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    Xt, yt = build_tensors(X_train, y_train)
    Xv, yv = build_tensors(X_val,   y_val)
    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=BATCH_SIZE)

    # OneCycleLR: super-convergence schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE * 3,
        steps_per_epoch=len(train_loader), epochs=epochs,
    )

    history: Dict = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state   = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        improved = vl_acc > best_val_acc
        if improved:
            best_val_acc = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or improved):
            star = "  ★ best" if improved else ""
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                  f"val_acc={vl_acc:.4f}{star}")

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  ── Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval().cpu()

    if verbose:
        print(f"[Central] Best val accuracy: {best_val_acc:.4f} "
              f"(stopped at epoch {epoch}/{epochs})")

    return model, history


def save_central_model(model: RiskMLP, history: Dict) -> None:
    torch.save(model.state_dict(), CENTRAL_MODEL_PATH)
    hist_path = MODELS_DIR / "central_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Central] Model saved → {CENTRAL_MODEL_PATH}")


def load_central_model() -> RiskMLP:
    model = RiskMLP()
    model.load_state_dict(torch.load(CENTRAL_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# ─── Evaluation helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: RiskMLP, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (predicted_class_indices, probabilities).
    """
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    logits = model(Xt)
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = probs.argmax(axis=1)
    return preds, probs


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    # LabelEncoder sorts class names alphabetically, so le.classes_[i] correctly
    # maps encoded integer i → string label.  RISK_CLASSES is in semantic/domain
    # order (Very_Low→Very_High) which does NOT match the encoded integers.
    le = joblib.load(LABEL_ENCODER_PATH)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
