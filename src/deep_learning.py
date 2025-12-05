"""
deep_learning.py
================
Train recurrent neural networks (LSTM / GRU) for binary stock-direction
prediction using windowed sequences of price + sentiment features.

Architecture
------------
::

    ┌──────────────────┐
    │  Input Sequence  │  (batch, seq_len, n_features)
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  LSTM / GRU      │  × num_layers  (with dropout)
    └────────┬─────────┘
             │  last hidden state
    ┌────────▼─────────┐
    │  Dropout          │
    ├───────────────────┤
    │  Linear (hidden → 64) + ReLU │
    ├───────────────────┤
    │  Dropout          │
    ├───────────────────┤
    │  Linear (64 → 1)  │
    ├───────────────────┤
    │  Sigmoid          │
    └───────────────────┘
             │
        P(UP)  ∈ [0, 1]

Data flow
---------
1. Accept the feature matrix from ``FeatureEngineer.build()`` (with
   ``target`` column).
2. Chronologically split into train / val / test **before** creating
   sequences — no data leakage.
3. Fit a ``StandardScaler`` on **train only**, transform all splits.
4. Generate overlapping sliding-window sequences of length ``seq_len``.
5. Train with Adam + ReduceLROnPlateau + early stopping on val loss.
6. Report accuracy, F1, ROC-AUC, confusion matrix on all splits.
7. Save best checkpoint with full metadata.

Quick-start
-----------
    from src.deep_learning import DeepLearningTrainer

    trainer = DeepLearningTrainer(cell_type="lstm", seq_len=20)
    trainer.train(features_df)
    trainer.print_report()
    trainer.save("models/lstm_v1.pt")

CLI
---
    python -m src.deep_learning --demo --cell lstm
    python -m src.deep_learning --input data/processed/features.csv --cell gru
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> — "
        "<level>{message}</level>"
    ),
    level="INFO",
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CellType = Literal["lstm", "gru"]

# Columns that should never be used as features
_NON_FEATURE_COLUMNS = {"target", "target_return"}


# ═══════════════════════════════════════════════════════════════════════
#  SEQUENCE DATASET
# ═══════════════════════════════════════════════════════════════════════


class SequenceDataset(Dataset):
    """Sliding-window dataset that converts a 2-D feature matrix into
    overlapping ``(seq_len, n_features)`` sequences with a scalar label.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(n_samples, n_features)``.
    targets : np.ndarray
        Shape ``(n_samples,)`` — binary labels.
    seq_len : int
        Number of time-steps per input window.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 20,
    ) -> None:
        self.seq_len = seq_len

        # Build sequences: each sample i uses features[i : i+seq_len] to
        # predict targets[i + seq_len - 1] (the label at the end of the
        # window).
        self.X: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []

        for i in range(len(features) - seq_len + 1):
            self.X.append(
                torch.tensor(
                    features[i : i + seq_len], dtype=torch.float32
                )
            )
            self.y.append(
                torch.tensor(
                    targets[i + seq_len - 1], dtype=torch.float32
                )
            )

        logger.debug(
            f"SequenceDataset: {len(self)} sequences "
            f"(seq_len={seq_len}, features={features.shape[1]})"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════
#  RECURRENT MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════


class StockRNN(nn.Module):
    """Configurable LSTM / GRU model for binary stock-direction prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per time-step.
    hidden_size : int
        Dimensionality of the recurrent hidden state.
    num_layers : int
        Number of stacked recurrent layers.
    dropout : float
        Dropout rate between recurrent layers and in the classifier head.
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    bidirectional : bool
        Use bidirectional RNN (doubles effective hidden size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: CellType = "lstm",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        classifier_input = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Tensor
            Shape ``(batch,)`` — predicted P(UP).
        """
        # rnn_out shape: (batch, seq_len, hidden * num_directions)
        rnn_out, _ = self.rnn(x)

        # Take the output of the last time-step
        last_hidden = rnn_out[:, -1, :]  # (batch, hidden * num_directions)

        out = self.classifier(last_hidden).squeeze(-1)  # (batch,)
        return out


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION RESULT
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DLEvalResult:
    """Container for deep-learning evaluation metrics on one split."""

    split_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion: list
    report: str

    def summary_line(self) -> str:
        auc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        return (
            f"[{self.split_name:>5}]  "
            f"acc={self.accuracy:.4f}  "
            f"prec={self.precision:.4f}  "
            f"rec={self.recall:.4f}  "
            f"f1={self.f1:.4f}  "
            f"auc={auc_str}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TrainingHistory:
    """Records per-epoch metrics during training."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time_secs: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  DEEP LEARNING TRAINER
# ═══════════════════════════════════════════════════════════════════════


class DeepLearningTrainer:
    """End-to-end deep-learning pipeline for stock-direction prediction.

    The pipeline:
    1. Chronological train / val / test split (no shuffling).
    2. Standard-scale features (fit on train only).
    3. Generate sliding-window sequences of length ``seq_len``.
    4. Train LSTM or GRU with early stopping, LR scheduling, and
       gradient clipping.
    5. Evaluate on all splits with full classification metrics.
    6. Save / load model checkpoints with metadata.

    Parameters
    ----------
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    seq_len : int
        Window width (number of historical bars per input sample).
    hidden_size : int
        RNN hidden dimensionality.
    num_layers : int
        Stacked recurrent layers.
    dropout : float
        Dropout rate.
    bidirectional : bool
        Bidirectional RNN.
    lr : float
        Initial learning rate for Adam.
    weight_decay : float
        L2 regularisation strength.
    batch_size : int
        Mini-batch size.
    epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience (epochs with no val-loss improvement).
    lr_patience : int
        ReduceLROnPlateau patience.
    lr_factor : float
        LR reduction factor.
    min_lr : float
        Minimum learning rate.
    grad_clip : float
        Max gradient norm for clipping.
    val_size : float
        Fraction for validation set.
    test_size : float
        Fraction for test set.
    random_state : int
        Seed for reproducibility.
    device : str | None
        PyTorch device (auto-detect if None).
    """

    def __init__(
        self,
        cell_type: CellType = "lstm",
        seq_len: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        lr_patience: int = 7,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        grad_clip: float = 1.0,
        val_size: float = 0.15,
        test_size: float = 0.20,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.cell_type = cell_type.lower()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.grad_clip = grad_clip
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # State — populated after training
        self.model: Optional[StockRNN] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.history: Optional[TrainingHistory] = None
        self.train_metrics: Optional[DLEvalResult] = None
        self.val_metrics: Optional[DLEvalResult] = None
        self.test_metrics: Optional[DLEvalResult] = None

        # Internal split storage (scaled numpy arrays)
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Chronological split + scaling
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Dict[str, DataLoader]:
        """Split, scale, sequence, and wrap into DataLoaders.

        Returns dict with keys ``'train'``, ``'val'``, ``'test'``.
        """
        n = len(df)
        test_start = int(n * (1 - self.test_size))
        val_start = int(n * (1 - self.test_size - self.val_size))

        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]

        # Feature columns
        feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]
        self.feature_names = feature_cols
        n_features = len(feature_cols)

        logger.info(
            f"Chronological split → "
            f"train={len(train_df)} ({val_start/n:.0%})  "
            f"val={len(val_df)} ({(test_start-val_start)/n:.0%})  "
            f"test={len(test_df)} ({(n-test_start)/n:.0%})  "
            f"features={n_features}"
        )

        # Date ranges
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(
                f"  Train: {train_df.index.min().date()} → {train_df.index.max().date()}"
            )
            logger.info(
                f"  Val:   {val_df.index.min().date()} → {val_df.index.max().date()}"
            )
            logger.info(
                f"  Test:  {test_df.index.min().date()} → {test_df.index.max().date()}"
            )

        # Extract arrays
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_col].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df[target_col].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df[target_col].values.astype(np.float32)

        # Handle NaN / Inf in features
        for arr_name, arr in [("train", X_train), ("val", X_val), ("test", X_test)]:
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(
                    f"  {arr_name}: {nan_count} NaN, {inf_count} Inf → replacing with 0"
                )
                arr[np.isnan(arr)] = 0.0
                arr[np.isinf(arr)] = 0.0

        # Scale features (fit on train only)
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        logger.info("Applied StandardScaler (fit on train only)")

        # Store for later use
        self._X_train, self._y_train = X_train, y_train
        self._X_val, self._y_val = X_val, y_val
        self._X_test, self._y_test = X_test, y_test

        # Create sequence datasets
        train_ds = SequenceDataset(X_train, y_train, self.seq_len)
        val_ds = SequenceDataset(X_val, y_val, self.seq_len)
        test_ds = SequenceDataset(X_test, y_test, self.seq_len)

        logger.info(
            f"Sequence datasets (seq_len={self.seq_len}): "
            f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
        )

        # Class balance
        for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            labels = torch.stack([ds[i][1] for i in range(len(ds))]).numpy()
            pos_pct = labels.mean() * 100
            logger.debug(f"  {name} class balance: {pos_pct:.1f}% positive")

        # DataLoaders (no shuffle — chronological order preserved)
        loaders = {
            "train": DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=False
            ),
            "val": DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False
            ),
            "test": DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False
            ),
        }
        return loaders

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> "DeepLearningTrainer":
        """Full training pipeline.

        Steps:
        1. Prepare data (split → scale → sequence → DataLoaders).
        2. Build the LSTM / GRU model.
        3. Train with early stopping + LR scheduling.
        4. Load best checkpoint weights.
        5. Evaluate on train / val / test.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from ``FeatureEngineer.build()`` with a
            ``target`` column.
        target_col : str
            Name of the binary target column.

        Returns
        -------
        DeepLearningTrainer
            Self (for method chaining).
        """
        # 1 — Data
        loaders = self._prepare_data(df, target_col)
        n_features = len(self.feature_names)

        # 2 — Model
        self.model = StockRNN(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
            bidirectional=self.bidirectional,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model: {self.cell_type.upper()} | "
            f"hidden={self.hidden_size} | layers={self.num_layers} | "
            f"bidir={self.bidirectional} | "
            f"params={total_params:,} (trainable={trainable:,})"
        )
        logger.info(f"Device: {self.device}")

        # 3 — Optimiser, scheduler, loss
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
        )

        # 4 — Training loop
        history = TrainingHistory()
        best_state_dict = None
        no_improve = 0
        t0 = time.time()

        logger.info(
            f"Training for up to {self.epochs} epochs "
            f"(patience={self.patience}, batch={self.batch_size}, "
            f"lr={self.lr})"
        )
        print(f"\n{'-' * 78}")
        print(
            f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>11}  "
            f"{'Train Acc':>10}  {'Val Acc':>10}  {'LR':>10}"
        )
        print(f"{'-' * 78}")

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in loaders["train"]:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                preds = (y_pred >= 0.5).float()
                train_correct += (preds == y_batch).sum().item()
                train_total += len(y_batch)

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # ── Validate ──
            val_loss, val_acc = self._eval_epoch(
                loaders["val"], criterion
            )

            # ── LR schedule ──
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            # ── Record ──
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_acc)
            history.learning_rates.append(current_lr)

            # ── Early stopping check ──
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
                marker = " * best"
            else:
                no_improve += 1
                marker = ""

            # ── Print epoch ──
            print(
                f"  {epoch:5d}  {train_loss:11.6f}  {val_loss:11.6f}  "
                f"{train_acc:10.4f}  {val_acc:10.4f}  {current_lr:10.2e}"
                f"{marker}"
            )

            if no_improve >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best={history.best_epoch})"
                )
                break

        history.total_time_secs = time.time() - t0
        print(f"{'-' * 78}")
        logger.info(
            f"Training complete in {history.total_time_secs:.1f}s  "
            f"(best epoch={history.best_epoch}, "
            f"best val_loss={history.best_val_loss:.6f})"
        )

        # 5 — Restore best weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(f"Restored best model weights from epoch {history.best_epoch}")

        self.history = history

        # 6 — Evaluate all splits
        self.train_metrics = self._evaluate_split(loaders["train"], "train")
        self.val_metrics = self._evaluate_split(loaders["val"], "val")
        self.test_metrics = self._evaluate_split(loaders["test"], "test")

        return self

    # ------------------------------------------------------------------
    # Epoch evaluation (loss + accuracy)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Compute average loss and accuracy over a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = (y_pred >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Full split evaluation (metrics)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_split(
        self,
        loader: DataLoader,
        split_name: str,
    ) -> DLEvalResult:
        """Compute classification metrics on one split."""
        self.model.eval()
        all_preds: List[float] = []
        all_proba: List[float] = []
        all_labels: List[float] = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            proba = self.model(X_batch).cpu().numpy()
            all_proba.extend(proba.tolist())
            all_preds.extend((proba >= 0.5).astype(float).tolist())
            all_labels.extend(y_batch.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_proba)

        roc = None
        try:
            roc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            pass

        result = DLEvalResult(
            split_name=split_name,
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=roc,
            confusion=confusion_matrix(y_true, y_pred).tolist(),
            report=classification_report(y_true, y_pred, zero_division=0),
        )
        logger.info(result.summary_line())
        return result

    # ------------------------------------------------------------------
    # Public evaluate API
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """Return test-set metrics as a dict (mirrors ModelTrainer API).

        Returns
        -------
        dict
            accuracy, precision, recall, f1, roc_auc, confusion_matrix, report.
        """
        if self.test_metrics is None:
            raise RuntimeError("Call .train() before .evaluate()")

        m = self.test_metrics
        return {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "roc_auc": m.roc_auc,
            "confusion_matrix": m.confusion,
            "report": m.report,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as training).
            Must have at least ``seq_len`` rows.

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1) for each valid window.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict P(UP) probabilities for new data.

        Returns
        -------
        np.ndarray
            Shape ``(n_windows,)`` — probability of class 1 for each
            valid sliding window.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .predict_proba()")

        X_arr = X[self.feature_names].values.astype(np.float32)
        X_arr[np.isnan(X_arr)] = 0.0
        X_arr[np.isinf(X_arr)] = 0.0

        if self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        self.model.eval()
        ds = SequenceDataset(X_arr, np.zeros(len(X_arr)), self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        probas: List[float] = []
        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device)
            out = self.model(X_batch).cpu().numpy()
            probas.extend(out.tolist())

        return np.array(probas)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path] = "models/dl_model.pt") -> Path:
        """Save model checkpoint, scaler, and metadata.

        Creates:
        - ``<path>`` — the PyTorch checkpoint (``.pt``)
        - ``<path>.meta.json`` — human-readable metadata

        Parameters
        ----------
        path : str | Path
            Output path for the checkpoint.

        Returns
        -------
        Path
            The path the model was saved to.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .save()")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "cell_type": self.cell_type,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "input_size": len(self.feature_names),
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved → {path}")

        # Metadata JSON
        meta: Dict[str, Any] = {
            "cell_type": self.cell_type,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "bidirectional": self.bidirectional,
            "device": str(self.device),
        }
        if self.history:
            meta["training"] = {
                "best_epoch": self.history.best_epoch,
                "best_val_loss": self.history.best_val_loss,
                "total_epochs": len(self.history.train_losses),
                "total_time_secs": round(self.history.total_time_secs, 1),
            }
        if self.test_metrics:
            meta["test_metrics"] = {
                "accuracy": self.test_metrics.accuracy,
                "precision": self.test_metrics.precision,
                "recall": self.test_metrics.recall,
                "f1": self.test_metrics.f1,
                "roc_auc": self.test_metrics.roc_auc,
            }

        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(f"Metadata saved → {meta_path}")

        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "DeepLearningTrainer":
        """Load a previously saved checkpoint.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` checkpoint.
        device : str | None
            Target device (auto-detect if None).

        Returns
        -------
        DeepLearningTrainer
            A trainer with model and scaler restored; ready for
            ``.predict()`` / ``.predict_proba()``.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Checkpoint loaded ← {path}")

        trainer = cls(
            cell_type=checkpoint["cell_type"],
            seq_len=checkpoint["seq_len"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.3),
            bidirectional=checkpoint.get("bidirectional", False),
            device=device,
        )

        trainer.feature_names = checkpoint["feature_names"]
        trainer.scaler = checkpoint.get("scaler")

        trainer.model = StockRNN(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.3),
            cell_type=checkpoint["cell_type"],
            bidirectional=checkpoint.get("bidirectional", False),
        ).to(trainer.device)

        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        logger.info(
            f"Model restored: {trainer.cell_type.upper()} "
            f"({checkpoint['input_size']} features, "
            f"seq_len={trainer.seq_len})"
        )
        return trainer

    # ------------------------------------------------------------------
    # Pretty-print report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted summary of training and evaluation results."""
        if self.test_metrics is None:
            raise RuntimeError("Call .train() before .print_report()")

        bidir_str = " (bidirectional)" if self.bidirectional else ""

        print(f"\n{'=' * 78}")
        print(
            f"  Deep Learning Report -- "
            f"{self.cell_type.upper()}{bidir_str}"
        )
        print(f"{'=' * 78}")

        print(f"\n  Architecture:")
        print(f"    Cell type    : {self.cell_type.upper()}")
        print(f"    Hidden size  : {self.hidden_size}")
        print(f"    Layers       : {self.num_layers}")
        print(f"    Bidirectional: {self.bidirectional}")
        print(f"    Dropout      : {self.dropout}")
        print(f"    Sequence len : {self.seq_len}")
        print(f"    Features     : {len(self.feature_names)}")
        print(f"    Device       : {self.device}")

        if self.history:
            print(f"\n  Training:")
            print(f"    Epochs       : {len(self.history.train_losses)}")
            print(f"    Best epoch   : {self.history.best_epoch}")
            print(f"    Best val loss: {self.history.best_val_loss:.6f}")
            print(f"    Time         : {self.history.total_time_secs:.1f}s")

        print(f"\n{'-' * 78}")
        print("  Performance Metrics:")
        print(f"{'-' * 78}")
        for m in [self.train_metrics, self.val_metrics, self.test_metrics]:
            if m:
                print(f"    {m.summary_line()}")

        print(f"\n{'-' * 78}")
        print("  Test Set Classification Report:")
        print(f"{'-' * 78}")
        for line in self.test_metrics.report.split("\n"):
            print(f"    {line}")

        print(f"\n{'-' * 78}")
        print("  Test Set Confusion Matrix:")
        print(f"{'-' * 78}")
        cm = np.array(self.test_metrics.confusion)
        print(f"                Predicted")
        print(f"              {'DOWN':>6}  {'UP':>6}")
        if len(cm) == 2:
            print(f"    Actual DOWN {cm[0][0]:>6}  {cm[0][1]:>6}")
            print(f"    Actual UP   {cm[1][0]:>6}  {cm[1][1]:>6}")
        else:
            for i, row in enumerate(cm):
                print(f"    Class {i}     {'  '.join(f'{v:>6}' for v in row)}")

        print(f"\n{'=' * 78}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def train_deep_model(
    df: pd.DataFrame,
    cell_type: CellType = "lstm",
    seq_len: int = 20,
    hidden_size: int = 128,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    target_col: str = "target",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple["DeepLearningTrainer", Dict[str, Any]]:
    """One-call function: train, evaluate, and optionally save.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix from ``FeatureEngineer.build()``.
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    seq_len : int
        Sequence window length.
    hidden_size : int
        RNN hidden dimensionality.
    num_layers : int
        Stacked recurrent layers.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.
    target_col : str
        Target column name.
    save_path : str | Path | None
        Save model checkpoint if provided.

    Returns
    -------
    tuple[DeepLearningTrainer, dict]
        The fitted trainer and a dict of test metrics.
    """
    trainer = DeepLearningTrainer(
        cell_type=cell_type,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    trainer.train(df, target_col=target_col)
    trainer.print_report()

    metrics = trainer.evaluate()

    if save_path:
        trainer.save(save_path)

    return trainer, metrics


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an LSTM / GRU model for stock-direction prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.deep_learning --demo --cell lstm\n"
            "  python -m src.deep_learning --demo --cell gru --seq-len 30\n"
            "  python -m src.deep_learning --demo --compare\n"
            "  python -m src.deep_learning --input data/processed/features.csv "
            "--cell lstm --save models/lstm_v1.pt\n"
        ),
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to feature matrix CSV (output of feature_engineering.py)",
    )
    parser.add_argument(
        "--cell", type=str, default="lstm",
        choices=["lstm", "gru"],
        help="RNN cell type (default: lstm)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=20,
        help="Sequence window length (default: 20)",
    )
    parser.add_argument(
        "--hidden", type=int, default=128,
        help="RNN hidden size (default: 128)",
    )
    parser.add_argument(
        "--layers", type=int, default=2,
        help="Number of RNN layers (default: 2)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3)",
    )
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="Use bidirectional RNN",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the trained model (.pt)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic data (no input file needed)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare LSTM vs GRU (demo mode only)",
    )
    args = parser.parse_args()

    if args.demo or args.compare:
        # ── Generate synthetic feature matrix ──
        from src.feature_engineering import FeatureEngineer

        np.random.seed(42)
        n = 500
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        base = 150 + np.random.randn(n).cumsum()
        prices = pd.DataFrame(
            {
                "Open": base + np.random.uniform(-1, 1, n),
                "High": base + np.abs(np.random.randn(n)) * 2,
                "Low": base - np.abs(np.random.randn(n)) * 2,
                "Close": base,
                "Volume": np.random.randint(500_000, 5_000_000, n),
            },
            index=dates,
        )
        prices["High"] = prices[["Open", "High", "Close"]].max(axis=1)
        prices["Low"] = prices[["Open", "Low", "Close"]].min(axis=1)

        eng = FeatureEngineer()
        features_df = eng.build(prices)

        if args.compare:
            # ── Compare LSTM vs GRU ──
            print(f"\n{'=' * 78}")
            print("  DEEP LEARNING COMPARISON -- LSTM vs GRU")
            print(f"{'=' * 78}")

            results = {}
            for cell in ["lstm", "gru"]:
                print(f"\n{'-' * 78}")
                print(f"  Training: {cell.upper()}")
                print(f"{'-' * 78}")
                trainer = DeepLearningTrainer(
                    cell_type=cell,
                    seq_len=args.seq_len,
                    hidden_size=args.hidden,
                    num_layers=args.layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    dropout=args.dropout,
                    bidirectional=args.bidirectional,
                )
                trainer.train(features_df)
                m = trainer.evaluate()
                results[cell] = m

            # Summary table
            print(f"\n{'=' * 78}")
            print("  COMPARISON SUMMARY")
            print(f"{'=' * 78}")
            print(
                f"  {'Model':<15} {'Acc':>8} {'Prec':>8} "
                f"{'Rec':>8} {'F1':>8} {'AUC':>8}"
            )
            print(f"  {'-' * 55}")
            for name, m in results.items():
                auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] else "N/A"
                print(
                    f"  {name.upper():<15} {m['accuracy']:>8.4f} "
                    f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                    f"{m['f1']:>8.4f} {auc:>8}"
                )
            print(f"{'=' * 78}\n")

        else:
            # ── Single model demo ──
            save_path = args.save or f"models/{args.cell}_demo.pt"
            trainer, metrics = train_deep_model(
                features_df,
                cell_type=args.cell,
                seq_len=args.seq_len,
                hidden_size=args.hidden,
                num_layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_path=save_path,
            )

    elif args.input:
        # ── Train on real data ──
        features_df = pd.read_csv(args.input, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(features_df)} rows from {args.input}")

        save_path = args.save or f"models/{args.cell}_trained.pt"
        trainer, metrics = train_deep_model(
            features_df,
            cell_type=args.cell,
            seq_len=args.seq_len,
            hidden_size=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=save_path,
        )

    else:
        parser.print_help()
