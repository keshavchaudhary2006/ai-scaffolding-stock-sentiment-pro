"""
hybrid_model.py
===============
Two-branch hybrid neural network that fuses sequential price features
with daily sentiment features for stock-direction prediction.

Architecture
------------
::

                      Feature Matrix (from FeatureEngineer)
                                    |
                    +---------------+---------------+
                    |                               |
            Price columns                  Sentiment columns
            (OHLCV-derived)                (sent_* columns)
                    |                               |
       +------------v-----------+       +-----------v-----------+
       |  Sliding-window seq    |       |  Sliding-window seq   |
       |  (batch, seq, P)       |       |  (batch, seq, S)      |
       +------------+-----------+       +-----------+-----------+
                    |                               |
       +------------v-----------+       +-----------v-----------+
       |  LSTM / GRU            |       |  FC Encoder           |
       |  num_layers, bidir     |       |  S -> 64 -> 32        |
       +------------+-----------+       +-----------+-----------+
                    |                               |
             last hidden                    encoded repr
            (hidden_size*D)                    (32,)
                    |                               |
                    +----------- CONCAT ------------+
                                    |
                          (hidden*D + 32)
                                    |
                    +---------------v---------------+
                    |       Fusion Head              |
                    |  Linear -> ReLU -> Dropout     |
                    |  Linear -> ReLU -> Dropout     |
                    |  Linear(1) -> Sigmoid           |
                    +---------------+---------------+
                                    |
                               P(UP) in [0,1]

Key design decisions
--------------------
- **Separate scalers**: price and sentiment features are scaled
  independently and fed to different branches.
- **The RNN branch** captures temporal dynamics (momentum, trend,
  volatility regimes) from multi-day price windows.
- **The sentiment branch** uses a feedforward encoder so it can capture
  non-linear interactions between sentiment stats (mean, std, skew,
  positive_pct, etc.) without imposing unnecessary sequential structure.
  Both branches still receive the *sequence* of sentiment days so the
  model can observe how sentiment evolves over the window.
- **Late fusion**: the two representation vectors are concatenated
  and passed through a shared classifier, allowing the model to learn
  cross-domain interactions.

Quick-start
-----------
    from src.hybrid_model import HybridTrainer

    trainer = HybridTrainer(cell_type="lstm", seq_len=20)
    trainer.train(features_df)          # features_df has sent_* cols
    trainer.print_report()
    trainer.save("models/hybrid_v1.pt")

CLI
---
    python -m src.hybrid_model --demo --cell lstm
    python -m src.hybrid_model --demo --cell gru --seq-len 30
    python -m src.hybrid_model --input data/processed/features.csv --cell lstm
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
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
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> -- "
        "<level>{message}</level>"
    ),
    level="INFO",
)

# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------

CellType = Literal["lstm", "gru"]

_NON_FEATURE_COLUMNS = {"target", "target_return"}

# Heuristic: columns starting with these prefixes are sentiment features
_SENTIMENT_PREFIXES = ("sent_",)


# ===================================================================
#  DUAL-BRANCH DATASET
# ===================================================================


class DualBranchDataset(Dataset):
    """Sliding-window dataset that produces two aligned tensors per sample:
    one for the price branch and one for the sentiment branch.

    Parameters
    ----------
    price_features : np.ndarray
        Shape ``(T, P)`` -- price / technical feature matrix.
    sent_features : np.ndarray
        Shape ``(T, S)`` -- sentiment feature matrix.
    targets : np.ndarray
        Shape ``(T,)`` -- binary labels.
    seq_len : int
        Sliding-window width.
    """

    def __init__(
        self,
        price_features: np.ndarray,
        sent_features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 20,
    ) -> None:
        self.seq_len = seq_len

        self.X_price: List[torch.Tensor] = []
        self.X_sent: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []

        for i in range(len(targets) - seq_len + 1):
            self.X_price.append(
                torch.tensor(price_features[i : i + seq_len], dtype=torch.float32)
            )
            self.X_sent.append(
                torch.tensor(sent_features[i : i + seq_len], dtype=torch.float32)
            )
            self.y.append(
                torch.tensor(targets[i + seq_len - 1], dtype=torch.float32)
            )

        logger.debug(
            f"DualBranchDataset: {len(self)} sequences "
            f"(seq_len={seq_len}, price_feats={price_features.shape[1]}, "
            f"sent_feats={sent_features.shape[1]})"
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_price[idx], self.X_sent[idx], self.y[idx]


# ===================================================================
#  HYBRID MODEL
# ===================================================================


class HybridStockModel(nn.Module):
    """Two-branch model: RNN for price + FC encoder for sentiment.

    Parameters
    ----------
    price_input_size : int
        Number of price / technical features per time-step.
    sent_input_size : int
        Number of sentiment features per time-step.
    hidden_size : int
        RNN hidden state dimensionality.
    num_layers : int
        Stacked RNN layers.
    dropout : float
        Dropout rate throughout the model.
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    bidirectional : bool
        Bidirectional RNN.
    sent_hidden : int
        Intermediate dimensionality of the sentiment encoder.
    sent_out : int
        Output dimensionality of the sentiment encoder.
    fusion_hidden : int
        Hidden size of the fusion classifier head.
    """

    def __init__(
        self,
        price_input_size: int,
        sent_input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: CellType = "lstm",
        bidirectional: bool = False,
        sent_hidden: int = 64,
        sent_out: int = 32,
        fusion_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.sent_input_size = sent_input_size
        self.price_input_size = price_input_size

        # -- Price branch: RNN -----------------------------------------------
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.price_rnn = rnn_cls(
            input_size=price_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        price_out_dim = hidden_size * self.num_directions

        # -- Sentiment branch: FC encoder ------------------------------------
        # Flatten the sentiment sequence (seq_len * S) then encode
        # We use a small 2-layer MLP to encode the sentiment window
        self.sent_encoder = nn.Sequential(
            nn.LazyLinear(sent_hidden),  # input dim inferred at first forward
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sent_hidden, sent_out),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # -- Fusion head -----------------------------------------------------
        fusion_in = price_out_dim + sent_out
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_price: torch.Tensor,
        x_sent: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_price : Tensor
            Shape ``(batch, seq_len, price_features)``.
        x_sent  : Tensor
            Shape ``(batch, seq_len, sent_features)``.

        Returns
        -------
        Tensor
            Shape ``(batch,)`` -- P(UP).
        """
        # Price branch -> last hidden
        rnn_out, _ = self.price_rnn(x_price)
        price_repr = rnn_out[:, -1, :]  # (batch, hidden * D)

        # Sentiment branch -> flatten window then encode
        batch_size = x_sent.size(0)
        sent_flat = x_sent.reshape(batch_size, -1)  # (batch, seq_len * S)
        sent_repr = self.sent_encoder(sent_flat)     # (batch, sent_out)

        # Fuse
        combined = torch.cat([price_repr, sent_repr], dim=1)
        out = self.fusion_head(combined).squeeze(-1)  # (batch,)
        return out


# ===================================================================
#  DATACLASSES
# ===================================================================


@dataclass
class HybridEvalResult:
    """Evaluation metrics for one data split."""

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


@dataclass
class HybridHistory:
    """Training history for the hybrid model."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time_secs: float = 0.0


# ===================================================================
#  HYBRID TRAINER
# ===================================================================


class HybridTrainer:
    """End-to-end trainer for the two-branch hybrid model.

    Accepts a single feature matrix (output of ``FeatureEngineer.build()``
    with sentiment columns present). Automatically splits columns into
    price vs sentiment branches.

    Parameters
    ----------
    cell_type : str
        ``"lstm"`` or ``"gru"`` for the price branch.
    seq_len : int
        Sliding-window size.
    hidden_size : int
        RNN hidden dimensionality.
    num_layers : int
        Stacked RNN layers.
    dropout : float
        Dropout rate.
    bidirectional : bool
        Bidirectional RNN.
    sent_hidden : int
        Sentiment encoder intermediate size.
    sent_out : int
        Sentiment encoder output size.
    fusion_hidden : int
        Fusion head hidden size.
    lr : float
        Initial learning rate.
    weight_decay : float
        L2 regularisation.
    batch_size : int
        Mini-batch size.
    epochs : int
        Maximum epochs.
    patience : int
        Early-stopping patience.
    lr_patience : int
        ReduceLROnPlateau patience.
    lr_factor : float
        LR reduction factor.
    min_lr : float
        Minimum learning rate.
    grad_clip : float
        Maximum gradient norm.
    val_size : float
        Validation fraction.
    test_size : float
        Test fraction.
    random_state : int
        Seed.
    device : str | None
        PyTorch device.
    """

    def __init__(
        self,
        cell_type: CellType = "lstm",
        seq_len: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        sent_hidden: int = 64,
        sent_out: int = 32,
        fusion_hidden: int = 64,
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
        self.sent_hidden = sent_hidden
        self.sent_out = sent_out
        self.fusion_hidden = fusion_hidden
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

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # State -- populated after training
        self.model: Optional[HybridStockModel] = None
        self.price_scaler: Optional[StandardScaler] = None
        self.sent_scaler: Optional[StandardScaler] = None
        self.price_cols: List[str] = []
        self.sent_cols: List[str] = []
        self.history: Optional[HybridHistory] = None
        self.train_metrics: Optional[HybridEvalResult] = None
        self.val_metrics: Optional[HybridEvalResult] = None
        self.test_metrics: Optional[HybridEvalResult] = None

    # ------------------------------------------------------------------
    # Column splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_columns(
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """Partition feature columns into price and sentiment groups.

        Sentiment columns are identified by the ``sent_`` prefix.
        Everything else (except target columns) goes to the price branch.

        Returns
        -------
        tuple[list[str], list[str]]
            (price_cols, sent_cols)
        """
        all_feature_cols = [
            c for c in df.columns if c not in _NON_FEATURE_COLUMNS
        ]

        sent_cols = [
            c for c in all_feature_cols
            if any(c.startswith(p) for p in _SENTIMENT_PREFIXES)
        ]
        price_cols = [c for c in all_feature_cols if c not in sent_cols]

        return price_cols, sent_cols

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Dict[str, DataLoader]:
        """Split, scale, sequence into dual-branch DataLoaders."""
        n = len(df)
        test_start = int(n * (1 - self.test_size))
        val_start = int(n * (1 - self.test_size - self.val_size))

        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]

        # Identify column groups
        price_cols, sent_cols = self._split_columns(df)
        self.price_cols = price_cols
        self.sent_cols = sent_cols

        # If no sentiment columns found, create a dummy one so the
        # sentiment branch still has an input dimension
        if not sent_cols:
            logger.warning(
                "No sentiment columns (sent_*) found -- "
                "creating a zero-filled dummy column. "
                "Consider providing sentiment data for best results."
            )
            for split_df in [train_df, val_df, test_df, df]:
                split_df["sent_dummy"] = 0.0
            sent_cols = ["sent_dummy"]
            self.sent_cols = sent_cols

        logger.info(
            f"Chronological split -> "
            f"train={len(train_df)} ({val_start/n:.0%})  "
            f"val={len(val_df)} ({(test_start-val_start)/n:.0%})  "
            f"test={len(test_df)} ({(n-test_start)/n:.0%})"
        )
        logger.info(
            f"  Price branch : {len(price_cols)} features"
        )
        logger.info(
            f"  Sentiment branch: {len(sent_cols)} features"
        )

        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(
                f"  Train: {train_df.index.min().date()} -> {train_df.index.max().date()}"
            )
            logger.info(
                f"  Val:   {val_df.index.min().date()} -> {val_df.index.max().date()}"
            )
            logger.info(
                f"  Test:  {test_df.index.min().date()} -> {test_df.index.max().date()}"
            )

        # Extract arrays
        def _extract(split_df: pd.DataFrame):
            p = split_df[price_cols].values.astype(np.float32)
            s = split_df[sent_cols].values.astype(np.float32)
            y = split_df[target_col].values.astype(np.float32)
            # Replace NaN / Inf
            for arr in [p, s]:
                arr[np.isnan(arr)] = 0.0
                arr[np.isinf(arr)] = 0.0
            return p, s, y

        p_train, s_train, y_train = _extract(train_df)
        p_val, s_val, y_val = _extract(val_df)
        p_test, s_test, y_test = _extract(test_df)

        # Separate scalers
        self.price_scaler = StandardScaler()
        p_train = self.price_scaler.fit_transform(p_train)
        p_val = self.price_scaler.transform(p_val)
        p_test = self.price_scaler.transform(p_test)

        self.sent_scaler = StandardScaler()
        s_train = self.sent_scaler.fit_transform(s_train)
        s_val = self.sent_scaler.transform(s_val)
        s_test = self.sent_scaler.transform(s_test)

        logger.info("Applied separate StandardScalers (fit on train only)")

        # Build datasets
        train_ds = DualBranchDataset(p_train, s_train, y_train, self.seq_len)
        val_ds = DualBranchDataset(p_val, s_val, y_val, self.seq_len)
        test_ds = DualBranchDataset(p_test, s_test, y_test, self.seq_len)

        logger.info(
            f"Sequence datasets (seq_len={self.seq_len}): "
            f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
        )

        loaders = {
            "train": DataLoader(train_ds, batch_size=self.batch_size, shuffle=False),
            "val": DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=self.batch_size, shuffle=False),
        }
        return loaders

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> "HybridTrainer":
        """Full training pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from ``FeatureEngineer.build()`` that includes
            both price and sentiment (``sent_*``) columns, plus a
            ``target`` column.
        target_col : str
            Binary target column name.

        Returns
        -------
        HybridTrainer
            Self, for method chaining.
        """
        # 1 -- Data
        loaders = self._prepare_data(df, target_col)

        # 2 -- Build model
        self.model = HybridStockModel(
            price_input_size=len(self.price_cols),
            sent_input_size=len(self.sent_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
            bidirectional=self.bidirectional,
            sent_hidden=self.sent_hidden,
            sent_out=self.sent_out,
            fusion_hidden=self.fusion_hidden,
        ).to(self.device)

        # Trigger LazyLinear materialisation with a dummy forward pass
        with torch.no_grad():
            dummy_p = torch.zeros(1, self.seq_len, len(self.price_cols)).to(self.device)
            dummy_s = torch.zeros(1, self.seq_len, len(self.sent_cols)).to(self.device)
            self.model(dummy_p, dummy_s)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model: Hybrid {self.cell_type.upper()} | "
            f"price_rnn={self.hidden_size}x{self.num_layers} | "
            f"sent_encoder -> {self.sent_out} | "
            f"fusion={self.fusion_hidden} | "
            f"params={total_params:,} (trainable={trainable:,})"
        )
        logger.info(f"Device: {self.device}")

        # 3 -- Optimiser, scheduler, loss
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

        # 4 -- Training loop
        history = HybridHistory()
        best_state_dict = None
        no_improve = 0
        t0 = time.time()

        logger.info(
            f"Training for up to {self.epochs} epochs "
            f"(patience={self.patience}, batch={self.batch_size}, lr={self.lr})"
        )
        print(f"\n{'-' * 78}")
        print(
            f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>11}  "
            f"{'Train Acc':>10}  {'Val Acc':>10}  {'LR':>10}"
        )
        print(f"{'-' * 78}")

        for epoch in range(1, self.epochs + 1):
            # -- Train --
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for xp, xs, yb in loaders["train"]:
                xp = xp.to(self.device)
                xs = xs.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(xp, xs)
                loss = criterion(y_pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                optimizer.step()

                train_loss += loss.item() * len(yb)
                preds = (y_pred >= 0.5).float()
                train_correct += (preds == yb).sum().item()
                train_total += len(yb)

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # -- Validate --
            val_loss, val_acc = self._eval_epoch(loaders["val"], criterion)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_acc)
            history.learning_rates.append(current_lr)

            # -- Early stopping --
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

        # 5 -- Restore best weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(f"Restored best weights from epoch {history.best_epoch}")

        self.history = history

        # 6 -- Evaluate all splits
        self.train_metrics = self._evaluate_split(loaders["train"], "train")
        self.val_metrics = self._evaluate_split(loaders["val"], "val")
        self.test_metrics = self._evaluate_split(loaders["test"], "test")

        return self

    # ------------------------------------------------------------------
    # Epoch evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Average loss + accuracy over one DataLoader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for xp, xs, yb in loader:
            xp = xp.to(self.device)
            xs = xs.to(self.device)
            yb = yb.to(self.device)

            y_pred = self.model(xp, xs)
            loss = criterion(y_pred, yb)

            total_loss += loss.item() * len(yb)
            preds = (y_pred >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)

        return total_loss / max(total, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    # Full split evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_split(
        self, loader: DataLoader, split_name: str
    ) -> HybridEvalResult:
        self.model.eval()
        all_preds, all_proba, all_labels = [], [], []

        for xp, xs, yb in loader:
            xp = xp.to(self.device)
            xs = xs.to(self.device)

            proba = self.model(xp, xs).cpu().numpy()
            all_proba.extend(proba.tolist())
            all_preds.extend((proba >= 0.5).astype(float).tolist())
            all_labels.extend(yb.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_proba)

        roc = None
        try:
            roc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            pass

        result = HybridEvalResult(
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
        """Return test-set metrics as a dict."""
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
        """Predict binary labels for new data."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict P(UP) for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns as training.

        Returns
        -------
        np.ndarray
            Shape ``(n_windows,)`` -- probabilities.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .predict_proba()")

        p_arr = X[self.price_cols].values.astype(np.float32)
        s_arr = X[self.sent_cols].values.astype(np.float32)
        for arr in [p_arr, s_arr]:
            arr[np.isnan(arr)] = 0.0
            arr[np.isinf(arr)] = 0.0

        if self.price_scaler is not None:
            p_arr = self.price_scaler.transform(p_arr)
        if self.sent_scaler is not None:
            s_arr = self.sent_scaler.transform(s_arr)

        self.model.eval()
        ds = DualBranchDataset(p_arr, s_arr, np.zeros(len(p_arr)), self.seq_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        probas: List[float] = []
        for xp, xs, _ in loader:
            xp = xp.to(self.device)
            xs = xs.to(self.device)
            out = self.model(xp, xs).cpu().numpy()
            probas.extend(out.tolist())

        return np.array(probas)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path] = "models/hybrid_model.pt") -> Path:
        """Save checkpoint + metadata JSON."""
        if self.model is None:
            raise RuntimeError("Call .train() before .save()")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "price_scaler": self.price_scaler,
            "sent_scaler": self.sent_scaler,
            "price_cols": self.price_cols,
            "sent_cols": self.sent_cols,
            "cell_type": self.cell_type,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "sent_hidden": self.sent_hidden,
            "sent_out": self.sent_out,
            "fusion_hidden": self.fusion_hidden,
            "price_input_size": len(self.price_cols),
            "sent_input_size": len(self.sent_cols),
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved -> {path}")

        meta: Dict[str, Any] = {
            "model_type": "hybrid",
            "cell_type": self.cell_type,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "n_price_features": len(self.price_cols),
            "n_sent_features": len(self.sent_cols),
            "price_cols": self.price_cols,
            "sent_cols": self.sent_cols,
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
        logger.info(f"Metadata saved -> {meta_path}")
        return path

    @classmethod
    def load(
        cls, path: Union[str, Path], device: Optional[str] = None
    ) -> "HybridTrainer":
        """Load a saved hybrid model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"Checkpoint loaded <- {path}")

        trainer = cls(
            cell_type=checkpoint["cell_type"],
            seq_len=checkpoint["seq_len"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.3),
            bidirectional=checkpoint.get("bidirectional", False),
            sent_hidden=checkpoint.get("sent_hidden", 64),
            sent_out=checkpoint.get("sent_out", 32),
            fusion_hidden=checkpoint.get("fusion_hidden", 64),
            device=device,
        )

        trainer.price_cols = checkpoint["price_cols"]
        trainer.sent_cols = checkpoint["sent_cols"]
        trainer.price_scaler = checkpoint.get("price_scaler")
        trainer.sent_scaler = checkpoint.get("sent_scaler")

        trainer.model = HybridStockModel(
            price_input_size=checkpoint["price_input_size"],
            sent_input_size=checkpoint["sent_input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.3),
            cell_type=checkpoint["cell_type"],
            bidirectional=checkpoint.get("bidirectional", False),
            sent_hidden=checkpoint.get("sent_hidden", 64),
            sent_out=checkpoint.get("sent_out", 32),
            fusion_hidden=checkpoint.get("fusion_hidden", 64),
        ).to(trainer.device)

        # Materialise LazyLinear
        with torch.no_grad():
            dummy_p = torch.zeros(
                1, trainer.seq_len, checkpoint["price_input_size"]
            ).to(trainer.device)
            dummy_s = torch.zeros(
                1, trainer.seq_len, checkpoint["sent_input_size"]
            ).to(trainer.device)
            trainer.model(dummy_p, dummy_s)

        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        logger.info(
            f"Hybrid model restored: {trainer.cell_type.upper()} "
            f"(price={checkpoint['price_input_size']}, "
            f"sent={checkpoint['sent_input_size']}, "
            f"seq_len={trainer.seq_len})"
        )
        return trainer

    # ------------------------------------------------------------------
    # Pretty-print report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted training and evaluation summary."""
        if self.test_metrics is None:
            raise RuntimeError("Call .train() before .print_report()")

        bidir_str = " (bidirectional)" if self.bidirectional else ""

        print(f"\n{'=' * 78}")
        print(
            f"  Hybrid Model Report -- "
            f"{self.cell_type.upper()}{bidir_str} + Sentiment Encoder"
        )
        print(f"{'=' * 78}")

        print(f"\n  Architecture:")
        print(f"    Price branch  : {self.cell_type.upper()} "
              f"h={self.hidden_size} x {self.num_layers} layers")
        print(f"    Price features: {len(self.price_cols)}")
        print(f"    Sent. branch  : FC {len(self.sent_cols)} "
              f"-> {self.sent_hidden} -> {self.sent_out}")
        print(f"    Sent. features: {len(self.sent_cols)}")
        print(f"    Fusion head   : {self.fusion_hidden}")
        print(f"    Bidirectional : {self.bidirectional}")
        print(f"    Dropout       : {self.dropout}")
        print(f"    Sequence len  : {self.seq_len}")
        print(f"    Device        : {self.device}")

        if self.model is not None:
            total_p = sum(p.numel() for p in self.model.parameters())
            print(f"    Parameters    : {total_p:,}")

        if self.history:
            print(f"\n  Training:")
            print(f"    Epochs        : {len(self.history.train_losses)}")
            print(f"    Best epoch    : {self.history.best_epoch}")
            print(f"    Best val loss : {self.history.best_val_loss:.6f}")
            print(f"    Time          : {self.history.total_time_secs:.1f}s")

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


# ===================================================================
#  CONVENIENCE FUNCTION
# ===================================================================


def train_hybrid_model(
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
) -> Tuple[HybridTrainer, Dict[str, Any]]:
    """One-call function: train, evaluate, and optionally save."""
    trainer = HybridTrainer(
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


# ===================================================================
#  CLI
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a hybrid RNN + sentiment encoder for stock prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.hybrid_model --demo --cell lstm\n"
            "  python -m src.hybrid_model --demo --cell gru --seq-len 30\n"
            "  python -m src.hybrid_model --input data/processed/features.csv "
            "--cell lstm --save models/hybrid_v1.pt\n"
        ),
    )
    parser.add_argument("--input", type=str, default=None, help="Feature matrix CSV")
    parser.add_argument("--cell", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic price + sentiment data")
    args = parser.parse_args()

    if args.demo:
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

        # Synthetic sentiment
        sent = pd.DataFrame(
            {
                "sent_mean": np.random.uniform(-0.5, 0.5, n),
                "sent_std": np.random.uniform(0, 0.3, n),
                "sent_min": np.random.uniform(-1, 0, n),
                "sent_max": np.random.uniform(0, 1, n),
                "sent_median": np.random.uniform(-0.3, 0.3, n),
                "sent_positive_pct": np.random.uniform(0, 1, n),
                "sent_negative_pct": np.random.uniform(0, 1, n),
                "sent_neutral_pct": np.random.uniform(0, 1, n),
                "sent_count": np.random.randint(1, 20, n),
            },
            index=dates,
        )

        eng = FeatureEngineer()
        features_df = eng.build(prices, sent)

        save_path = args.save or f"models/hybrid_{args.cell}_demo.pt"
        trainer, metrics = train_hybrid_model(
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
        features_df = pd.read_csv(args.input, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(features_df)} rows from {args.input}")

        save_path = args.save or f"models/hybrid_{args.cell}_trained.pt"
        trainer, metrics = train_hybrid_model(
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
