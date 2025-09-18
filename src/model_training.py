"""
model_training.py
=================
Train, evaluate, and persist baseline ML models for stock-direction
prediction using the feature matrix produced by ``feature_engineering.py``.

Supported models
----------------
- **Logistic Regression** — linear baseline (fast, interpretable)
- **Random Forest** — non-linear ensemble baseline
- **XGBoost** — gradient-boosted trees (production default)
- **LightGBM** — alternative gradient boosting

Data splitting
--------------
All splits are **strictly chronological** (no shuffling) to prevent
look-ahead bias::

    ┌────────────── 60% ──────────────┐┌── 20% ──┐┌── 20% ──┐
    │           TRAIN                 ││   VAL   ││   TEST  │
    └─────────────────────────────────┘└─────────┘└─────────┘
    oldest                                              newest

The validation set is used for:
- Early-stopping in tree-based models
- Intermediate metric reporting
- Walk-forward cross-validation

Quick-start
-----------
    from src.model_training import ModelTrainer

    trainer = ModelTrainer(model_type="xgboost")
    trainer.train(features_df)
    metrics = trainer.evaluate()
    trainer.save("models/xgb_v1.joblib")

CLI
---
    python -m src.model_training --model xgboost --demo
    python -m src.model_training --input data/processed/features.csv --model random_forest
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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

ModelType = Literal["logistic_regression", "random_forest", "xgboost", "lightgbm"]

# Columns that should never be used as features
_NON_FEATURE_COLUMNS = {"target", "target_return"}


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION RESULT
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """Container for evaluation metrics on a single data split."""

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
#  MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════


class ModelTrainer:
    """End-to-end model training pipeline with chronological splitting.

    Parameters
    ----------
    model_type : str
        One of ``"logistic_regression"``, ``"random_forest"``,
        ``"xgboost"``, ``"lightgbm"``.
    val_size : float
        Fraction of data for the validation set (default 0.15).
    test_size : float
        Fraction of data for the final test set (default 0.20).
    random_state : int
        Seed for reproducibility.
    n_cv_splits : int
        Number of walk-forward CV folds.
    scale_features : bool
        If True, apply ``StandardScaler`` before training.  Recommended
        for logistic regression; tree-based models don't require it.
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        val_size: float = 0.15,
        test_size: float = 0.20,
        random_state: int = 42,
        n_cv_splits: int = 5,
        scale_features: bool = False,
    ) -> None:
        self.model_type = model_type.lower()
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.n_cv_splits = n_cv_splits
        self.scale_features = scale_features

        # State populated after training
        self.model: Any = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.train_metrics: Optional[EvalResult] = None
        self.val_metrics: Optional[EvalResult] = None
        self.test_metrics: Optional[EvalResult] = None

    # ------------------------------------------------------------------
    # Chronological 3-way split
    # ------------------------------------------------------------------

    def _split(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> None:
        """Split data chronologically into train / val / test.

        The data is **never shuffled** — the earliest rows go to train,
        then validation, then test.  This mirrors real-world deployment
        where you only ever predict the future.
        """
        n = len(df)
        test_start = int(n * (1 - self.test_size))
        val_start = int(n * (1 - self.test_size - self.val_size))

        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]

        # Identify feature columns (exclude target + target_return)
        feature_cols = [
            c for c in df.columns if c not in _NON_FEATURE_COLUMNS
        ]
        self.feature_names = feature_cols

        self.X_train, self.y_train = train_df[feature_cols], train_df[target_col]
        self.X_val, self.y_val = val_df[feature_cols], val_df[target_col]
        self.X_test, self.y_test = test_df[feature_cols], test_df[target_col]

        logger.info(
            f"Chronological split → "
            f"train={len(self.X_train)} ({val_start/n:.0%})  "
            f"val={len(self.X_val)} ({(test_start-val_start)/n:.0%})  "
            f"test={len(self.X_test)} ({(n-test_start)/n:.0%})  "
            f"features={len(feature_cols)}"
        )

        # Date ranges for logging
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

        # Class balance
        for name, y in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            pos_pct = y.mean() * 100
            logger.debug(f"  {name} class balance: {pos_pct:.1f}% positive")

    # ------------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------------

    def _apply_scaling(self) -> None:
        """Fit scaler on train, transform train/val/test."""
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index,
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_names,
            index=self.X_val.index,
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index,
        )
        logger.info("Applied StandardScaler (fit on train only)")

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _build_model(self, params: Optional[Dict] = None) -> Any:
        """Instantiate the underlying sklearn/xgboost/lightgbm model.

        Parameters
        ----------
        params : dict | None
            Override default hyperparameters.

        Returns
        -------
        Classifier instance (unfitted).
        """
        p = params or {}

        if self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            defaults = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": self.random_state,
            }
            defaults.update(p)
            return LogisticRegression(**defaults)

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            defaults = {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            defaults.update(p)
            return RandomForestClassifier(**defaults)

        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier

            defaults = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "early_stopping_rounds": 30,
            }
            defaults.update(p)
            return XGBClassifier(**defaults)

        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            defaults = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
                "verbose": -1,
            }
            defaults.update(p)
            return LGBMClassifier(**defaults)

        else:
            available = ", ".join(
                ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
            )
            raise ValueError(
                f"Unknown model_type '{self.model_type}'.  "
                f"Available: {available}"
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        params: Optional[Dict] = None,
    ) -> "ModelTrainer":
        """Train the model on the feature matrix.

        Steps:
        1. Chronological train/val/test split.
        2. Optional feature scaling.
        3. Model instantiation.
        4. Fit — with early stopping on val set for tree models.
        5. Evaluate on train + val + test splits.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with ``target`` (and optionally ``target_return``)
            columns, output from ``FeatureEngineer.build()``.
        target_col : str
            Name of the binary target column.
        params : dict | None
            Override model hyperparameters.

        Returns
        -------
        ModelTrainer
            Self, for method chaining.
        """
        # 1 — Split
        self._split(df, target_col)

        # 2 — Scale (auto-enable for logistic regression if not explicitly set)
        if self.scale_features or self.model_type == "logistic_regression":
            self._apply_scaling()

        # 3 — Build model
        self.model = self._build_model(params)
        logger.info(f"Training {self.model_type} model …")

        # 4 — Fit (with early stopping for boosted trees)
        if self.model_type == "xgboost":
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )
            best_iter = getattr(self.model, "best_iteration", None)
            if best_iter is not None:
                logger.info(f"XGBoost early stop at iteration {best_iter}")

        elif self.model_type == "lightgbm":
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
            )
        else:
            # Logistic Regression / Random Forest
            self.model.fit(self.X_train, self.y_train)

        logger.info("Training complete ✓")

        # 5 — Evaluate all splits
        self.train_metrics = self._evaluate_split(
            self.X_train, self.y_train, "train"
        )
        self.val_metrics = self._evaluate_split(
            self.X_val, self.y_val, "val"
        )
        self.test_metrics = self._evaluate_split(
            self.X_test, self.y_test, "test"
        )

        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_name: str,
    ) -> EvalResult:
        """Compute classification metrics on a given split."""
        preds = self.model.predict(X)
        proba = (
            self.model.predict_proba(X)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        roc = None
        if proba is not None:
            try:
                roc = float(roc_auc_score(y, proba))
            except ValueError:
                roc = None  # single class in split

        result = EvalResult(
            split_name=split_name,
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1=float(f1_score(y, preds, zero_division=0)),
            roc_auc=roc,
            confusion=confusion_matrix(y, preds).tolist(),
            report=classification_report(y, preds, zero_division=0),
        )
        logger.info(result.summary_line())
        return result

    def evaluate(self) -> Dict[str, Any]:
        """Return test-set metrics as a dict (backward-compatible API).

        Returns
        -------
        dict
            Keys: accuracy, precision, recall, f1, roc_auc, report,
            confusion_matrix.
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
    # Walk-forward Cross-Validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Dict[str, Any]:
        """Run walk-forward CV and return per-fold + average metrics.

        Walk-forward CV respects temporal ordering: each fold uses only
        past data for training and a subsequent window for validation.

        Returns
        -------
        dict
            ``mean_accuracy``, ``mean_f1``, ``fold_results`` (list of
            per-fold metric dicts).
        """
        feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]
        X, y = df[feature_cols], df[target_col]
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        fold_results: List[Dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # Scale if needed
            if self.scale_features or self.model_type == "logistic_regression":
                scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    scaler.fit_transform(X_tr),
                    columns=feature_cols,
                    index=X_tr.index,
                )
                X_va = pd.DataFrame(
                    scaler.transform(X_va),
                    columns=feature_cols,
                    index=X_va.index,
                )

            model = self._build_model()

            if self.model_type == "xgboost":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
            elif self.model_type == "lightgbm":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                )
            else:
                model.fit(X_tr, y_tr)

            preds = model.predict(X_va)
            proba = (
                model.predict_proba(X_va)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            acc = float(accuracy_score(y_va, preds))
            f1 = float(f1_score(y_va, preds, zero_division=0))
            roc = None
            if proba is not None:
                try:
                    roc = float(roc_auc_score(y_va, proba))
                except ValueError:
                    pass

            fold_results.append({"fold": fold, "accuracy": acc, "f1": f1, "roc_auc": roc})
            logger.info(
                f"  Fold {fold}/{self.n_cv_splits}: "
                f"acc={acc:.4f}  f1={f1:.4f}  "
                f"auc={roc:.4f if roc else 'N/A'}"
            )

        mean_acc = float(np.mean([r["accuracy"] for r in fold_results]))
        mean_f1 = float(np.mean([r["f1"] for r in fold_results]))
        logger.info(
            f"CV summary → mean_acc={mean_acc:.4f}  mean_f1={mean_f1:.4f}"
        )
        return {
            "mean_accuracy": mean_acc,
            "mean_f1": mean_f1,
            "fold_results": fold_results,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Extract feature importances from the trained model.

        Works for tree-based models (Random Forest, XGBoost, LightGBM)
        and for Logistic Regression (absolute coefficient values).

        Parameters
        ----------
        top_n : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.  Sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .feature_importance()")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            raise RuntimeError(
                f"Model type '{self.model_type}' does not expose "
                "feature importances."
            )

        fi = (
            pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return fi

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as training).

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1).
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .predict()")

        X_in = X[self.feature_names]
        if self.scaler is not None:
            X_in = pd.DataFrame(
                self.scaler.transform(X_in),
                columns=self.feature_names,
                index=X_in.index,
            )
        return self.model.predict(X_in)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for new data.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` — probabilities for class 0 and 1.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .predict_proba()")
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                f"Model '{self.model_type}' does not support predict_proba."
            )

        X_in = X[self.feature_names]
        if self.scaler is not None:
            X_in = pd.DataFrame(
                self.scaler.transform(X_in),
                columns=self.feature_names,
                index=X_in.index,
            )
        return self.model.predict_proba(X_in)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path] = "models/model.joblib") -> Path:
        """Save the trained model, scaler, feature names, and metrics.

        Creates two files:
        - ``<path>`` — the joblib-serialised model + scaler bundle
        - ``<path>.meta.json`` — human-readable metadata

        Parameters
        ----------
        path : str | Path
            Output path for the model file.

        Returns
        -------
        Path
            The path the model was saved to.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .save()")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Bundle model + scaler + feature names
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }
        joblib.dump(bundle, path)
        logger.info(f"Model saved → {path}")

        # Save human-readable metadata
        meta = {
            "model_type": self.model_type,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "scale_features": self.scaler is not None,
            "train_size": len(self.X_train) if self.X_train is not None else 0,
            "val_size": len(self.X_val) if self.X_val is not None else 0,
            "test_size": len(self.X_test) if self.X_test is not None else 0,
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
    def load(cls, path: Union[str, Path]) -> "ModelTrainer":
        """Load a previously saved model bundle.

        Parameters
        ----------
        path : str | Path
            Path to the ``.joblib`` file.

        Returns
        -------
        ModelTrainer
            A trainer instance with the model, scaler, and feature names
            restored.  You can call ``.predict()`` / ``.predict_proba()``
            immediately.
        """
        bundle = joblib.load(path)
        logger.info(f"Model loaded ← {path}")

        trainer = cls(model_type=bundle.get("model_type", "xgboost"))
        trainer.model = bundle["model"]
        trainer.scaler = bundle.get("scaler")
        trainer.feature_names = bundle.get("feature_names", [])
        return trainer

    # ------------------------------------------------------------------
    # Pretty-print results
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted summary of train / val / test metrics."""
        if self.test_metrics is None:
            raise RuntimeError("Call .train() before .print_report()")

        print(f"\n{'=' * 72}")
        print(f"  Model Training Report -- {self.model_type.upper()}")
        print(f"{'=' * 72}")
        print(f"\n  Data splits:")
        print(f"    Train : {len(self.X_train):>6} rows")
        print(f"    Val   : {len(self.X_val):>6} rows")
        print(f"    Test  : {len(self.X_test):>6} rows")
        print(f"    Total : {len(self.X_train)+len(self.X_val)+len(self.X_test):>6} rows")
        print(f"    Features: {len(self.feature_names)}")

        print(f"\n{'-' * 72}")
        print("  Performance Metrics:")
        print(f"{'-' * 72}")
        for m in [self.train_metrics, self.val_metrics, self.test_metrics]:
            print(f"    {m.summary_line()}")

        print(f"\n{'-' * 72}")
        print("  Test Set Classification Report:")
        print(f"{'-' * 72}")
        for line in self.test_metrics.report.split("\n"):
            print(f"    {line}")

        print(f"\n{'-' * 72}")
        print("  Test Set Confusion Matrix:")
        print(f"{'-' * 72}")
        cm = np.array(self.test_metrics.confusion)
        print(f"                Predicted")
        print(f"              {'DOWN':>6}  {'UP':>6}")
        print(f"    Actual DOWN {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"    Actual UP   {cm[1][0]:>6}  {cm[1][1]:>6}")

        # Feature importance (top 15)
        try:
            fi = self.feature_importance(top_n=15)
            print(f"\n{'-' * 72}")
            print("  Top 15 Feature Importances:")
            print(f"{'-' * 72}")
            for _, row in fi.iterrows():
                bar_len = int(row["importance"] / fi["importance"].max() * 30)
                bar = "#" * bar_len
                print(f"    {row['feature']:<30} {row['importance']:.4f}  {bar}")
        except RuntimeError:
            pass

        print(f"\n{'=' * 72}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def train_and_evaluate(
    df: pd.DataFrame,
    model_type: ModelType = "xgboost",
    target_col: str = "target",
    val_size: float = 0.15,
    test_size: float = 0.20,
    params: Optional[Dict] = None,
    save_path: Optional[Union[str, Path]] = None,
    run_cv: bool = False,
) -> Tuple[ModelTrainer, Dict[str, Any]]:
    """One-call function: train, evaluate, and optionally save.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix from ``FeatureEngineer.build()``.
    model_type : str
        Model to train.
    target_col : str
        Binary target column name.
    val_size, test_size : float
        Split proportions.
    params : dict | None
        Model hyperparameters.
    save_path : str | Path | None
        Save the model to this path if provided.
    run_cv : bool
        Also run walk-forward cross-validation.

    Returns
    -------
    tuple[ModelTrainer, dict]
        The fitted trainer and a dict of test metrics.
    """
    trainer = ModelTrainer(
        model_type=model_type,
        val_size=val_size,
        test_size=test_size,
    )
    trainer.train(df, target_col=target_col, params=params)
    trainer.print_report()

    if run_cv:
        cv_results = trainer.cross_validate(df, target_col=target_col)
        print(f"  Walk-forward CV mean accuracy: {cv_results['mean_accuracy']:.4f}")
        print(f"  Walk-forward CV mean F1:       {cv_results['mean_f1']:.4f}")

    metrics = trainer.evaluate()

    if save_path:
        trainer.save(save_path)

    return trainer, metrics


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a baseline ML model for stock direction prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.model_training --demo --model xgboost\n"
            "  python -m src.model_training --demo --model logistic_regression\n"
            "  python -m src.model_training --demo --model random_forest --cv\n"
            "  python -m src.model_training --input data/processed/features.csv "
            "--model xgboost --save models/xgb_v1.joblib\n"
        ),
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to feature matrix CSV (output of feature_engineering.py)",
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        help="Model type (default: xgboost)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the trained model (.joblib)",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Also run walk-forward cross-validation",
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15,
        help="Validation set fraction (default: 0.15)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Test set fraction (default: 0.20)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic data (no input file needed)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Train all models and compare (demo mode only)",
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
            # -- Compare all models --
            print(f"\n{'=' * 72}")
            print("  MODEL COMPARISON -- Synthetic Data")
            print(f"{'=' * 72}")

            results = {}
            for model_name in ["logistic_regression", "random_forest", "xgboost"]:
                print(f"\n{'-' * 72}")
                print(f"  Training: {model_name}")
                print(f"{'-' * 72}")
                trainer = ModelTrainer(
                    model_type=model_name,
                    val_size=args.val_size,
                    test_size=args.test_size,
                )
                trainer.train(features_df)
                m = trainer.evaluate()
                results[model_name] = m

            # Summary table
            print(f"\n{'=' * 72}")
            print("  COMPARISON SUMMARY")
            print(f"{'=' * 72}")
            print(
                f"  {'Model':<25} {'Acc':>8} {'Prec':>8} "
                f"{'Rec':>8} {'F1':>8} {'AUC':>8}"
            )
            print(f"  {'-' * 65}")
            for name, m in results.items():
                auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] else "N/A"
                print(
                    f"  {name:<25} {m['accuracy']:>8.4f} "
                    f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                    f"{m['f1']:>8.4f} {auc:>8}"
                )
            print(f"{'=' * 72}\n")

        else:
            # ── Single model demo ──
            save_path = args.save or f"models/{args.model}_demo.joblib"
            trainer, metrics = train_and_evaluate(
                features_df,
                model_type=args.model,
                save_path=save_path,
                run_cv=args.cv,
                val_size=args.val_size,
                test_size=args.test_size,
            )

    elif args.input:
        # ── Train on real data ──
        features_df = pd.read_csv(args.input, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(features_df)} rows from {args.input}")

        save_path = args.save or f"models/{args.model}_trained.joblib"
        trainer, metrics = train_and_evaluate(
            features_df,
            model_type=args.model,
            save_path=save_path,
            run_cv=args.cv,
            val_size=args.val_size,
            test_size=args.test_size,
        )

    else:
        parser.print_help()
