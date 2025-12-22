"""
test_deep_learning.py
=====================
Unit tests for the deep learning (LSTM / GRU) training module.

All tests use small synthetic data and tiny models to run fast on CPU.
"""

import importlib
import numpy as np
import pandas as pd
import pytest
import torch


# ── Helpers ───────────────────────────────────────────────────────────


def _make_feature_matrix(n: int = 200) -> pd.DataFrame:
    """Build a small synthetic feature matrix via FeatureEngineer."""
    from src.feature_engineering import FeatureEngineer

    np.random.seed(42)
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
    return eng.build(prices)


# ── Import Tests ──────────────────────────────────────────────────────


class TestImports:
    """Verify module imports and key symbols exist."""

    def test_import_module(self):
        mod = importlib.import_module("src.deep_learning")
        assert hasattr(mod, "DeepLearningTrainer")
        assert hasattr(mod, "StockRNN")
        assert hasattr(mod, "SequenceDataset")
        assert hasattr(mod, "train_deep_model")

    def test_import_eval_result(self):
        mod = importlib.import_module("src.deep_learning")
        assert hasattr(mod, "DLEvalResult")
        assert hasattr(mod, "TrainingHistory")


# ── SequenceDataset Tests ────────────────────────────────────────────


class TestSequenceDataset:
    """Test sliding-window sequence generation."""

    def test_sequence_count(self):
        from src.deep_learning import SequenceDataset

        n, feats, seq_len = 50, 10, 5
        X = np.random.randn(n, feats).astype(np.float32)
        y = np.random.randint(0, 2, n).astype(np.float32)

        ds = SequenceDataset(X, y, seq_len=seq_len)
        assert len(ds) == n - seq_len + 1

    def test_sequence_shape(self):
        from src.deep_learning import SequenceDataset

        n, feats, seq_len = 50, 10, 5
        X = np.random.randn(n, feats).astype(np.float32)
        y = np.random.randint(0, 2, n).astype(np.float32)

        ds = SequenceDataset(X, y, seq_len=seq_len)
        x_sample, y_sample = ds[0]
        assert x_sample.shape == (seq_len, feats)
        assert y_sample.shape == ()

    def test_label_alignment(self):
        """Label at position i should correspond to the *end* of the window."""
        from src.deep_learning import SequenceDataset

        n, feats, seq_len = 20, 3, 5
        X = np.random.randn(n, feats).astype(np.float32)
        y = np.arange(n, dtype=np.float32)  # use indices as labels

        ds = SequenceDataset(X, y, seq_len=seq_len)
        _, label_0 = ds[0]
        assert label_0.item() == seq_len - 1  # index 4 for seq_len=5

        _, label_1 = ds[1]
        assert label_1.item() == seq_len  # index 5

    def test_seq_len_one(self):
        from src.deep_learning import SequenceDataset

        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, 10).astype(np.float32)
        ds = SequenceDataset(X, y, seq_len=1)
        assert len(ds) == 10


# ── StockRNN Tests ───────────────────────────────────────────────────


class TestStockRNN:
    """Test model architecture and forward pass."""

    def test_lstm_forward(self):
        from src.deep_learning import StockRNN

        model = StockRNN(input_size=10, hidden_size=16, num_layers=1,
                         dropout=0.0, cell_type="lstm")
        batch = torch.randn(4, 5, 10)  # (batch, seq_len, features)
        out = model(batch)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gru_forward(self):
        from src.deep_learning import StockRNN

        model = StockRNN(input_size=10, hidden_size=16, num_layers=1,
                         dropout=0.0, cell_type="gru")
        batch = torch.randn(4, 5, 10)
        out = model(batch)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_bidirectional(self):
        from src.deep_learning import StockRNN

        model = StockRNN(input_size=10, hidden_size=16, num_layers=1,
                         dropout=0.0, cell_type="lstm", bidirectional=True)
        batch = torch.randn(4, 5, 10)
        out = model(batch)
        assert out.shape == (4,)

    def test_multi_layer(self):
        from src.deep_learning import StockRNN

        model = StockRNN(input_size=10, hidden_size=16, num_layers=3,
                         dropout=0.1, cell_type="gru")
        batch = torch.randn(4, 5, 10)
        out = model(batch)
        assert out.shape == (4,)


# ── DeepLearningTrainer Tests ────────────────────────────────────────


class TestDeepLearningTrainer:
    """End-to-end training tests (small model, few epochs)."""

    @pytest.fixture(scope="class")
    def features_df(self):
        return _make_feature_matrix(n=200)

    def test_train_lstm(self, features_df):
        from src.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer(
            cell_type="lstm",
            seq_len=5,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            epochs=3,
            batch_size=32,
            patience=10,
        )
        trainer.train(features_df)

        assert trainer.model is not None
        assert trainer.scaler is not None
        assert len(trainer.feature_names) > 0
        assert trainer.train_metrics is not None
        assert trainer.val_metrics is not None
        assert trainer.test_metrics is not None

    def test_train_gru(self, features_df):
        from src.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer(
            cell_type="gru",
            seq_len=5,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            epochs=3,
            batch_size=32,
            patience=10,
        )
        trainer.train(features_df)
        assert trainer.model is not None

    def test_evaluate_returns_dict(self, features_df):
        from src.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        metrics = trainer.evaluate()

        assert isinstance(metrics, dict)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics

    def test_predict(self, features_df):
        from src.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        preds = trainer.predict(features_df)

        assert isinstance(preds, np.ndarray)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, features_df):
        from src.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        proba = trainer.predict_proba(features_df)

        assert isinstance(proba, np.ndarray)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_history_populated(self, features_df):
        from src.deep_learning import DeepLearningTrainer

        trainer = DeepLearningTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=3, batch_size=32, patience=10,
        )
        trainer.train(features_df)

        h = trainer.history
        assert h is not None
        assert len(h.train_losses) == 3
        assert len(h.val_losses) == 3
        assert len(h.train_accs) == 3
        assert len(h.val_accs) == 3
        assert h.best_epoch >= 1
        assert h.total_time_secs > 0


# ── Save/Load Tests ─────────────────────────────────────────────────


class TestSaveLoad:
    """Test model persistence and loading."""

    def test_save_and_load(self, tmp_path):
        from src.deep_learning import DeepLearningTrainer

        features_df = _make_feature_matrix(n=150)
        trainer = DeepLearningTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)

        save_path = tmp_path / "test_model.pt"
        trainer.save(save_path)

        assert save_path.exists()
        assert save_path.with_suffix(".pt.meta.json").exists()

        # Load and verify
        loaded = DeepLearningTrainer.load(save_path)
        assert loaded.model is not None
        assert loaded.cell_type == "lstm"
        assert loaded.seq_len == 5
        assert loaded.feature_names == trainer.feature_names

        # Predictions should match
        proba_orig = trainer.predict_proba(features_df)
        proba_loaded = loaded.predict_proba(features_df)
        np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-5)


# ── Convenience Function Tests ──────────────────────────────────────


class TestConvenienceFunction:
    """Test the train_deep_model one-call function."""

    def test_train_deep_model(self, tmp_path):
        from src.deep_learning import train_deep_model

        features_df = _make_feature_matrix(n=150)
        save_path = tmp_path / "convenience_model.pt"

        trainer, metrics = train_deep_model(
            features_df,
            cell_type="gru",
            seq_len=5,
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
            save_path=save_path,
        )

        assert trainer.model is not None
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert save_path.exists()
