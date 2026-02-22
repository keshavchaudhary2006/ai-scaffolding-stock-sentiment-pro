"""
test_hybrid_model.py
====================
Unit tests for the hybrid two-branch model (RNN + Sentiment encoder).
Uses tiny models and few epochs for fast CPU execution.
"""

import importlib
import numpy as np
import pandas as pd
import pytest
import torch


# -- Helpers ---------------------------------------------------------------


def _make_feature_matrix_with_sentiment(n: int = 200) -> pd.DataFrame:
    """Build a synthetic feature matrix with sentiment via FeatureEngineer."""
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

    sent = pd.DataFrame(
        {
            "sent_mean": np.random.uniform(-0.5, 0.5, n),
            "sent_std": np.random.uniform(0, 0.3, n),
            "sent_positive_pct": np.random.uniform(0, 1, n),
            "sent_negative_pct": np.random.uniform(0, 1, n),
            "sent_neutral_pct": np.random.uniform(0, 1, n),
            "sent_count": np.random.randint(1, 20, n),
        },
        index=dates,
    )

    eng = FeatureEngineer()
    return eng.build(prices, sent)


def _make_feature_matrix_no_sentiment(n: int = 200) -> pd.DataFrame:
    """Feature matrix without sentiment columns."""
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


# -- Import Tests ---------------------------------------------------------


class TestImports:
    def test_import_module(self):
        mod = importlib.import_module("src.hybrid_model")
        assert hasattr(mod, "HybridTrainer")
        assert hasattr(mod, "HybridStockModel")
        assert hasattr(mod, "DualBranchDataset")
        assert hasattr(mod, "train_hybrid_model")

    def test_import_dataclasses(self):
        mod = importlib.import_module("src.hybrid_model")
        assert hasattr(mod, "HybridEvalResult")
        assert hasattr(mod, "HybridHistory")


# -- DualBranchDataset Tests ----------------------------------------------


class TestDualBranchDataset:
    def test_sequence_count(self):
        from src.hybrid_model import DualBranchDataset

        n, p_feats, s_feats, seq_len = 50, 10, 5, 8
        ds = DualBranchDataset(
            np.random.randn(n, p_feats).astype(np.float32),
            np.random.randn(n, s_feats).astype(np.float32),
            np.random.randint(0, 2, n).astype(np.float32),
            seq_len=seq_len,
        )
        assert len(ds) == n - seq_len + 1

    def test_output_shapes(self):
        from src.hybrid_model import DualBranchDataset

        n, p_feats, s_feats, seq_len = 50, 10, 5, 8
        ds = DualBranchDataset(
            np.random.randn(n, p_feats).astype(np.float32),
            np.random.randn(n, s_feats).astype(np.float32),
            np.random.randint(0, 2, n).astype(np.float32),
            seq_len=seq_len,
        )
        xp, xs, y = ds[0]
        assert xp.shape == (seq_len, p_feats)
        assert xs.shape == (seq_len, s_feats)
        assert y.shape == ()


# -- HybridStockModel Tests -----------------------------------------------


class TestHybridStockModel:
    def test_forward_lstm(self):
        from src.hybrid_model import HybridStockModel

        model = HybridStockModel(
            price_input_size=10, sent_input_size=5,
            hidden_size=16, num_layers=1, dropout=0.0, cell_type="lstm",
        )
        xp = torch.randn(4, 8, 10)
        xs = torch.randn(4, 8, 5)
        out = model(xp, xs)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_gru(self):
        from src.hybrid_model import HybridStockModel

        model = HybridStockModel(
            price_input_size=10, sent_input_size=5,
            hidden_size=16, num_layers=1, dropout=0.0, cell_type="gru",
        )
        xp = torch.randn(4, 8, 10)
        xs = torch.randn(4, 8, 5)
        out = model(xp, xs)
        assert out.shape == (4,)

    def test_bidirectional(self):
        from src.hybrid_model import HybridStockModel

        model = HybridStockModel(
            price_input_size=10, sent_input_size=5,
            hidden_size=16, num_layers=1, dropout=0.0,
            cell_type="lstm", bidirectional=True,
        )
        xp = torch.randn(4, 8, 10)
        xs = torch.randn(4, 8, 5)
        out = model(xp, xs)
        assert out.shape == (4,)


# -- Column Splitting Tests ------------------------------------------------


class TestColumnSplitting:
    def test_splits_correctly(self):
        from src.hybrid_model import HybridTrainer

        df = _make_feature_matrix_with_sentiment()
        price_cols, sent_cols = HybridTrainer._split_columns(df)

        assert len(price_cols) > 0
        assert len(sent_cols) > 0
        assert all(c.startswith("sent_") for c in sent_cols)
        assert "target" not in price_cols
        assert "target" not in sent_cols

    def test_no_sentiment(self):
        from src.hybrid_model import HybridTrainer

        df = _make_feature_matrix_no_sentiment()
        price_cols, sent_cols = HybridTrainer._split_columns(df)
        assert len(sent_cols) == 0


# -- HybridTrainer End-to-End Tests ----------------------------------------


class TestHybridTrainer:
    @pytest.fixture(scope="class")
    def features_df(self):
        return _make_feature_matrix_with_sentiment(n=200)

    def test_train_with_sentiment(self, features_df):
        from src.hybrid_model import HybridTrainer

        trainer = HybridTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=3, batch_size=32, patience=10,
        )
        trainer.train(features_df)

        assert trainer.model is not None
        assert trainer.price_scaler is not None
        assert trainer.sent_scaler is not None
        assert len(trainer.price_cols) > 0
        assert len(trainer.sent_cols) > 0
        assert trainer.train_metrics is not None
        assert trainer.val_metrics is not None
        assert trainer.test_metrics is not None

    def test_train_without_sentiment(self):
        from src.hybrid_model import HybridTrainer

        df = _make_feature_matrix_no_sentiment(n=200)
        trainer = HybridTrainer(
            cell_type="gru", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(df)
        assert trainer.model is not None
        # Should have created dummy sentiment column
        assert len(trainer.sent_cols) == 1

    def test_evaluate_dict(self, features_df):
        from src.hybrid_model import HybridTrainer

        trainer = HybridTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        metrics = trainer.evaluate()

        assert isinstance(metrics, dict)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics

    def test_predict(self, features_df):
        from src.hybrid_model import HybridTrainer

        trainer = HybridTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        preds = trainer.predict(features_df)
        assert isinstance(preds, np.ndarray)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self, features_df):
        from src.hybrid_model import HybridTrainer

        trainer = HybridTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        proba = trainer.predict_proba(features_df)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_history(self, features_df):
        from src.hybrid_model import HybridTrainer

        trainer = HybridTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=3, batch_size=32, patience=10,
        )
        trainer.train(features_df)
        h = trainer.history
        assert len(h.train_losses) == 3
        assert len(h.val_losses) == 3
        assert h.best_epoch >= 1
        assert h.total_time_secs > 0


# -- Save / Load Tests ----------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        from src.hybrid_model import HybridTrainer

        df = _make_feature_matrix_with_sentiment(n=150)
        trainer = HybridTrainer(
            cell_type="lstm", seq_len=5, hidden_size=16,
            num_layers=1, dropout=0.0, epochs=2, batch_size=32, patience=10,
        )
        trainer.train(df)

        save_path = tmp_path / "hybrid_test.pt"
        trainer.save(save_path)
        assert save_path.exists()
        assert save_path.with_suffix(".pt.meta.json").exists()

        loaded = HybridTrainer.load(save_path)
        assert loaded.model is not None
        assert loaded.cell_type == "lstm"
        assert loaded.price_cols == trainer.price_cols
        assert loaded.sent_cols == trainer.sent_cols

        proba_orig = trainer.predict_proba(df)
        proba_loaded = loaded.predict_proba(df)
        np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-5)


# -- Convenience Function Tests -------------------------------------------


class TestConvenienceFunction:
    def test_train_hybrid_model(self, tmp_path):
        from src.hybrid_model import train_hybrid_model

        df = _make_feature_matrix_with_sentiment(n=150)
        save_path = tmp_path / "hybrid_convenience.pt"

        trainer, metrics = train_hybrid_model(
            df, cell_type="gru", seq_len=5, hidden_size=16,
            num_layers=1, epochs=2, batch_size=32, save_path=save_path,
        )
        assert trainer.model is not None
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert save_path.exists()
