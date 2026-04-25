"""
test_placeholder.py
====================
Placeholder tests to verify the test harness works and basic imports succeed.
Replace these with comprehensive unit/integration tests as modules mature.
"""

import importlib


class TestImports:
    """Verify that all core modules can be imported without errors."""

    def test_import_data_fetcher(self):
        mod = importlib.import_module("src.data_fetcher")
        assert hasattr(mod, "StockDataFetcher")
        assert hasattr(mod, "NewsDataFetcher")

    def test_import_sentiment_analyzer(self):
        mod = importlib.import_module("src.sentiment_analyzer")
        assert hasattr(mod, "SentimentAnalyzer")
        assert hasattr(mod, "score_headlines")
        assert hasattr(mod, "aggregate_daily_sentiment")
        assert hasattr(mod, "build_daily_sentiment")

    def test_import_feature_engineering(self):
        mod = importlib.import_module("src.feature_engineering")
        assert hasattr(mod, "FeatureEngineer")

    def test_import_model_training(self):
        mod = importlib.import_module("src.model_training")
        assert hasattr(mod, "ModelTrainer")

    def test_import_backtesting(self):
        mod = importlib.import_module("src.backtesting")
        assert hasattr(mod, "SimpleBacktester")

    def test_import_deep_learning(self):
        mod = importlib.import_module("src.deep_learning")
        assert hasattr(mod, "DeepLearningTrainer")
        assert hasattr(mod, "StockRNN")
        assert hasattr(mod, "train_deep_model")

    def test_import_news_fetcher(self):
        mod = importlib.import_module("src.news_fetcher")
        assert hasattr(mod, "fetch_news")
        assert hasattr(mod, "NewsSource")
        assert hasattr(mod, "SOURCE_REGISTRY")


class TestFeatureEngineer:
    """Tests for the rewritten feature_engineering module."""

    @staticmethod
    def _make_prices(n=200):
        """Generate synthetic but realistic OHLCV data."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        base = 150 + np.random.randn(n).cumsum()
        df = pd.DataFrame(
            {
                "Open": base + np.random.uniform(-1, 1, n),
                "High": base + np.abs(np.random.randn(n)) * 2,
                "Low": base - np.abs(np.random.randn(n)) * 2,
                "Close": base,
                "Volume": np.random.randint(500_000, 5_000_000, n),
            },
            index=dates,
        )
        df["High"] = df[["Open", "High", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
        return df

    @staticmethod
    def _make_sentiment(n=200):
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        return pd.DataFrame(
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

    def test_build_returns_target(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices())
        assert "target" in result.columns
        assert "target_return" in result.columns
        assert set(result["target"].unique()).issubset({0, 1})

    def test_no_nans_after_build(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices())
        assert result.isna().sum().sum() == 0

    def test_rsi_columns(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        assert "rsi" in result.columns
        assert "rsi_overbought" in result.columns
        assert "rsi_oversold" in result.columns

    def test_macd_columns(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        for col in ("macd", "macd_signal", "macd_hist", "macd_cross"):
            assert col in result.columns

    def test_bollinger_columns(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        for col in ("bb_upper", "bb_lower", "bb_width", "bb_pct_b"):
            assert col in result.columns

    def test_volatility_columns(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        assert "atr" in result.columns
        assert "atr_pct" in result.columns
        assert "gk_volatility" in result.columns
        assert "parkinson_vol" in result.columns
        assert "volatility_5d" in result.columns

    def test_return_features(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        for col in ("return_1d", "return_5d", "log_return_1d",
                     "intraday_return", "overnight_gap"):
            assert col in result.columns

    def test_lagged_returns(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer(lag_depths=[1, 3, 5])
        result = eng.build(self._make_prices(), drop_na=False)
        assert "return_lag_1" in result.columns
        assert "return_lag_3" in result.columns
        assert "return_lag_5" in result.columns

    def test_volume_features(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        for col in ("obv", "volume_zscore", "volume_ratio", "vpt"):
            assert col in result.columns

    def test_rolling_stats(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        assert "roll_skew_5d" in result.columns
        assert "roll_kurt_10d" in result.columns

    def test_calendar_features(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), drop_na=False)
        for col in ("day_of_week", "month", "quarter"):
            assert col in result.columns

    def test_sentiment_merge(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices(), self._make_sentiment())
        assert "sent_mean" in result.columns
        assert "sent_smooth_3d" in result.columns
        assert "sent_momentum_5d" in result.columns

    def test_function_interface(self):
        from src.feature_engineering import build_feature_matrix

        result = build_feature_matrix(self._make_prices(), forecast_horizon=1)
        assert "target" in result.columns
        assert result.isna().sum().sum() == 0

    def test_feature_count_minimum(self):
        from src.feature_engineering import FeatureEngineer

        eng = FeatureEngineer()
        result = eng.build(self._make_prices())
        n_features = len([c for c in result.columns
                          if c not in ("target", "target_return")])
        # Should have at least 60 features
        assert n_features >= 60, f"Only {n_features} features produced"


class TestNewsFetcher:
    """Offline-safe tests for the news_fetcher module (no API calls)."""

    def test_list_sources(self):
        from src.news_fetcher import list_sources

        sources = list_sources()
        assert "rss" in sources
        assert "newsapi" in sources
        assert "finnhub" in sources
        assert "gnews" in sources

    def test_get_source_returns_instance(self):
        from src.news_fetcher import NewsSource, get_source

        source = get_source("rss")
        assert isinstance(source, NewsSource)
        assert source.name == "rss"

    def test_get_source_unknown_raises(self):
        import pytest
        from src.news_fetcher import get_source

        with pytest.raises(KeyError, match="Unknown news source"):
            get_source("nonexistent_api")

    def test_normalize_enforces_columns(self):
        import pandas as pd
        from src.news_fetcher import COLUMNS, RSSSource

        raw = pd.DataFrame(
            {
                "date": ["2024-06-15T10:00:00+00:00"],
                "headline": ["Test headline"],
                "source": ["TestSource"],
                "url": ["https://example.com"],
                "extra_col": ["ignored"],
            }
        )
        result = RSSSource._normalize(raw)
        assert list(result.columns) == COLUMNS
        assert "extra_col" not in result.columns

    def test_ticker_to_query_known(self):
        from src.news_fetcher import NewsSource

        assert "Apple" in NewsSource._ticker_to_query("AAPL")
        assert "Tesla" in NewsSource._ticker_to_query("TSLA")

    def test_ticker_to_query_unknown_passthrough(self):
        from src.news_fetcher import NewsSource

        result = NewsSource._ticker_to_query("ZXYZ")
        assert "ZXYZ" in result

    def test_register_custom_source(self):
        import pandas as pd
        from src.news_fetcher import (
            NewsSource,
            register_source,
            get_source,
            SOURCE_REGISTRY,
        )

        class DummySource(NewsSource):
            name = "dummy"

            def fetch_headlines(self, query, **kw):
                return self._normalize(
                    pd.DataFrame(
                        {
                            "date": ["2024-01-01T00:00:00+00:00"],
                            "headline": [f"Dummy: {query}"],
                            "source": ["DummyAPI"],
                            "url": ["https://dummy.test"],
                        }
                    )
                )

        register_source("dummy", DummySource)
        assert "dummy" in SOURCE_REGISTRY

        src = get_source("dummy")
        df = src.fetch_headlines("test")
        assert len(df) == 1
        assert "Dummy" in df["headline"].iloc[0]

        # Clean up
        del SOURCE_REGISTRY["dummy"]


class TestSentimentAnalyzer:
    """Tests for the sentiment_analyzer module (VADER only — no GPU needed)."""

    def test_vader_score_text(self):
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer(backend="vader")
        result = analyzer.score_text("Stock surges after record earnings!")

        assert "label" in result
        assert "compound" in result
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert result["label"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        assert -1.0 <= result["compound"] <= 1.0

    def test_vader_positive_negative(self):
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer(backend="vader")

        pos = analyzer.score_text("This is a great and amazing product!")
        neg = analyzer.score_text("This is terrible and awful, very bad.")

        assert pos["compound"] > 0
        assert neg["compound"] < 0

    def test_score_texts_batch(self):
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer(backend="vader")
        texts = ["Great results", "Terrible losses", "Markets unchanged"]
        results = analyzer.score_texts(texts)

        assert len(results) == 3
        assert all("compound" in r for r in results)

    def test_score_dataframe(self):
        import pandas as pd
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer(backend="vader")
        df = pd.DataFrame(
            {
                "date": ["2024-06-10", "2024-06-10", "2024-06-11"],
                "headline": [
                    "Stock surges on strong earnings",
                    "Massive recall announced",
                    "Markets close flat",
                ],
            }
        )
        scored = analyzer.score_dataframe(df)

        assert "label" in scored.columns
        assert "compound" in scored.columns
        assert "positive" in scored.columns
        assert "negative" in scored.columns
        assert "neutral" in scored.columns
        assert len(scored) == 3

    def test_aggregate_daily(self):
        import pandas as pd
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer(backend="vader")
        df = pd.DataFrame(
            {
                "date": [
                    "2024-06-10", "2024-06-10", "2024-06-10",
                    "2024-06-11", "2024-06-11",
                ],
                "headline": [
                    "Great earnings beat",
                    "Stock drops on concerns",
                    "Markets flat today",
                    "Analyst upgrades stock",
                    "Revenue misses estimates",
                ],
            }
        )
        scored = analyzer.score_dataframe(df)
        daily = analyzer.aggregate_daily(scored)

        assert len(daily) == 2  # two distinct dates
        assert "sent_mean" in daily.columns
        assert "sent_std" in daily.columns
        assert "sent_positive_pct" in daily.columns
        assert "sent_negative_pct" in daily.columns
        assert "sent_neutral_pct" in daily.columns
        assert "sent_count" in daily.columns

        # Proportions should sum to ~1 for each day
        for _, row in daily.iterrows():
            total = row["sent_positive_pct"] + row["sent_negative_pct"] + row["sent_neutral_pct"]
            assert abs(total - 1.0) < 0.01

    def test_build_daily_sentiment_convenience(self):
        import pandas as pd
        from src.sentiment_analyzer import build_daily_sentiment

        df = pd.DataFrame(
            {
                "date": ["2024-06-10", "2024-06-11"],
                "headline": ["Stock surges", "Stock crashes"],
            }
        )
        daily = build_daily_sentiment(df, backend="vader")

        assert len(daily) == 2
        assert daily.index.name == "date"
        assert "sent_mean" in daily.columns
