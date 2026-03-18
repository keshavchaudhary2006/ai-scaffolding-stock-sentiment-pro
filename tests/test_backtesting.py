"""
test_backtesting.py
===================
Unit tests for the backtesting module (long/flat strategy with baselines).
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    SimpleBacktester,
    StrategyResult,
    TradeRecord,
    backtest_signals,
)


# -- Helpers ---------------------------------------------------------------


def _make_prices(n: int = 252, seed: int = 42) -> pd.Series:
    """Generate synthetic trending prices."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    daily_ret = np.random.normal(0.0003, 0.015, n)
    cum = (1 + pd.Series(daily_ret)).cumprod()
    return pd.Series(100.0 * cum.values, index=dates, name="Close")


def _make_signals(n: int = 252, seed: int = 42, p_long: float = 0.5) -> pd.Series:
    """Generate random binary signals."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    vals = np.random.choice([0.0, 1.0], size=n, p=[1 - p_long, p_long])
    return pd.Series(vals, index=dates, name="signal")


# -- Basic Instantiation --------------------------------------------------


class TestSimpleBacktester:
    def test_default_init(self):
        bt = SimpleBacktester()
        assert bt.initial_cash == 100_000
        assert bt.commission_pct == 0.001
        assert bt.result is None

    def test_custom_init(self):
        bt = SimpleBacktester(
            initial_cash=50_000,
            commission_pct=0.002,
            risk_free_rate=0.05,
        )
        assert bt.initial_cash == 50_000
        assert bt.commission_pct == 0.002
        assert bt.risk_free_rate == 0.05


# -- Core Run Tests --------------------------------------------------------


class TestRun:
    def test_run_returns_result(self):
        bt = SimpleBacktester()
        prices = _make_prices()
        signals = _make_signals()
        result = bt.run(prices, signals)

        assert isinstance(result, StrategyResult)
        assert result.name == "Model Strategy"
        assert len(result.equity_curve) == len(prices)
        assert len(result.daily_returns) == len(prices)

    def test_result_has_all_metrics(self):
        bt = SimpleBacktester()
        result = bt.run(_make_prices(), _make_signals())

        for attr in [
            "total_return_pct", "annual_return_pct", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "hit_rate_pct", "profit_factor", "win_rate_pct",
            "avg_win_pct", "avg_loss_pct", "total_trades",
            "total_commission", "exposure_pct",
        ]:
            assert hasattr(result, attr), f"Missing: {attr}"

    def test_equity_starts_at_initial_cash(self):
        bt = SimpleBacktester(initial_cash=50_000)
        result = bt.run(_make_prices(), _make_signals())
        # First bar equity ~ initial_cash (may differ slightly due to returns)
        assert abs(result.equity_curve.iloc[0] - 50_000) < 1_000

    def test_all_long_matches_market(self):
        """Always-long signal should track the market (minus tiny commission)."""
        bt = SimpleBacktester(commission_pct=0.0)  # no commission
        prices = _make_prices()
        signals = pd.Series(1.0, index=prices.index)
        result = bt.run(prices, signals)

        market_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        # Should be very close (vectorised approximation)
        assert abs(result.total_return_pct - market_return) < 2.0

    def test_all_flat_zero_return(self):
        """Always-flat signal should yield zero return."""
        bt = SimpleBacktester()
        prices = _make_prices()
        signals = pd.Series(0.0, index=prices.index)
        result = bt.run(prices, signals)
        assert abs(result.total_return_pct) < 0.01
        assert result.total_trades == 0

    def test_max_drawdown_non_positive(self):
        bt = SimpleBacktester()
        result = bt.run(_make_prices(), _make_signals())
        assert result.max_drawdown_pct <= 0

    def test_hit_rate_range(self):
        bt = SimpleBacktester()
        result = bt.run(_make_prices(), _make_signals())
        assert 0 <= result.hit_rate_pct <= 100

    def test_exposure_range(self):
        bt = SimpleBacktester()
        result = bt.run(_make_prices(), _make_signals())
        assert 0 <= result.exposure_pct <= 100


# -- Commission Tests ------------------------------------------------------


class TestCommission:
    def test_zero_commission(self):
        bt = SimpleBacktester(commission_pct=0.0)
        result = bt.run(_make_prices(), _make_signals())
        assert result.total_commission == 0.0

    def test_higher_commission_lower_return(self):
        prices = _make_prices()
        signals = _make_signals()

        bt_low = SimpleBacktester(commission_pct=0.0)
        r_low = bt_low.run(prices, signals)

        bt_high = SimpleBacktester(commission_pct=0.01)  # 1%
        r_high = bt_high.run(prices, signals)

        assert r_low.total_return_pct >= r_high.total_return_pct


# -- Trade Extraction Tests ------------------------------------------------


class TestTradeExtraction:
    def test_trades_are_trade_records(self):
        bt = SimpleBacktester()
        result = bt.run(_make_prices(), _make_signals())
        assert all(isinstance(t, TradeRecord) for t in result.trades)

    def test_all_long_single_trade(self):
        """Always long should produce a single open trade."""
        bt = SimpleBacktester()
        prices = _make_prices()
        signals = pd.Series(1.0, index=prices.index)
        result = bt.run(prices, signals)
        assert result.total_trades == 1

    def test_trade_fields(self):
        bt = SimpleBacktester()
        result = bt.run(_make_prices(), _make_signals())
        if result.trades:
            t = result.trades[0]
            assert t.direction == "long"
            assert isinstance(t.pnl, float)
            assert isinstance(t.holding_days, int)
            assert t.holding_days >= 1


# -- Baseline Comparison Tests ---------------------------------------------


class TestBaselines:
    def test_compare_baselines(self):
        bt = SimpleBacktester()
        bt.run(_make_prices(), _make_signals())
        baselines = bt.compare_baselines()

        assert "Buy-and-Hold" in baselines
        assert "Always Flat" in baselines
        assert "Random" in baselines
        assert all(isinstance(v, StrategyResult) for v in baselines.values())

    def test_buy_and_hold_exposure_100(self):
        bt = SimpleBacktester()
        bt.run(_make_prices(), _make_signals())
        baselines = bt.compare_baselines()
        # Buy-and-hold exposure: ~100% (first bar is NaN shifted, so ~99.6%)
        assert baselines["Buy-and-Hold"].exposure_pct > 95

    def test_always_flat_zero(self):
        bt = SimpleBacktester()
        bt.run(_make_prices(), _make_signals())
        baselines = bt.compare_baselines()
        assert abs(baselines["Always Flat"].total_return_pct) < 0.01
        assert baselines["Always Flat"].sharpe_ratio == 0.0

    def test_without_random(self):
        bt = SimpleBacktester()
        bt.run(_make_prices(), _make_signals())
        baselines = bt.compare_baselines(include_random=False)
        assert "Random" not in baselines

    def test_raises_before_run(self):
        bt = SimpleBacktester()
        with pytest.raises(RuntimeError, match="Call .run"):
            bt.compare_baselines()


# -- Summary Dict Tests ----------------------------------------------------


class TestSummary:
    def test_summary_dict(self):
        bt = SimpleBacktester()
        bt.run(_make_prices(), _make_signals())
        s = bt.summary()
        assert isinstance(s, dict)
        assert "total_return_pct" in s
        assert "sharpe_ratio" in s
        assert "hit_rate_pct" in s

    def test_summary_raises_before_run(self):
        bt = SimpleBacktester()
        with pytest.raises(RuntimeError):
            bt.summary()


# -- Convenience Function Tests --------------------------------------------


class TestConvenienceFunction:
    def test_backtest_signals(self):
        prices = _make_prices()
        signals = _make_signals()

        result, baselines = backtest_signals(
            prices, signals, compare_baselines=True,
        )
        assert isinstance(result, StrategyResult)
        assert len(baselines) > 0

    def test_backtest_signals_no_baseline(self):
        prices = _make_prices()
        signals = _make_signals()

        result, baselines = backtest_signals(
            prices, signals, compare_baselines=False,
        )
        assert isinstance(result, StrategyResult)
        assert len(baselines) == 0


# -- run_from_predictions Tests -------------------------------------------


class TestRunFromPredictions:
    def test_from_dataframe(self):
        prices = _make_prices()
        signals = _make_signals()
        df = pd.DataFrame({
            "Close": prices,
            "prediction": signals,
        })

        bt = SimpleBacktester()
        result = bt.run_from_predictions(df)
        assert isinstance(result, StrategyResult)
        assert result.total_trades > 0
