"""
backtesting.py
==============
Simulate a long/flat trading strategy driven by binary model predictions
and compare against naive baselines (buy-and-hold, always-flat, random).

Responsibilities
----------------
- Accept model signals (1 = go long, 0 = stay flat) aligned with a price series.
- Calculate daily strategy returns with configurable transaction costs.
- Report: cumulative return, annualised return, Sharpe ratio, max drawdown,
  hit rate (directional accuracy), profit factor, and Calmar ratio.
- Compare model performance against baselines side-by-side.
- Extract a log of individual round-trip trades.

Architecture
------------
::

    prices + signals
         |
    +----v-----------+       +------------------+
    | StrategyEngine |       | BaselineEngine   |
    | (long/flat)    |       | (buy-and-hold,   |
    |  - tx costs    |       |  random, flat)   |
    +----+-----------+       +--------+---------+
         |                            |
    +----v----------------------------v---------+
    |           BacktestComparison              |
    |  cumulative return, Sharpe, max DD,       |
    |  hit rate, trade log, baseline vs model   |
    +-------------------------------------------+

Quick-start
-----------
    from src.backtesting import SimpleBacktester

    bt = SimpleBacktester(initial_cash=100_000)
    result = bt.run(prices_series, signals_series)
    bt.print_report()

    # With baseline comparison
    bt.compare_baselines()

CLI
---
    python -m src.backtesting --demo
    python -m src.backtesting --prices data/raw/AAPL.csv --signals predictions.csv
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

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


# ===================================================================
#  DATA CLASSES
# ===================================================================


@dataclass
class TradeRecord:
    """Record for a single round-trip trade."""

    entry_date: str
    exit_date: str
    direction: str  # "long"
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    holding_days: int
    commission_paid: float


@dataclass
class StrategyResult:
    """Comprehensive backtest result for one strategy."""

    name: str
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    hit_rate_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    total_trades: int = 0
    total_commission: float = 0.0
    exposure_pct: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    trades: List[TradeRecord] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        """Flat dict of all scalar metrics."""
        return {
            "name": self.name,
            "total_return_pct": round(self.total_return_pct, 4),
            "annual_return_pct": round(self.annual_return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "hit_rate_pct": round(self.hit_rate_pct, 4),
            "profit_factor": round(self.profit_factor, 4),
            "win_rate_pct": round(self.win_rate_pct, 4),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "total_trades": self.total_trades,
            "total_commission": round(self.total_commission, 4),
            "exposure_pct": round(self.exposure_pct, 4),
        }


# ===================================================================
#  STRATEGY ENGINE
# ===================================================================


class _StrategyEngine:
    """Core computation engine for a single signal stream.

    Implements the vectorised long/flat strategy:
    - When signal == 1: fully invested (earn market return)
    - When signal == 0: flat / cash (earn nothing)
    - Transaction costs deducted on every position change.

    Parameters
    ----------
    initial_cash : float
        Starting portfolio value.
    commission_pct : float
        One-way commission as a fraction (applied on each entry and exit).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe/Sortino.
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission_pct: float = 0.001,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_pct = commission_pct
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        actual_directions: Optional[pd.Series] = None,
        name: str = "strategy",
    ) -> StrategyResult:
        """Execute the backtest for one signal stream.

        Parameters
        ----------
        prices : pd.Series
            Close prices with DatetimeIndex.
        signals : pd.Series
            Binary signals (1 = long, 0 = flat) aligned to prices.
        actual_directions : pd.Series | None
            Actual next-day direction (1 = up, 0 = down) for hit-rate.
            If None, computed from prices.
        name : str
            Strategy name for labelling.

        Returns
        -------
        StrategyResult
        """
        prices, signals = prices.align(signals, join="inner")
        signals = signals.fillna(0).astype(float)

        n = len(prices)
        if n < 2:
            logger.warning(f"Too few bars ({n}) for backtest")
            return StrategyResult(name=name)

        # -- Market returns --
        market_returns = prices.pct_change().fillna(0)

        # -- Position changes (for commission) --
        # signals.shift(1) is the position we held *during* today's bar
        positions = signals.shift(1).fillna(0)
        position_changes = positions.diff().abs().fillna(0)

        # Commission: deducted on each bar where position changes
        # Two-way (entry + exit) modelled as 2x one-way rate per flip
        commissions = position_changes * self.commission_pct * 2
        total_commission_pct = commissions.sum()
        total_commission = self.initial_cash * total_commission_pct

        # -- Strategy returns --
        strategy_returns = positions * market_returns - commissions

        # -- Equity curve --
        equity = self.initial_cash * (1 + strategy_returns).cumprod()

        # -- Drawdown --
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() * 100  # negative percent

        # -- Annualised return --
        total_return = (equity.iloc[-1] / self.initial_cash - 1) * 100
        n_years = n / 252
        annual_return = (
            ((1 + total_return / 100) ** (1 / max(n_years, 0.01)) - 1) * 100
        )

        # -- Sharpe ratio --
        rf_daily = self.risk_free_rate / 252
        excess = strategy_returns - rf_daily
        std_ret = excess.std()
        sharpe = (
            np.sqrt(252) * excess.mean() / std_ret
            if std_ret > 1e-10
            else 0.0
        )

        # -- Sortino ratio (downside deviation) --
        downside = excess[excess < 0]
        downside_std = downside.std() if len(downside) > 1 else 0.0
        sortino = (
            np.sqrt(252) * excess.mean() / downside_std
            if downside_std > 1e-10
            else 0.0
        )

        # -- Calmar ratio --
        calmar = (
            annual_return / abs(max_dd)
            if max_dd != 0
            else 0.0
        )

        # -- Hit rate (directional accuracy) --
        if actual_directions is not None:
            actual_dir = actual_directions.align(signals, join="inner")[0]
        else:
            # Compute from prices: 1 if tomorrow > today
            actual_dir = (prices.shift(-1) > prices).astype(float)
            actual_dir = actual_dir.iloc[:-1]  # drop last (unknown)

        # Align predictions with actual directions
        pred_aligned, actual_aligned = signals.align(actual_dir, join="inner")
        if len(pred_aligned) > 0:
            hit_rate = (pred_aligned == actual_aligned).mean() * 100
        else:
            hit_rate = 0.0

        # -- Exposure (% of time in the market) --
        exposure = positions.mean() * 100

        # -- Trade extraction --
        trades = self._extract_trades(prices, signals)

        # -- Win rate, profit factor, avg win/loss --
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0

        total_win_pnl = sum(t.pnl for t in wins) if wins else 0.0
        total_loss_pnl = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = (
            total_win_pnl / total_loss_pnl
            if total_loss_pnl > 0
            else float("inf") if total_win_pnl > 0 else 0.0
        )

        avg_win = (
            np.mean([t.return_pct for t in wins]) if wins else 0.0
        )
        avg_loss = (
            np.mean([t.return_pct for t in losses]) if losses else 0.0
        )

        return StrategyResult(
            name=name,
            total_return_pct=total_return,
            annual_return_pct=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            hit_rate_pct=hit_rate,
            profit_factor=profit_factor,
            win_rate_pct=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            total_trades=len(trades),
            total_commission=total_commission,
            exposure_pct=exposure,
            equity_curve=equity,
            daily_returns=strategy_returns,
            drawdown_series=drawdown,
            trades=trades,
        )

    @staticmethod
    def _extract_trades(
        prices: pd.Series, signals: pd.Series
    ) -> List[TradeRecord]:
        """Parse signal flips into round-trip trade records."""
        trades: List[TradeRecord] = []
        in_trade = False
        entry_date = entry_price = None

        for date, sig in signals.items():
            price = prices.loc[date]
            if sig == 1 and not in_trade:
                in_trade = True
                entry_date = date
                entry_price = price
            elif sig == 0 and in_trade:
                in_trade = False
                pnl = price - entry_price
                ret = pnl / entry_price * 100
                days = 1
                if hasattr(date, "date") and hasattr(entry_date, "date"):
                    days = max((date - entry_date).days, 1)
                trades.append(
                    TradeRecord(
                        entry_date=str(entry_date),
                        exit_date=str(date),
                        direction="long",
                        entry_price=round(entry_price, 4),
                        exit_price=round(price, 4),
                        shares=1,
                        pnl=round(pnl, 4),
                        return_pct=round(ret, 4),
                        holding_days=days,
                        commission_paid=0.0,  # tracked globally
                    )
                )

        # Close any open position at the end
        if in_trade:
            last_date = prices.index[-1]
            last_price = prices.iloc[-1]
            pnl = last_price - entry_price
            ret = pnl / entry_price * 100
            days = 1
            if hasattr(last_date, "date") and hasattr(entry_date, "date"):
                days = max((last_date - entry_date).days, 1)
            trades.append(
                TradeRecord(
                    entry_date=str(entry_date),
                    exit_date=str(last_date) + " (open)",
                    direction="long",
                    entry_price=round(entry_price, 4),
                    exit_price=round(last_price, 4),
                    shares=1,
                    pnl=round(pnl, 4),
                    return_pct=round(ret, 4),
                    holding_days=days,
                    commission_paid=0.0,
                )
            )

        return trades


# ===================================================================
#  SIMPLE BACKTESTER (public API)
# ===================================================================


class SimpleBacktester:
    """Long/flat strategy backtester with baseline comparison.

    The model signals drive a simple rule:
    - Signal = 1: go long (fully invested in the stock)
    - Signal = 0: stay flat (cash, earn nothing)

    Parameters
    ----------
    initial_cash : float
        Starting portfolio value.
    commission_pct : float
        One-way transaction cost as a fraction (e.g. 0.001 = 10 bps).
        Applied as 2x on each position flip (entry + exit).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe / Sortino calculation.

    Examples
    --------
    >>> bt = SimpleBacktester(initial_cash=100_000, commission_pct=0.001)
    >>> result = bt.run(prices, model_signals)
    >>> bt.print_report()
    >>> comparison = bt.compare_baselines()
    """

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission_pct: float = 0.001,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission_pct = commission_pct
        self.risk_free_rate = risk_free_rate

        self._engine = _StrategyEngine(
            initial_cash=initial_cash,
            commission_pct=commission_pct,
            risk_free_rate=risk_free_rate,
        )

        # State
        self.result: Optional[StrategyResult] = None
        self.baselines: Dict[str, StrategyResult] = {}
        self._prices: Optional[pd.Series] = None
        self._signals: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        actual_directions: Optional[pd.Series] = None,
    ) -> StrategyResult:
        """Execute the backtest.

        Parameters
        ----------
        prices : pd.Series
            Close prices with DatetimeIndex.
        signals : pd.Series
            Binary predictions (1 = long, 0 = flat).
        actual_directions : pd.Series | None
            Actual next-day direction (1 = up, 0 = down). If None,
            computed automatically from prices.

        Returns
        -------
        StrategyResult
        """
        self._prices = prices
        self._signals = signals

        logger.info(f"Running backtest on {len(prices)} bars ...")
        self.result = self._engine.run(
            prices, signals,
            actual_directions=actual_directions,
            name="Model Strategy",
        )
        logger.info(
            f"Backtest complete -> "
            f"return={self.result.total_return_pct:+.2f}%, "
            f"sharpe={self.result.sharpe_ratio:.2f}, "
            f"max_dd={self.result.max_drawdown_pct:.2f}%, "
            f"hit_rate={self.result.hit_rate_pct:.1f}%"
        )
        return self.result

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def compare_baselines(
        self,
        include_random: bool = True,
        random_seed: int = 42,
    ) -> Dict[str, StrategyResult]:
        """Run baseline strategies and compare against the model.

        Baselines
        ---------
        - **Buy-and-Hold**: always long (signal = 1 every day)
        - **Always Flat**: never invest (signal = 0 every day)
        - **Random**: random 50/50 signals (optional)

        Returns
        -------
        dict[str, StrategyResult]
            Keyed by strategy name.
        """
        if self._prices is None or self._signals is None:
            raise RuntimeError("Call .run() before .compare_baselines()")

        prices = self._prices

        # Buy-and-Hold
        bah_signals = pd.Series(1.0, index=prices.index, name="signal")
        self.baselines["Buy-and-Hold"] = self._engine.run(
            prices, bah_signals, name="Buy-and-Hold"
        )
        logger.info(
            f"  Buy-and-Hold: return={self.baselines['Buy-and-Hold'].total_return_pct:+.2f}%"
        )

        # Always Flat
        flat_signals = pd.Series(0.0, index=prices.index, name="signal")
        self.baselines["Always Flat"] = self._engine.run(
            prices, flat_signals, name="Always Flat"
        )

        # Random baseline
        if include_random:
            np.random.seed(random_seed)
            rand_signals = pd.Series(
                np.random.choice([0.0, 1.0], size=len(prices)),
                index=prices.index,
                name="signal",
            )
            self.baselines["Random"] = self._engine.run(
                prices, rand_signals, name="Random"
            )
            logger.info(
                f"  Random: return={self.baselines['Random'].total_return_pct:+.2f}%"
            )

        return self.baselines

    # ------------------------------------------------------------------
    # Summary dict
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return model result metrics as a plain dict."""
        if self.result is None:
            raise RuntimeError("Call .run() before .summary()")
        return self.result.summary_dict()

    # ------------------------------------------------------------------
    # Pretty-print report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a comprehensive backtest report."""
        if self.result is None:
            raise RuntimeError("Call .run() before .print_report()")

        r = self.result

        print(f"\n{'=' * 78}")
        print(f"  BACKTEST REPORT -- {r.name}")
        print(f"{'=' * 78}")

        print(f"\n  Configuration:")
        print(f"    Initial cash     : ${self.initial_cash:,.0f}")
        print(f"    Commission (1-way): {self.commission_pct*100:.2f}%")
        print(f"    Risk-free rate   : {self.risk_free_rate*100:.1f}%")
        print(f"    Bars             : {len(r.equity_curve)}")

        if isinstance(r.equity_curve.index, pd.DatetimeIndex) and len(r.equity_curve) > 0:
            print(f"    Period           : {r.equity_curve.index[0].date()} -> "
                  f"{r.equity_curve.index[-1].date()}")

        print(f"\n{'-' * 78}")
        print(f"  Returns:")
        print(f"{'-' * 78}")
        print(f"    Total return     : {r.total_return_pct:+.2f}%")
        print(f"    Annual return    : {r.annual_return_pct:+.2f}%")
        print(f"    Final equity     : ${r.equity_curve.iloc[-1]:,.2f}"
              if len(r.equity_curve) > 0 else "")

        print(f"\n{'-' * 78}")
        print(f"  Risk Metrics:")
        print(f"{'-' * 78}")
        print(f"    Sharpe ratio     : {r.sharpe_ratio:.4f}")
        print(f"    Sortino ratio    : {r.sortino_ratio:.4f}")
        print(f"    Calmar ratio     : {r.calmar_ratio:.4f}")
        print(f"    Max drawdown     : {r.max_drawdown_pct:.2f}%")

        print(f"\n{'-' * 78}")
        print(f"  Signal Quality:")
        print(f"{'-' * 78}")
        print(f"    Hit rate         : {r.hit_rate_pct:.2f}%  "
              f"(directional accuracy)")
        print(f"    Exposure         : {r.exposure_pct:.1f}%  "
              f"(time in market)")

        print(f"\n{'-' * 78}")
        print(f"  Trade Statistics:")
        print(f"{'-' * 78}")
        print(f"    Total trades     : {r.total_trades}")
        print(f"    Win rate         : {r.win_rate_pct:.1f}%")
        print(f"    Avg win          : {r.avg_win_pct:+.2f}%")
        print(f"    Avg loss         : {r.avg_loss_pct:+.2f}%")
        print(f"    Profit factor    : {r.profit_factor:.2f}")
        print(f"    Total commission : ${r.total_commission:,.2f}")

        # Baseline comparison table (if available)
        if self.baselines:
            print(f"\n{'-' * 78}")
            print(f"  Baseline Comparison:")
            print(f"{'-' * 78}")

            all_results = {"Model": r}
            all_results.update(self.baselines)

            header = (
                f"  {'Strategy':<18} {'Return':>9} {'Annual':>9} "
                f"{'Sharpe':>8} {'MaxDD':>9} {'HitRate':>8} {'Trades':>7}"
            )
            print(header)
            print(f"  {'-' * 68}")

            for name, res in all_results.items():
                print(
                    f"  {name:<18} "
                    f"{res.total_return_pct:>+8.2f}% "
                    f"{res.annual_return_pct:>+8.2f}% "
                    f"{res.sharpe_ratio:>8.3f} "
                    f"{res.max_drawdown_pct:>8.2f}% "
                    f"{res.hit_rate_pct:>7.1f}% "
                    f"{res.total_trades:>7d}"
                )

            # Performance vs buy-and-hold
            if "Buy-and-Hold" in self.baselines:
                bah = self.baselines["Buy-and-Hold"]
                excess = r.total_return_pct - bah.total_return_pct
                label = "OUTPERFORMED" if excess > 0 else "UNDERPERFORMED"
                print(f"\n  Model {label} buy-and-hold by {excess:+.2f}%")

        # Recent trades
        if r.trades:
            n_show = min(10, len(r.trades))
            print(f"\n{'-' * 78}")
            print(f"  Recent Trades (last {n_show}):")
            print(f"{'-' * 78}")
            print(
                f"  {'Entry':<12} {'Exit':<12} {'Entry$':>9} "
                f"{'Exit$':>9} {'P&L':>9} {'Ret%':>8} {'Days':>5}"
            )
            print(f"  {'-' * 64}")
            for t in r.trades[-n_show:]:
                entry_short = t.entry_date[:10] if len(t.entry_date) >= 10 else t.entry_date
                exit_short = t.exit_date[:10] if len(t.exit_date) >= 10 else t.exit_date
                print(
                    f"  {entry_short:<12} {exit_short:<12} "
                    f"{t.entry_price:>9.2f} {t.exit_price:>9.2f} "
                    f"{t.pnl:>+9.2f} {t.return_pct:>+7.2f}% "
                    f"{t.holding_days:>5d}"
                )

        print(f"\n{'=' * 78}\n")

    # ------------------------------------------------------------------
    # Convenience: run from model predictions DataFrame
    # ------------------------------------------------------------------

    def run_from_predictions(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        signal_col: str = "prediction",
        target_col: Optional[str] = "target",
    ) -> StrategyResult:
        """Run backtest from a DataFrame with price, signal, and target cols.

        This is useful when you have a DataFrame from model evaluation
        that already contains predictions aligned with prices and targets.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex containing the required columns.
        price_col : str
            Column for close prices.
        signal_col : str
            Column for binary model predictions (1 = long).
        target_col : str | None
            Column for actual direction. If None, computed from prices.

        Returns
        -------
        StrategyResult
        """
        prices = df[price_col]
        signals = df[signal_col]
        actual = df[target_col] if target_col and target_col in df.columns else None
        return self.run(prices, signals, actual_directions=actual)


# ===================================================================
#  CONVENIENCE FUNCTIONS
# ===================================================================


def backtest_signals(
    prices: pd.Series,
    signals: pd.Series,
    initial_cash: float = 100_000,
    commission_pct: float = 0.001,
    compare_baselines: bool = True,
) -> Tuple[StrategyResult, Dict[str, StrategyResult]]:
    """One-call function to run a full backtest with baselines.

    Parameters
    ----------
    prices : pd.Series
        Close prices.
    signals : pd.Series
        Binary model predictions (1 = long, 0 = flat).
    initial_cash : float
        Starting capital.
    commission_pct : float
        Transaction cost (one-way, as fraction).
    compare_baselines : bool
        Whether to run baseline comparisons.

    Returns
    -------
    tuple[StrategyResult, dict]
        (model_result, baselines_dict)
    """
    bt = SimpleBacktester(
        initial_cash=initial_cash,
        commission_pct=commission_pct,
    )
    result = bt.run(prices, signals)

    baselines = {}
    if compare_baselines:
        baselines = bt.compare_baselines()

    bt.print_report()
    return result, baselines


# ===================================================================
#  CLI
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest a long/flat strategy with baseline comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.backtesting --demo\n"
            "  python -m src.backtesting --demo --commission 0.002\n"
            "  python -m src.backtesting --prices data/raw/AAPL.csv "
            "--signals predictions.csv\n"
        ),
    )
    parser.add_argument("--prices", type=str, default=None,
                        help="CSV with Close prices (Date index)")
    parser.add_argument("--signals", type=str, default=None,
                        help="CSV with binary signals (Date index)")
    parser.add_argument("--cash", type=float, default=100_000,
                        help="Initial cash (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="One-way commission fraction (default: 0.001)")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data")
    parser.add_argument("--demo-bars", type=int, default=504,
                        help="Number of bars for demo (default: 504 = ~2yrs)")
    args = parser.parse_args()

    if args.demo:
        # -- Synthetic demo --
        np.random.seed(42)
        n = args.demo_bars
        dates = pd.date_range("2023-01-02", periods=n, freq="B")

        # Simulate a trending stock with noise
        daily_ret = np.random.normal(0.0004, 0.015, n)  # slight upward drift
        cum_ret = (1 + pd.Series(daily_ret)).cumprod()
        prices = pd.Series(100.0 * cum_ret.values, index=dates, name="Close")

        # Simulate a model with ~55% directional accuracy (better than random)
        actual_direction = (prices.shift(-1) > prices).astype(float).fillna(0)
        model_accuracy = 0.55
        correct_mask = np.random.random(n) < model_accuracy
        model_signals = actual_direction.copy()
        # Flip signals where model is wrong
        model_signals[~correct_mask] = 1 - model_signals[~correct_mask]
        model_signals = model_signals.astype(float)

        print(f"\n  Demo: {n} bars, model accuracy ~{model_accuracy*100:.0f}%, "
              f"commission={args.commission*100:.2f}%")

        result, baselines = backtest_signals(
            prices,
            model_signals,
            initial_cash=args.cash,
            commission_pct=args.commission,
            compare_baselines=True,
        )

    elif args.prices and args.signals:
        prices_df = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        signals_df = pd.read_csv(args.signals, index_col=0, parse_dates=True)

        # Assume first column is the price / signal
        prices = prices_df.iloc[:, 0]
        signals = signals_df.iloc[:, 0]

        result, baselines = backtest_signals(
            prices,
            signals,
            initial_cash=args.cash,
            commission_pct=args.commission,
            compare_baselines=True,
        )

    else:
        parser.print_help()
