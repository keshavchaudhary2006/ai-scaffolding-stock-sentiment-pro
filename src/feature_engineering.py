"""
feature_engineering.py
======================
Merge stock price data with daily news sentiment, compute a rich set of
technical indicators and statistical features, and output a clean modelling
dataset with binary target labels for next-day stock movement.

Feature categories
------------------
1. **Price-based returns** — simple, log, multi-horizon, lagged
2. **Moving averages** — SMA / EMA at multiple windows, price-to-MA ratios
3. **RSI** — 14-day relative strength index
4. **MACD** — line, signal, histogram
5. **Bollinger Bands** — upper/lower/width, %B position
6. **Volatility** — rolling standard deviation, ATR, Garman-Klass, Parkinson
7. **Volume** — OBV, volume Z-score, volume-return correlation
8. **Rolling statistics** — rolling skew, kurtosis, min, max
9. **Lagged returns** — configurable lag depths
10. **Sentiment features** — merged daily sentiment from ``sentiment_analyzer``
    + rolling sentiment smoothing

Output
------
A single ``pd.DataFrame`` with:
- All features as numeric columns
- A ``target`` column: **1** if next-day Close > today's Close, else **0**
- A ``target_return`` column: next-day simple return (for regression tasks)
- DatetimeIndex sorted chronologically
- No NaN values

Quick-start
-----------
    from src.feature_engineering import build_feature_matrix, FeatureEngineer

    # Function interface
    df = build_feature_matrix(prices_df, sentiment_daily_df,
                              output_path="data/processed/features.csv")

    # Class interface
    eng = FeatureEngineer()
    df  = eng.build(prices_df, sentiment_daily_df)

CLI
---
    python -m src.feature_engineering --prices data/raw/AAPL.csv \\
                                      --sentiment data/processed/AAPL_daily_sent.csv \\
                                      --output data/processed/AAPL_features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

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
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> — "
        "<level>{message}</level>"
    ),
    level="INFO",
)


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEER CLASS
# ═══════════════════════════════════════════════════════════════════════


class FeatureEngineer:
    """Configurable feature-engineering pipeline.

    Parameters
    ----------
    ma_windows : list[int]
        Windows for Simple & Exponential Moving Averages.
    rsi_period : int
        Look-back for RSI.
    macd_fast, macd_slow, macd_signal : int
        MACD EMA spans.
    bb_period, bb_std : int, float
        Bollinger Bands look-back and std multiplier.
    atr_period : int
        ATR look-back.
    volatility_windows : list[int]
        Windows for rolling volatility / stats.
    lag_depths : list[int]
        Number of lagged-return columns (e.g. [1,2,3,5,10]).
    sentiment_rolling_windows : list[int]
        Extra rolling smooths applied to daily sentiment after merge.
    forecast_horizon : int
        How many bars ahead the target label looks.
    """

    def __init__(
        self,
        ma_windows: Optional[List[int]] = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        volatility_windows: Optional[List[int]] = None,
        lag_depths: Optional[List[int]] = None,
        sentiment_rolling_windows: Optional[List[int]] = None,
        forecast_horizon: int = 1,
    ) -> None:
        self.ma_windows = ma_windows or [5, 10, 20, 50]
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volatility_windows = volatility_windows or [5, 10, 20]
        self.lag_depths = lag_depths or [1, 2, 3, 5, 10]
        self.sentiment_rolling_windows = sentiment_rolling_windows or [3, 5, 7]
        self.forecast_horizon = forecast_horizon

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN BUILD METHOD
    # ═══════════════════════════════════════════════════════════════════

    def build(
        self,
        prices_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        output_path: Optional[Union[str, Path]] = None,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """Full pipeline: indicators → merge sentiment → target → clean.

        Parameters
        ----------
        prices_df : pd.DataFrame
            OHLCV data with DatetimeIndex.  Expected columns:
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
        sentiment_df : pd.DataFrame | None
            Daily sentiment features from ``sentiment_analyzer.aggregate_daily()``.
            Columns like ``sent_mean``, ``sent_std``, etc.  DatetimeIndex.
        output_path : str | Path | None
            Save final CSV if provided.
        drop_na : bool
            Drop rows with any NaN (warm-up rows from indicators).

        Returns
        -------
        pd.DataFrame
            Modelling-ready feature matrix with ``target`` column.
        """
        prices_df = self._ensure_datetime_index(prices_df.copy())
        logger.info(
            f"Building features: {len(prices_df)} price rows, "
            f"sentiment={'yes' if sentiment_df is not None else 'no'}"
        )

        # 1 — Returns
        df = self._add_returns(prices_df)

        # 2 — Lagged returns
        df = self._add_lagged_returns(df)

        # 3 — Moving averages + price-to-MA ratios
        df = self._add_moving_averages(df)

        # 4 — RSI
        df = self._add_rsi(df)

        # 5 — MACD
        df = self._add_macd(df)

        # 6 — Bollinger Bands
        df = self._add_bollinger_bands(df)

        # 7 — Volatility (ATR, rolling std, Garman-Klass, Parkinson)
        df = self._add_volatility(df)

        # 8 — Volume features
        df = self._add_volume_features(df)

        # 9 — Rolling statistics (skew, kurtosis, min, max)
        df = self._add_rolling_stats(df)

        # 10 — Day-of-week / month features
        df = self._add_calendar_features(df)

        # 11 — Merge sentiment
        if sentiment_df is not None and not sentiment_df.empty:
            df = self._merge_sentiment(df, sentiment_df)

        # 12 — Target labels
        df = self._add_target(df)

        # 13 — Clean
        if drop_na:
            before = len(df)
            df.dropna(inplace=True)
            logger.info(f"Dropped {before - len(df)} warm-up rows → {len(df)} remain")

        # 14 — Summary
        n_features = len([c for c in df.columns if c not in ("target", "target_return")])
        logger.info(
            f"Feature matrix ready: {df.shape[0]} rows × {n_features} features "
            f"+ target  (horizon={self.forecast_horizon})"
        )

        if output_path:
            self._save(df, output_path)

        return df

    # ═══════════════════════════════════════════════════════════════════
    #  1. RETURNS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Simple, log, and multi-horizon returns."""
        c = df["Close"]

        df["return_1d"] = c.pct_change(1)
        df["return_2d"] = c.pct_change(2)
        df["return_5d"] = c.pct_change(5)
        df["return_10d"] = c.pct_change(10)
        df["return_20d"] = c.pct_change(20)

        df["log_return_1d"] = np.log(c / c.shift(1))
        df["log_return_5d"] = np.log(c / c.shift(5))

        # Intraday return (open → close)
        df["intraday_return"] = (c - df["Open"]) / df["Open"]

        # Overnight gap (previous close → today open)
        df["overnight_gap"] = (df["Open"] - c.shift(1)) / c.shift(1)

        logger.debug("Added return features")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  2. LAGGED RETURNS
    # ═══════════════════════════════════════════════════════════════════

    def _add_lagged_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag the 1-day return by multiple depths."""
        for lag in self.lag_depths:
            df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)
        logger.debug(f"Added lagged returns: lags={self.lag_depths}")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  3. MOVING AVERAGES
    # ═══════════════════════════════════════════════════════════════════

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA, EMA, and price-to-MA ratios for each window."""
        c = df["Close"]
        for w in self.ma_windows:
            sma = c.rolling(window=w).mean()
            ema = c.ewm(span=w, adjust=False).mean()

            df[f"sma_{w}"] = sma
            df[f"ema_{w}"] = ema

            # Price relative to its moving average (mean-reversion signal)
            df[f"price_to_sma_{w}"] = c / sma
            df[f"price_to_ema_{w}"] = c / ema

        # MA crossovers (short vs long)
        if len(self.ma_windows) >= 2:
            short_w, long_w = sorted(self.ma_windows)[:2]
            df["ma_crossover"] = df[f"sma_{short_w}"] - df[f"sma_{long_w}"]

        logger.debug(f"Added moving averages: windows={self.ma_windows}")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  4. RSI
    # ═══════════════════════════════════════════════════════════════════

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wilder's RSI with configurable period."""
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))

        # Wilder smoothing (exponential with alpha=1/period)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI zones (useful for tree models)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

        logger.debug(f"Added RSI (period={self.rsi_period})")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  5. MACD
    # ═══════════════════════════════════════════════════════════════════

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD line, signal line, and histogram."""
        c = df["Close"]
        ema_fast = c.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = c.ewm(span=self.macd_slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Signal crossover direction
        df["macd_cross"] = np.sign(df["macd_hist"]).diff()

        logger.debug(
            f"Added MACD (fast={self.macd_fast}, slow={self.macd_slow}, "
            f"signal={self.macd_signal})"
        )
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  6. BOLLINGER BANDS
    # ═══════════════════════════════════════════════════════════════════

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands with %B and bandwidth."""
        c = df["Close"]
        sma = c.rolling(window=self.bb_period).mean()
        std = c.rolling(window=self.bb_period).std()

        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        df["bb_upper"] = upper
        df["bb_middle"] = sma
        df["bb_lower"] = lower

        # Bandwidth: normalised range of the bands
        df["bb_width"] = (upper - lower) / sma

        # %B: where price sits within the bands (0 = lower, 1 = upper)
        band_range = upper - lower
        df["bb_pct_b"] = (c - lower) / band_range.replace(0, np.nan)

        logger.debug(f"Added Bollinger Bands (period={self.bb_period}, std={self.bb_std})")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  7. VOLATILITY
    # ═══════════════════════════════════════════════════════════════════

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiple volatility estimators."""

        # --- ATR ---
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=self.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["Close"]  # normalised ATR

        # --- Rolling standard deviation of returns ---
        for w in self.volatility_windows:
            df[f"volatility_{w}d"] = df["return_1d"].rolling(window=w).std()

        # --- Garman-Klass volatility estimator ---
        log_hl = np.log(df["High"] / df["Low"])
        log_co = np.log(df["Close"] / df["Open"])
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        df["gk_volatility"] = gk.rolling(window=self.volatility_windows[0]).mean()

        # --- Parkinson volatility ---
        df["parkinson_vol"] = (
            (1.0 / (4.0 * np.log(2)))
            * (np.log(df["High"] / df["Low"]) ** 2)
        ).rolling(window=self.volatility_windows[0]).mean().apply(np.sqrt)

        logger.debug(f"Added volatility features (ATR period={self.atr_period})")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  8. VOLUME FEATURES
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """OBV, volume Z-score, volume SMA ratio."""
        vol = df["Volume"].astype(float)

        # On-Balance Volume
        obv = (np.sign(df["Close"].diff()) * vol).fillna(0).cumsum()
        df["obv"] = obv

        # Volume Z-score (how unusual is today's volume?)
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        df["volume_zscore"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

        # Volume ratio (today vs 20-day average)
        df["volume_ratio"] = vol / vol_mean.replace(0, np.nan)

        # Volume-price trend (VPT)
        df["vpt"] = (vol * df["Close"].pct_change()).fillna(0).cumsum()

        logger.debug("Added volume features (OBV, Z-score, VPT)")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  9. ROLLING STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling skew, kurtosis, min, max of daily returns."""
        ret = df["return_1d"]

        for w in self.volatility_windows:
            df[f"roll_skew_{w}d"] = ret.rolling(window=w).skew()
            df[f"roll_kurt_{w}d"] = ret.rolling(window=w).kurt()
            df[f"roll_min_{w}d"] = ret.rolling(window=w).min()
            df[f"roll_max_{w}d"] = ret.rolling(window=w).max()
            df[f"roll_range_{w}d"] = df[f"roll_max_{w}d"] - df[f"roll_min_{w}d"]

        logger.debug(f"Added rolling stats for windows={self.volatility_windows}")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  10. CALENDAR FEATURES
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """Day-of-week and month as numeric features."""
        df["day_of_week"] = df.index.dayofweek      # Mon=0 … Fri=4
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)

        logger.debug("Added calendar features")
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  11. SENTIMENT MERGE
    # ═══════════════════════════════════════════════════════════════════

    def _merge_sentiment(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join daily sentiment features and add rolling smooths.

        The sentiment DataFrame is expected to come from
        ``sentiment_analyzer.aggregate_daily_sentiment()`` with columns
        like ``sent_mean``, ``sent_std``, ``sent_positive_pct``, etc.
        and a DatetimeIndex named ``date``.
        """
        sentiment_df = self._ensure_datetime_index(sentiment_df.copy())

        # Normalise both indexes to date-only for clean merge
        df_date = df.index.normalize()
        sent_date = sentiment_df.index.normalize()

        # Temporarily set date-only index for merge
        df_temp = df.copy()
        df_temp["_merge_date"] = df_date
        sentiment_df["_merge_date"] = sent_date

        merged = df_temp.merge(
            sentiment_df,
            on="_merge_date",
            how="left",
            suffixes=("", "_sent"),
        )
        merged.index = df.index  # restore original index
        merged.drop(columns=["_merge_date"], inplace=True, errors="ignore")

        # Forward-fill sentiment on days with no news
        sent_cols = [c for c in merged.columns if c.startswith("sent_")]
        merged[sent_cols] = merged[sent_cols].ffill()

        # Rolling smooths on the primary sentiment score
        if "sent_mean" in merged.columns:
            for w in self.sentiment_rolling_windows:
                merged[f"sent_smooth_{w}d"] = (
                    merged["sent_mean"].rolling(window=w, min_periods=1).mean()
                )
                merged[f"sent_momentum_{w}d"] = merged["sent_mean"].diff(w)

        n_sent = len(sent_cols) + 2 * len(self.sentiment_rolling_windows)
        logger.info(
            f"Merged {len(sentiment_df)} sentiment days → "
            f"{n_sent} sentiment features"
        )
        return merged

    # ═══════════════════════════════════════════════════════════════════
    #  12. TARGET LABELS
    # ═══════════════════════════════════════════════════════════════════

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target and continuous target-return columns.

        - ``target``: 1 if Close rises over *forecast_horizon* bars, else 0
        - ``target_return``: simple return over *forecast_horizon* bars
        """
        future_close = df["Close"].shift(-self.forecast_horizon)
        df["target_return"] = (future_close - df["Close"]) / df["Close"]
        df["target"] = (future_close > df["Close"]).astype(int)

        logger.debug(
            f"Target: horizon={self.forecast_horizon}, "
            f"class balance = {df['target'].mean():.2%} positive"
        )
        return df

    # ═══════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce index (or 'Date'/'date' column) to DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("Date", "date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
                    df.set_index(col, inplace=True)
                    break
            else:
                df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _save(df: pd.DataFrame, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info(f"Feature matrix saved → {path}")
        return path

    # ═══════════════════════════════════════════════════════════════════
    #  FEATURE CATALOG
    # ═══════════════════════════════════════════════════════════════════

    def describe_features(self) -> pd.DataFrame:
        """Return a DataFrame cataloguing every feature group and count."""
        groups = {
            "Returns (simple, log, intraday, overnight)": 9,
            f"Lagged returns (depths={self.lag_depths})": len(self.lag_depths),
            f"Moving averages SMA/EMA (windows={self.ma_windows})": len(self.ma_windows) * 4 + 1,
            "RSI + zone indicators": 3,
            "MACD (line, signal, hist, cross)": 4,
            "Bollinger Bands (upper, mid, lower, width, %B)": 5,
            f"Volatility (ATR, rolling std, GK, Parkinson)": 2 + len(self.volatility_windows) + 2,
            "Volume (OBV, Z-score, ratio, VPT)": 4,
            f"Rolling stats (skew, kurt, min, max, range)": len(self.volatility_windows) * 5,
            "Calendar (dow, month, quarter, month edges)": 5,
            "Sentiment (if provided)": "variable",
            "Target columns": 2,
        }
        return pd.DataFrame(
            [{"group": k, "count": v} for k, v in groups.items()]
        )


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION INTERFACE
# ═══════════════════════════════════════════════════════════════════════


def build_feature_matrix(
    prices_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
    forecast_horizon: int = 1,
    ma_windows: Optional[List[int]] = None,
    lag_depths: Optional[List[int]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """One-call function to produce a modelling-ready feature matrix.

    Parameters
    ----------
    prices_df : pd.DataFrame
        OHLCV data (columns: Open, High, Low, Close, Volume).
    sentiment_df : pd.DataFrame | None
        Daily sentiment features (output of ``aggregate_daily_sentiment``).
    output_path : str | Path | None
        Save to CSV if provided.
    forecast_horizon : int
        Target look-ahead (1 = next-day prediction).
    ma_windows : list[int] | None
        Override moving-average windows.
    lag_depths : list[int] | None
        Override lagged-return depths.
    drop_na : bool
        Drop warm-up NaN rows.

    Returns
    -------
    pd.DataFrame
        Complete feature matrix with ``target`` column.

    Examples
    --------
    >>> df = build_feature_matrix(prices, sentiment_daily,
    ...                           output_path="data/processed/features.csv")
    >>> print(df.shape)
    (220, 87)
    """
    eng = FeatureEngineer(
        forecast_horizon=forecast_horizon,
        ma_windows=ma_windows,
        lag_depths=lag_depths,
    )
    return eng.build(
        prices_df,
        sentiment_df=sentiment_df,
        output_path=output_path,
        drop_na=drop_na,
    )


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a modelling-ready feature matrix from price + sentiment data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.feature_engineering --prices data/raw/AAPL.csv\n"
            "  python -m src.feature_engineering --prices data/raw/AAPL.csv \\\n"
            "      --sentiment data/processed/AAPL_daily_sent.csv \\\n"
            "      --output data/processed/AAPL_features.csv\n"
            "  python -m src.feature_engineering --demo\n"
        ),
    )
    parser.add_argument("--prices", type=str, help="Input OHLCV CSV (Date index)")
    parser.add_argument("--sentiment", type=str, default=None, help="Daily sentiment CSV")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--horizon", type=int, default=1, help="Target forecast horizon")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    if args.demo:
        # ── Demo with synthetic data ──
        np.random.seed(42)
        n = 300
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
        # Ensure High >= Close >= Low
        prices["High"] = prices[["Open", "High", "Close"]].max(axis=1)
        prices["Low"] = prices[["Open", "Low", "Close"]].min(axis=1)

        # Fake daily sentiment
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
        result = eng.build(prices, sent)

        print(f"\n{'═' * 70}")
        print(f"  Feature Matrix — Demo")
        print(f"{'═' * 70}")
        print(f"\n  Shape  : {result.shape}")
        print(f"  Target : {result['target'].value_counts().to_dict()}")
        print(f"\n  Feature groups:")
        print(eng.describe_features().to_string(index=False))
        print(f"\n  Columns ({len(result.columns)}):")
        for i, col in enumerate(result.columns):
            print(f"    {i+1:3d}. {col}")
        print(f"\n  First 5 rows (selected columns):")
        preview = result[["return_1d", "rsi", "macd", "bb_pct_b", "atr_pct",
                          "volume_zscore", "sent_mean", "target"]].head()
        print(preview.to_string())

    elif args.prices:
        prices_df = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        sentiment_df = None
        if args.sentiment:
            sentiment_df = pd.read_csv(args.sentiment, index_col=0, parse_dates=True)

        out = args.output or args.prices.replace(".csv", "_features.csv")
        result = build_feature_matrix(
            prices_df,
            sentiment_df=sentiment_df,
            output_path=out,
            forecast_horizon=args.horizon,
        )
        print(f"\n  Feature matrix: {result.shape}")
        print(f"  Saved → {out}")

    else:
        parser.print_help()
