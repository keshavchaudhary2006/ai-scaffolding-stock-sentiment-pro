"""
data_fetcher.py
===============
Download, clean, and enrich historical stock price data using **yfinance**.

Responsibilities
----------------
- Download OHLCV price data for any ticker and date range.
- Clean missing / NaN values with configurable strategies (drop, ffill, bfill, interpolate).
- Compute daily simple and logarithmic returns.
- Persist cleaned results to CSV.
- Pull news headlines from NewsAPI for sentiment pipelines.

Quick-start (function interface)
--------------------------------
    from src.data_fetcher import fetch_stock_data

    df = fetch_stock_data(
        ticker="AAPL",
        start="2023-01-01",
        end="2024-01-01",
        output_path="data/processed/AAPL_clean.csv",
    )

Class interface
---------------
    from src.data_fetcher import StockDataFetcher

    fetcher = StockDataFetcher(tickers=["AAPL", "GOOGL"])
    prices = fetcher.fetch(start="2023-01-01", end="2024-01-01")

CLI
---
    python -m src.data_fetcher --ticker AAPL --start 2023-01-01 --end 2024-01-01
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

# Remove the default handler and add a cleaner one
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
    level="INFO",
)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config() -> dict:
    """Load the project-wide YAML configuration.

    Returns
    -------
    dict
        Parsed contents of ``config/config.yaml``.

    Raises
    ------
    FileNotFoundError
        If the configuration file is missing.
    """
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file not found at {CONFIG_PATH}; using defaults")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ═══════════════════════════════════════════════════════════════════════
#  REUSABLE FUNCTION INTERFACE
# ═══════════════════════════════════════════════════════════════════════


def fetch_stock_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    interval: str = "1d",
    clean_method: Literal["drop", "ffill", "bfill", "interpolate"] = "ffill",
    compute_returns: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Download, clean, and enrich historical price data for a single ticker.

    This is the **primary entry-point** for programmatic use.  It wraps
    ``yfinance.download``, applies missing-value treatment, optionally
    computes daily returns, and saves a CSV.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (e.g. ``"AAPL"``).
    start : str | None
        Start date in ``YYYY-MM-DD`` format.  Ignored when *period* is set.
    end : str | None
        End date in ``YYYY-MM-DD`` format.  Defaults to today.
    period : str | None
        Convenience look-back shorthand (``"1mo"``, ``"6mo"``, ``"1y"``, …).
        When provided, *start* and *end* are ignored.
    interval : str
        Bar interval — ``"1d"``, ``"1wk"``, ``"1mo"``, ``"1h"``, etc.
    clean_method : str
        Strategy for handling missing values:

        - ``"drop"``        — remove rows with any NaN
        - ``"ffill"``       — forward-fill then drop remaining leading NaNs
        - ``"bfill"``       — back-fill then drop remaining trailing NaNs
        - ``"interpolate"`` — linear interpolation then drop edges
    compute_returns : bool
        If True, append ``daily_return`` (simple %) and ``log_return`` columns.
    output_path : str | Path | None
        If provided, write the cleaned DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV data with an optional ``daily_return`` / ``log_return``
        column.  Index is ``DatetimeIndex`` named ``Date``.

    Raises
    ------
    ValueError
        If neither *start* / *end* nor *period* is specified.
    RuntimeError
        If ``yfinance`` returns an empty DataFrame (bad ticker, no data, …).

    Examples
    --------
    >>> df = fetch_stock_data("MSFT", start="2024-01-01", end="2024-06-01")
    >>> df.columns.tolist()
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'daily_return', 'log_return']
    """
    # ── Validate inputs ──────────────────────────────────────────────
    ticker = ticker.strip().upper()
    if not period and not start:
        raise ValueError(
            "Provide either 'period' (e.g. '1y') or a 'start' date."
        )

    logger.info(
        f"Fetching {ticker}  "
        f"{'period=' + period if period else f'start={start}  end={end or 'today'}'}"
        f"  interval={interval}"
    )

    # ── Download ─────────────────────────────────────────────────────
    raw_df = _download(ticker, start=start, end=end, period=period, interval=interval)

    # ── Clean ────────────────────────────────────────────────────────
    clean_df = _clean(raw_df, method=clean_method)
    logger.info(
        f"Cleaned {ticker}: {len(raw_df)} → {len(clean_df)} rows  "
        f"(method={clean_method})"
    )

    # ── Returns ──────────────────────────────────────────────────────
    if compute_returns:
        clean_df = _add_returns(clean_df)

    # ── Persist ──────────────────────────────────────────────────────
    if output_path:
        _save_csv(clean_df, output_path)

    return clean_df


def fetch_multiple(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    interval: str = "1d",
    clean_method: Literal["drop", "ffill", "bfill", "interpolate"] = "ffill",
    compute_returns: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """Download and clean data for multiple tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols.
    output_dir : str | Path | None
        Directory to save individual CSVs (one per ticker).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → cleaned DataFrame.

    See Also
    --------
    fetch_stock_data : Single-ticker variant with full parameter docs.
    """
    results: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        out = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out = Path(output_dir) / f"{t.upper()}_clean.csv"
        try:
            results[t.upper()] = fetch_stock_data(
                ticker=t,
                start=start,
                end=end,
                period=period,
                interval=interval,
                clean_method=clean_method,
                compute_returns=compute_returns,
                output_path=out,
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(f"Skipping {t}: {exc}")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _download(
    ticker: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Call ``yfinance.download`` with error handling.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV data exactly as returned by yfinance.

    Raises
    ------
    RuntimeError
        If the download yields an empty DataFrame.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required.  Install it with:  pip install yfinance"
        ) from exc

    try:
        if period:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
        else:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
    except Exception as exc:
        logger.error(f"yfinance download failed for {ticker}: {exc}")
        raise RuntimeError(f"Download failed for {ticker}: {exc}") from exc

    # yfinance may return MultiIndex columns for single ticker — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}'.  "
            "Check the ticker symbol and date range."
        )

    df.index.name = "Date"
    logger.debug(
        f"Downloaded {ticker}: {len(df)} rows, "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    return df


def _clean(
    df: pd.DataFrame,
    method: Literal["drop", "ffill", "bfill", "interpolate"] = "ffill",
) -> pd.DataFrame:
    """Handle missing / NaN values in OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw price data.
    method : str
        - ``"drop"``        — drop any row containing NaN.
        - ``"ffill"``       — forward-fill gaps, then drop leading NaNs.
        - ``"bfill"``       — back-fill gaps, then drop trailing NaNs.
        - ``"interpolate"`` — linear interpolation, then drop edges.

    Returns
    -------
    pd.DataFrame
        Data with no NaN values.
    """
    nan_before = int(df.isna().sum().sum())
    if nan_before:
        logger.warning(f"Found {nan_before} NaN values — applying '{method}'")

    if method == "drop":
        df = df.dropna()
    elif method == "ffill":
        df = df.ffill().dropna()
    elif method == "bfill":
        df = df.bfill().dropna()
    elif method == "interpolate":
        df = df.interpolate(method="linear").dropna()
    else:
        raise ValueError(
            f"Unknown clean_method '{method}'. "
            "Choose from: drop, ffill, bfill, interpolate."
        )

    # Drop duplicate index entries (weekends / holidays edge cases)
    df = df[~df.index.duplicated(keep="first")]

    nan_after = int(df.isna().sum().sum())
    if nan_after:
        logger.error(f"{nan_after} NaN values remain after cleaning!")

    return df


def _add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Append daily simple return and log return columns.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned price data.
    price_col : str
        Column to derive returns from (default ``"Close"``).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with ``daily_return`` and ``log_return`` appended.
    """
    df = df.copy()
    df["daily_return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    # The first row will be NaN by definition; drop it
    df.dropna(subset=["daily_return", "log_return"], inplace=True)

    logger.debug(
        f"Returns: mean={df['daily_return'].mean():.6f}  "
        f"std={df['daily_return'].std():.6f}  "
        f"min={df['daily_return'].min():.4f}  "
        f"max={df['daily_return'].max():.4f}"
    )
    return df


def _save_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info(f"Saved {len(df)} rows → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  CLASS INTERFACE  (wraps the functions for statefulness / config)
# ═══════════════════════════════════════════════════════════════════════


class StockDataFetcher:
    """Configurable fetcher bound to a list of tickers and output directory.

    Parameters
    ----------
    tickers : list[str] | None
        Equity symbols.  Falls back to ``config.yaml → data.tickers``.
    output_dir : Path | str | None
        Directory for raw CSVs.  Falls back to ``config.yaml → data.raw_dir``.
    clean_method : str
        Default missing-value strategy.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        clean_method: Literal["drop", "ffill", "bfill", "interpolate"] = "ffill",
    ) -> None:
        cfg = load_config()
        data_cfg = cfg.get("data", {})
        self.tickers = tickers or data_cfg.get("tickers", [])
        self.output_dir = Path(
            output_dir or ROOT_DIR / data_cfg.get("raw_dir", "data/raw")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clean_method = clean_method

    def fetch(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = "1y",
        interval: str = "1d",
        compute_returns: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Download, clean, and return data for all configured tickers.

        Parameters
        ----------
        start, end : str | None
            Date range (``YYYY-MM-DD``).  Ignored when *period* is set.
        period : str | None
            Look-back shorthand (e.g. ``"1y"``).
        interval : str
            Bar interval.
        compute_returns : bool
            Append return columns.

        Returns
        -------
        dict[str, pd.DataFrame]
            One cleaned DataFrame per ticker.
        """
        return fetch_multiple(
            tickers=self.tickers,
            start=start,
            end=end,
            period=period,
            interval=interval,
            clean_method=self.clean_method,
            compute_returns=compute_returns,
            output_dir=self.output_dir,
        )


# ═══════════════════════════════════════════════════════════════════════
#  NEWS DATA (unchanged from scaffold)
# ═══════════════════════════════════════════════════════════════════════


class NewsDataFetcher:
    """Fetch news articles from NewsAPI (or RSS fallback).

    Parameters
    ----------
    api_key : str | None
        NewsAPI key; falls back to ``config.yaml`` / env var.
    output_dir : Path | str | None
        Directory to persist raw JSON / CSV files.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        cfg = load_config()
        self.api_key = api_key or cfg.get("api_keys", {}).get("news_api", "")
        self.output_dir = Path(
            output_dir or ROOT_DIR / cfg.get("data", {}).get("raw_dir", "data/raw")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        query: str,
        from_date: Optional[dt.date] = None,
        to_date: Optional[dt.date] = None,
        page_size: int = 100,
    ) -> pd.DataFrame:
        """Query NewsAPI and return a DataFrame of articles.

        Parameters
        ----------
        query : str
            Free-text search query.
        from_date, to_date : date, optional
            Date range filter.
        page_size : int
            Max articles per request (API limit: 100).

        Returns
        -------
        pd.DataFrame
            Columns: ``title``, ``description``, ``content``,
            ``published_at``, ``source``, ``url``.
        """
        import requests

        params = {
            "q": query,
            "pageSize": page_size,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "publishedAt",
        }
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        logger.info(f"Fetching news for query='{query}' …")
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        df = pd.json_normalize(articles)
        if not df.empty:
            out_path = self.output_dir / f"news_{query.replace(' ', '_')}.csv"
            df.to_csv(out_path, index=False)
            logger.debug(f"  → saved {out_path}")

        return df


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY-POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and clean historical stock price data."
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--period", type=str, default="1y", help="Look-back period (e.g. 6mo, 1y, 2y)")
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval")
    parser.add_argument("--clean", type=str, default="ffill", choices=["drop", "ffill", "bfill", "interpolate"])
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    # Use date range if start is given, otherwise use period
    use_period = args.period if not args.start else None

    out_path = args.output or f"data/processed/{args.ticker.upper()}_clean.csv"

    df = fetch_stock_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        period=use_period,
        interval=args.interval,
        clean_method=args.clean,
        compute_returns=True,
        output_path=out_path,
    )

    print(f"\n{'═' * 60}")
    print(f"  {args.ticker.upper()}  —  {len(df)} rows  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    print(f"{'═' * 60}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\n{df.tail(10).to_string()}")
    print(f"\n  Avg daily return : {df['daily_return'].mean():+.5f}")
    print(f"  Return std dev   : {df['daily_return'].std():.5f}")
    print(f"  Max daily gain   : {df['daily_return'].max():+.4f}")
    print(f"  Max daily loss   : {df['daily_return'].min():+.4f}")
    print(f"\n  Saved → {out_path}")
