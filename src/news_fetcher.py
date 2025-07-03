"""
news_fetcher.py
===============
Fetch financial news headlines for a given ticker and date range from
**pluggable** data sources.

Architecture
------------
Swapping APIs is a one-line change — every source implements the same
``NewsSource`` abstract interface.  Register new sources in
``SOURCE_REGISTRY`` and you're done.

    ┌──────────────┐
    │  NewsSource   │  ← abstract base class
    └──────┬───────┘
           │
    ┌──────┴──────────────────────────────────┐
    │      │              │           │        │
    ▼      ▼              ▼           ▼        ▼
  NewsAPI  GNews      Finnhub     RSS      (yours)

Output schema (every source returns exactly this)
-------------------------------------------------
    Date  │  Headline  │  Source  │  URL

Quick-start
-----------
    from src.news_fetcher import fetch_news

    df = fetch_news(
        ticker="AAPL",
        start="2024-01-01",
        end="2024-06-01",
        source="gnews",                      # swap to "newsapi", "finnhub", "rss"
        output_path="data/raw/AAPL_news.csv",
    )

CLI
---
    python -m src.news_fetcher --ticker AAPL --start 2024-01-01 --source gnews
"""

from __future__ import annotations

import abc
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Logger setup
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
# Configuration helpers
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# Canonical column names — every source MUST return these
COLUMNS = ["date", "headline", "source", "url"]


def _load_config() -> dict:
    """Load ``config/config.yaml``; returns ``{}`` if missing."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _get_api_key(name: str) -> str:
    """Retrieve an API key from config → environment → empty string."""
    import os

    cfg = _load_config()
    key = cfg.get("api_keys", {}).get(name, "")
    # Support ${ENV_VAR} placeholders in the YAML
    if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
        key = os.environ.get(key[2:-1], "")
    return key or os.environ.get(name.upper() + "_KEY", "")


# ═══════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════


class NewsSource(abc.ABC):
    """Base class that every news-data provider must implement.

    To add a new source:

    1. Subclass ``NewsSource``.
    2. Implement ``fetch_headlines()``.
    3. Register in ``SOURCE_REGISTRY`` at the bottom of this file.

    Every implementation **must** return a ``pd.DataFrame`` with exactly
    the columns defined in ``COLUMNS``: ``date``, ``headline``,
    ``source``, ``url``.
    """

    name: str = "base"

    @abc.abstractmethod
    def fetch_headlines(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """Fetch headlines matching *query* within the date window.

        Parameters
        ----------
        query : str
            Free-text or ticker-based search term.
        start : str | None
            ``YYYY-MM-DD`` start date.
        end : str | None
            ``YYYY-MM-DD`` end date (inclusive).
        max_results : int
            Upper limit on returned articles.

        Returns
        -------
        pd.DataFrame
            Must contain columns: ``date``, ``headline``, ``source``, ``url``.
        """

    # ── Shared utilities ──

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce canonical schema, sort by date descending."""
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.dropna(subset=["date", "headline"], inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.sort_values("date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _ticker_to_query(ticker: str) -> str:
        """Convert a ticker symbol to a more search-friendly query string."""
        # Common mappings — extend as needed
        TICKER_MAP: Dict[str, str] = {
            "AAPL": "Apple",
            "GOOGL": "Google Alphabet",
            "GOOG": "Google Alphabet",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta Facebook",
            "NVDA": "Nvidia",
            "NFLX": "Netflix",
            "JPM": "JPMorgan",
        }
        company = TICKER_MAP.get(ticker.upper(), ticker)
        return f"{company} stock"


# ═══════════════════════════════════════════════════════════════════════
#  CONCRETE SOURCES
# ═══════════════════════════════════════════════════════════════════════


class NewsAPISource(NewsSource):
    """Fetch from `newsapi.org <https://newsapi.org/>`_ (free tier: 100 req/day).

    Requires an API key set via ``config.yaml → api_keys.news_api``
    or the ``NEWS_API_KEY`` environment variable.
    """

    name = "newsapi"
    BASE_URL = "https://newsapi.org/v2/everything"

    def fetch_headlines(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        import requests

        api_key = _get_api_key("news_api")
        if not api_key:
            raise ValueError(
                "NewsAPI key is required.  "
                "Set NEWS_API_KEY env var or config.yaml → api_keys.news_api"
            )

        params: Dict[str, Any] = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 100),
            "apiKey": api_key,
        }
        if start:
            params["from"] = start
        if end:
            params["to"] = end

        logger.info(f"[NewsAPI] Searching: '{query}'  ({start} → {end})")
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {data.get('message', 'unknown')}")

        articles = data.get("articles", [])
        logger.info(f"[NewsAPI] Received {len(articles)} articles")

        rows = [
            {
                "date": a.get("publishedAt"),
                "headline": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "url": a.get("url"),
            }
            for a in articles
        ]
        return self._normalize(pd.DataFrame(rows))


class GNewsSource(NewsSource):
    """Fetch from `GNews.io <https://gnews.io/>`_ (free tier: 100 req/day).

    Requires a GNews API key set via ``GNEWS_KEY`` env var or
    ``config.yaml → api_keys.gnews``.
    """

    name = "gnews"
    BASE_URL = "https://gnews.io/api/v4/search"

    def fetch_headlines(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        import requests

        api_key = _get_api_key("gnews")
        if not api_key:
            raise ValueError(
                "GNews key is required.  "
                "Set GNEWS_KEY env var or config.yaml → api_keys.gnews"
            )

        params: Dict[str, Any] = {
            "q": query,
            "lang": "en",
            "max": min(max_results, 100),
            "token": api_key,
            "sortby": "publishedAt",
        }
        if start:
            params["from"] = f"{start}T00:00:00Z"
        if end:
            params["to"] = f"{end}T23:59:59Z"

        logger.info(f"[GNews] Searching: '{query}'  ({start} → {end})")
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        articles = resp.json().get("articles", [])
        logger.info(f"[GNews] Received {len(articles)} articles")

        rows = [
            {
                "date": a.get("publishedAt"),
                "headline": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "url": a.get("url"),
            }
            for a in articles
        ]
        return self._normalize(pd.DataFrame(rows))


class FinnhubSource(NewsSource):
    """Fetch from `Finnhub.io <https://finnhub.io/>`_ (free tier: 60 calls/min).

    Requires a Finnhub API key set via ``FINNHUB_KEY`` env var or
    ``config.yaml → api_keys.finnhub``.
    """

    name = "finnhub"
    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def fetch_headlines(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        import requests

        api_key = _get_api_key("finnhub")
        if not api_key:
            raise ValueError(
                "Finnhub key is required.  "
                "Set FINNHUB_KEY env var or config.yaml → api_keys.finnhub"
            )

        # Finnhub's company-news endpoint takes the raw ticker symbol
        ticker = query.split()[0].upper()
        today = dt.date.today().isoformat()

        params: Dict[str, Any] = {
            "symbol": ticker,
            "from": start or (dt.date.today() - dt.timedelta(days=30)).isoformat(),
            "to": end or today,
            "token": api_key,
        }

        logger.info(f"[Finnhub] Company news: {ticker}  ({params['from']} → {params['to']})")
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        articles = resp.json()
        if not isinstance(articles, list):
            raise RuntimeError(f"Finnhub unexpected response: {articles}")

        logger.info(f"[Finnhub] Received {len(articles)} articles")

        rows = [
            {
                "date": dt.datetime.fromtimestamp(a["datetime"], tz=dt.timezone.utc).isoformat()
                if a.get("datetime")
                else None,
                "headline": a.get("headline"),
                "source": a.get("source"),
                "url": a.get("url"),
            }
            for a in articles[:max_results]
        ]
        return self._normalize(pd.DataFrame(rows))


class RSSSource(NewsSource):
    """Fetch from Google News RSS — **no API key needed**.

    This is the zero-config fallback.  Results are limited to
    whatever Google's RSS feed returns (typically ~20–50 items).
    """

    name = "rss"
    BASE_URL = "https://news.google.com/rss/search"

    def fetch_headlines(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        try:
            import feedparser
        except ImportError as exc:
            raise ImportError(
                "feedparser is required for RSS source.  "
                "Install it with:  pip install feedparser"
            ) from exc

        url = f"{self.BASE_URL}?q={query}&hl=en-US&gl=US&ceid=US:en"

        logger.info(f"[RSS] Fetching Google News RSS for: '{query}'")
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            raise RuntimeError(
                f"RSS feed parse error: {feed.bozo_exception}"
            )

        logger.info(f"[RSS] Received {len(feed.entries)} entries")

        rows: List[Dict[str, Any]] = []
        for entry in feed.entries[:max_results]:
            published = entry.get("published_parsed")
            date_str = (
                dt.datetime(*published[:6], tzinfo=dt.timezone.utc).isoformat()
                if published
                else None
            )
            # Google RSS wraps the real source in the title: "Headline - Source"
            title = entry.get("title", "")
            source_name = ""
            if " - " in title:
                title, source_name = title.rsplit(" - ", 1)

            rows.append(
                {
                    "date": date_str,
                    "headline": title.strip(),
                    "source": source_name.strip() or "Google News",
                    "url": entry.get("link"),
                }
            )

        df = self._normalize(pd.DataFrame(rows))

        # Post-filter by date range if requested (RSS doesn't support it natively)
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= f"{end} 23:59:59"]

        return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  SOURCE REGISTRY  —  add new sources here
# ═══════════════════════════════════════════════════════════════════════

SOURCE_REGISTRY: Dict[str, Type[NewsSource]] = {
    "newsapi": NewsAPISource,
    "gnews": GNewsSource,
    "finnhub": FinnhubSource,
    "rss": RSSSource,
}

SourceName = Literal["newsapi", "gnews", "finnhub", "rss"]


def get_source(name: str) -> NewsSource:
    """Factory — instantiate a registered ``NewsSource`` by name.

    Parameters
    ----------
    name : str
        Key in ``SOURCE_REGISTRY`` (``"newsapi"``, ``"gnews"``,
        ``"finnhub"``, ``"rss"``).

    Returns
    -------
    NewsSource

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    try:
        return SOURCE_REGISTRY[name.lower()]()
    except KeyError:
        available = ", ".join(sorted(SOURCE_REGISTRY))
        raise KeyError(
            f"Unknown news source '{name}'.  Available: {available}"
        )


def register_source(name: str, cls: Type[NewsSource]) -> None:
    """Register a custom ``NewsSource`` implementation at runtime.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"bloomberg"``).
    cls : type[NewsSource]
        Subclass of ``NewsSource``.

    Example
    -------
    >>> class BloombergSource(NewsSource):
    ...     name = "bloomberg"
    ...     def fetch_headlines(self, query, **kw):
    ...         ...
    >>> register_source("bloomberg", BloombergSource)
    >>> df = fetch_news("AAPL", source="bloomberg")
    """
    if not issubclass(cls, NewsSource):
        raise TypeError(f"{cls} must be a subclass of NewsSource")
    SOURCE_REGISTRY[name.lower()] = cls
    logger.info(f"Registered news source: '{name}'")


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION INTERFACE
# ═══════════════════════════════════════════════════════════════════════


def fetch_news(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: SourceName = "rss",
    max_results: int = 100,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Fetch financial news headlines for a ticker — one function, any source.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    start : str | None
        ``YYYY-MM-DD`` start date.
    end : str | None
        ``YYYY-MM-DD`` end date (inclusive).
    source : str
        News source backend — ``"rss"`` (default, no key needed),
        ``"newsapi"``, ``"gnews"``, or ``"finnhub"``.
    max_results : int
        Maximum number of headlines.
    output_path : str | Path | None
        If given, save the DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``headline``, ``source``, ``url``.

    Raises
    ------
    KeyError
        If *source* is not registered.
    ValueError
        If a required API key is missing.
    RuntimeError
        If the API returns an error or no data.

    Examples
    --------
    >>> # Zero-config (uses Google News RSS — no key)
    >>> df = fetch_news("TSLA", start="2024-06-01", end="2024-06-30")

    >>> # Use NewsAPI
    >>> df = fetch_news("MSFT", source="newsapi", max_results=50,
    ...                 output_path="data/raw/MSFT_news.csv")

    >>> # Swap to Finnhub — just change one string
    >>> df = fetch_news("MSFT", source="finnhub")
    """
    ticker = ticker.strip().upper()
    provider = get_source(source)

    # Finnhub wants the raw ticker; others work better with company names
    if source == "finnhub":
        query = ticker
    else:
        query = provider._ticker_to_query(ticker)

    logger.info(f"Fetching news for {ticker} via {provider.name}")

    df = provider.fetch_headlines(
        query=query,
        start=start,
        end=end,
        max_results=max_results,
    )

    logger.info(f"Retrieved {len(df)} headlines for {ticker}")

    if output_path:
        _save(df, output_path)

    return df


def fetch_news_multi(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: SourceName = "rss",
    max_results: int = 100,
    output_dir: Optional[Union[str, Path]] = None,
    delay: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """Fetch news for multiple tickers with rate-limit-friendly delays.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols.
    delay : float
        Seconds to sleep between API calls (respect rate limits).
    output_dir : str | Path | None
        Directory for per-ticker CSVs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → headlines DataFrame.
    """
    results: Dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers):
        out = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out = Path(output_dir) / f"{ticker.upper()}_news.csv"
        try:
            results[ticker.upper()] = fetch_news(
                ticker=ticker,
                start=start,
                end=end,
                source=source,
                max_results=max_results,
                output_path=out,
            )
        except Exception as exc:
            logger.error(f"Failed to fetch news for {ticker}: {exc}")

        # Rate-limit delay between calls (skip after last)
        if delay and i < len(tickers) - 1:
            time.sleep(delay)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _save(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Save DataFrame to CSV, creating directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(df)} headlines → {path}")
    return path


def list_sources() -> List[str]:
    """Return the names of all registered news sources."""
    return sorted(SOURCE_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch financial news headlines for a stock ticker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.news_fetcher --ticker AAPL\n"
            "  python -m src.news_fetcher --ticker TSLA --source newsapi --start 2024-01-01\n"
            "  python -m src.news_fetcher --ticker MSFT --source finnhub --output data/raw/MSFT.csv\n"
        ),
    )
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--source",
        type=str,
        default="rss",
        choices=list(SOURCE_REGISTRY.keys()),
        help="News data source (default: rss)",
    )
    parser.add_argument("--max", type=int, default=50, help="Max headlines to fetch")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    out_path = args.output or f"data/raw/{args.ticker.upper()}_news.csv"

    df = fetch_news(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        source=args.source,
        max_results=args.max,
        output_path=out_path,
    )

    # Pretty-print results
    print(f"\n{'═' * 80}")
    print(f"  {args.ticker.upper()} — {len(df)} headlines via {args.source}")
    print(f"{'═' * 80}\n")

    if df.empty:
        print("  No headlines found.")
    else:
        # Truncate headline for terminal display
        display = df.copy()
        display["headline"] = display["headline"].str[:70] + "…"
        display["url"] = display["url"].str[:40] + "…"
        print(display.to_string(index=False))

    print(f"\n  Saved → {out_path}")
