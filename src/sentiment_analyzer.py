"""
sentiment_analyzer.py
=====================
Score financial news headlines with two complementary approaches:

1. **VADER** — fast, rule-based lexicon baseline (no GPU, no download).
2. **FinBERT** — transformer fine-tuned on financial text for production
   quality sentiment (``ProsusAI/finbert``).

Both backends return the same schema so they are drop-in interchangeable.
The module also aggregates per-headline scores into **daily sentiment
features** suitable for downstream ML pipelines.

Output per headline
-------------------
``date | headline | label | positive | negative | neutral | compound``

Daily aggregation
-----------------
``date | sent_mean | sent_std | sent_min | sent_max | sent_positive_pct
| sent_negative_pct | sent_neutral_pct | sent_count``

Quick-start
-----------
    from src.sentiment_analyzer import (
        SentimentAnalyzer,
        score_headlines,
        aggregate_daily_sentiment,
    )

    # Function interface (simplest)
    scored_df = score_headlines(headlines_df, backend="vader")
    daily_df  = aggregate_daily_sentiment(scored_df)

    # Class interface
    analyzer = SentimentAnalyzer(backend="finbert")
    scored   = analyzer.score_dataframe(headlines_df)
    daily    = analyzer.aggregate_daily(scored)

CLI
---
    python -m src.sentiment_analyzer --input data/raw/AAPL_news.csv --backend vader
"""

from __future__ import annotations

import abc
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import yaml
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# Canonical per-headline output columns
HEADLINE_COLUMNS = [
    "date",
    "headline",
    "label",       # POSITIVE | NEGATIVE | NEUTRAL
    "positive",    # probability / score
    "negative",
    "neutral",
    "compound",    # single summary score in [-1, +1]
]


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ═══════════════════════════════════════════════════════════════════════
#  ABSTRACT BACKEND
# ═══════════════════════════════════════════════════════════════════════


class _SentimentBackend(abc.ABC):
    """Contract that every scoring backend must satisfy."""

    name: str = "base"

    @abc.abstractmethod
    def score_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Score a list of texts.

        Returns
        -------
        list[dict]
            Each dict must contain:
            ``label``, ``positive``, ``negative``, ``neutral``, ``compound``.
        """

    def score_one(self, text: str) -> Dict[str, Any]:
        """Score a single text (default: call batch with size 1)."""
        return self.score_batch([text])[0]


# ═══════════════════════════════════════════════════════════════════════
#  VADER BACKEND  (baseline)
# ═══════════════════════════════════════════════════════════════════════


class _VaderBackend(_SentimentBackend):
    """VADER lexicon-based sentiment — fast, no model download.

    Pros:
    - Zero latency, works offline
    - Good for social-media-style text

    Cons:
    - Not trained on financial language
    - Misses sarcasm and domain-specific nuance
    """

    name = "vader"

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError(
                "VADER is required.  Install: pip install vaderSentiment"
            ) from exc
        self._analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment backend initialised")

    def score_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            vs = self._analyzer.polarity_scores(text or "")
            compound = vs["compound"]
            label = (
                "POSITIVE"  if compound >= 0.05
                else "NEGATIVE" if compound <= -0.05
                else "NEUTRAL"
            )
            results.append(
                {
                    "label": label,
                    "positive": round(vs["pos"], 4),
                    "negative": round(vs["neg"], 4),
                    "neutral": round(vs["neu"], 4),
                    "compound": round(compound, 4),
                }
            )
        return results


# ═══════════════════════════════════════════════════════════════════════
#  FINBERT BACKEND  (advanced)
# ═══════════════════════════════════════════════════════════════════════


class _FinBERTBackend(_SentimentBackend):
    """ProsusAI/finbert — transformer fine-tuned on financial text.

    The model outputs three-class probabilities (positive, negative,
    neutral) which we also convert to a single compound score in [-1, 1]
    for compatibility with the VADER interface:

        compound = positive - negative

    Pros:
    - Trained specifically on financial communications (10-K, analyst reports)
    - Much higher accuracy on earnings / guidance / merger language

    Cons:
    - Requires ~500 MB model download on first run
    - Slower than VADER (~50–200 ms per headline on CPU)
    """

    name = "finbert"
    MODEL_ID = "ProsusAI/finbert"

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or self.MODEL_ID
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers + torch are required for FinBERT.  "
                "Install: pip install transformers torch"
            ) from exc

        logger.info(f"Loading FinBERT model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model,
            tokenizer=self._tokenizer,
            truncation=True,
            max_length=self.max_length,
            top_k=None,             # return all 3 class probabilities
            device=device if device is not None else -1,
        )
        # FinBERT label mapping (model config order)
        self._label_map = {
            "positive": "POSITIVE",
            "negative": "NEGATIVE",
            "neutral": "NEUTRAL",
        }
        logger.info("FinBERT backend ready")

    def score_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        # Process in chunks to avoid OOM
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:self.max_length] if t else "" for t in texts[i : i + self.batch_size]]
            outputs = self._pipeline(batch)

            for output in outputs:
                # output is a list of dicts: [{"label": "positive", "score": ...}, ...]
                probs = {item["label"].lower(): item["score"] for item in output}
                pos = round(probs.get("positive", 0.0), 4)
                neg = round(probs.get("negative", 0.0), 4)
                neu = round(probs.get("neutral", 0.0), 4)
                compound = round(pos - neg, 4)

                # Winning label
                best = max(output, key=lambda x: x["score"])
                label = self._label_map.get(best["label"].lower(), "NEUTRAL")

                results.append(
                    {
                        "label": label,
                        "positive": pos,
                        "negative": neg,
                        "neutral": neu,
                        "compound": compound,
                    }
                )
        return results


# ═══════════════════════════════════════════════════════════════════════
#  BACKEND REGISTRY
# ═══════════════════════════════════════════════════════════════════════

BackendName = Literal["vader", "finbert"]

_BACKEND_REGISTRY: Dict[str, type] = {
    "vader": _VaderBackend,
    "finbert": _FinBERTBackend,
}


def _build_backend(
    name: str, **kwargs: Any
) -> _SentimentBackend:
    """Instantiate a sentiment backend by name."""
    cls = _BACKEND_REGISTRY.get(name.lower())
    if cls is None:
        available = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown backend '{name}'.  Available: {available}"
        )
    return cls(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC CLASS INTERFACE
# ═══════════════════════════════════════════════════════════════════════


class SentimentAnalyzer:
    """Unified sentiment analysis with swappable backends.

    Parameters
    ----------
    backend : str
        ``"vader"`` (baseline) or ``"finbert"`` (advanced).
    model_name : str | None
        Override HuggingFace model ID (FinBERT only).
    batch_size : int
        Chunk size for transformer inference.
    max_length : int
        Max token length for transformer inputs.
    auto_fallback : bool
        If True and FinBERT fails to load, silently fall back to VADER.

    Examples
    --------
    >>> analyzer = SentimentAnalyzer(backend="vader")
    >>> analyzer.score_text("Tesla stock surges 15%!")
    {'label': 'POSITIVE', 'positive': 0.273, 'negative': 0.0, 'neutral': 0.727, 'compound': 0.7184}
    """

    def __init__(
        self,
        backend: BackendName = "vader",
        model_name: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        auto_fallback: bool = True,
    ) -> None:
        self.backend_name = backend.lower()
        self.auto_fallback = auto_fallback

        kwargs: Dict[str, Any] = {}
        if self.backend_name == "finbert":
            kwargs.update(
                model_name=model_name,
                batch_size=batch_size,
                max_length=max_length,
            )

        try:
            self._backend = _build_backend(self.backend_name, **kwargs)
        except (ImportError, OSError) as exc:
            if auto_fallback and self.backend_name != "vader":
                logger.warning(
                    f"Failed to load {self.backend_name}: {exc}  "
                    "→ falling back to VADER"
                )
                self._backend = _build_backend("vader")
                self.backend_name = "vader"
            else:
                raise

    # ------------------------------------------------------------------
    # Per-text scoring
    # ------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, Any]:
        """Score a single headline / sentence.

        Returns
        -------
        dict
            Keys: ``label``, ``positive``, ``negative``, ``neutral``,
            ``compound``.
        """
        return self._backend.score_one(text)

    def score_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch-score a list of texts.

        Parameters
        ----------
        texts : list[str]
            Raw headline strings.

        Returns
        -------
        list[dict]
            One sentiment dict per input text.
        """
        logger.info(f"Scoring {len(texts)} texts with {self.backend_name}")
        return self._backend.score_batch(texts)

    # ------------------------------------------------------------------
    # DataFrame scoring
    # ------------------------------------------------------------------

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Add per-row sentiment columns to a headlines DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *text_column* and *date_column*.
        text_column : str
            Column with headline text.
        date_column : str
            Column with dates (parsed to datetime).

        Returns
        -------
        pd.DataFrame
            Original rows + ``label``, ``positive``, ``negative``,
            ``neutral``, ``compound`` columns.
        """
        df = df.copy()
        texts = df[text_column].fillna("").astype(str).tolist()

        logger.info(
            f"Scoring {len(texts)} headlines "
            f"(backend={self.backend_name}, column='{text_column}')"
        )

        results = self._backend.score_batch(texts)

        df["label"]    = [r["label"]    for r in results]
        df["positive"] = [r["positive"] for r in results]
        df["negative"] = [r["negative"] for r in results]
        df["neutral"]  = [r["neutral"]  for r in results]
        df["compound"] = [r["compound"] for r in results]

        # Ensure date is datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(
                df[date_column], errors="coerce", utc=True
            )

        logger.info(
            f"Sentiment breakdown → "
            f"POS={sum(1 for r in results if r['label']=='POSITIVE')}  "
            f"NEG={sum(1 for r in results if r['label']=='NEGATIVE')}  "
            f"NEU={sum(1 for r in results if r['label']=='NEUTRAL')}"
        )
        return df

    # ------------------------------------------------------------------
    # Daily aggregation
    # ------------------------------------------------------------------

    def aggregate_daily(
        self,
        scored_df: pd.DataFrame,
        date_column: str = "date",
        score_column: str = "compound",
    ) -> pd.DataFrame:
        """Aggregate per-headline scores into daily sentiment features.

        Delegates to the module-level ``aggregate_daily_sentiment()``
        function.
        """
        return aggregate_daily_sentiment(
            scored_df,
            date_column=date_column,
            score_column=score_column,
        )


# ═══════════════════════════════════════════════════════════════════════
#  DAILY AGGREGATION  (stand-alone function)
# ═══════════════════════════════════════════════════════════════════════


def aggregate_daily_sentiment(
    scored_df: pd.DataFrame,
    date_column: str = "date",
    score_column: str = "compound",
) -> pd.DataFrame:
    """Roll up per-headline sentiment into daily feature vectors.

    For each calendar date this produces:

    - **sent_mean** — average compound score
    - **sent_std** — standard deviation (disagreement among headlines)
    - **sent_min / sent_max** — extremes
    - **sent_median** — median compound score
    - **sent_positive_pct** — fraction of POSITIVE headlines
    - **sent_negative_pct** — fraction of NEGATIVE headlines
    - **sent_neutral_pct** — fraction of NEUTRAL headlines
    - **sent_count** — total headlines that day

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output from ``SentimentAnalyzer.score_dataframe()`` — must contain
        ``date_column``, ``score_column``, and ``label`` columns.
    date_column : str
        Column with datetime values.
    score_column : str
        Numeric score column to aggregate (default ``"compound"``).

    Returns
    -------
    pd.DataFrame
        One row per date, indexed by date, sorted chronologically.

    Examples
    --------
    >>> daily = aggregate_daily_sentiment(scored_df)
    >>> daily.columns.tolist()
    ['sent_mean', 'sent_std', 'sent_min', 'sent_max', 'sent_median',
     'sent_positive_pct', 'sent_negative_pct', 'sent_neutral_pct', 'sent_count']
    """
    df = scored_df.copy()

    # Normalise date to date-only (no time component)
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce", utc=True)
    df["_date_key"] = df[date_column].dt.date

    # ── Numeric aggregations ──
    numeric_agg = (
        df.groupby("_date_key")[score_column]
        .agg(
            sent_mean="mean",
            sent_std="std",
            sent_min="min",
            sent_max="max",
            sent_median="median",
            sent_count="count",
        )
        .fillna({"sent_std": 0.0})
    )

    # ── Label-proportion aggregations ──
    label_counts = (
        df.groupby(["_date_key", "label"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all three label columns exist
    for lbl in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        if lbl not in label_counts.columns:
            label_counts[lbl] = 0

    totals = label_counts.sum(axis=1)
    label_pcts = pd.DataFrame(
        {
            "sent_positive_pct": (label_counts["POSITIVE"] / totals).round(4),
            "sent_negative_pct": (label_counts["NEGATIVE"] / totals).round(4),
            "sent_neutral_pct":  (label_counts["NEUTRAL"]  / totals).round(4),
        },
        index=label_counts.index,
    )

    # ── Combine ──
    daily = numeric_agg.join(label_pcts)
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily.sort_index(inplace=True)

    logger.info(
        f"Daily aggregation: {len(daily)} days, "
        f"mean compound={daily['sent_mean'].mean():+.4f}"
    )
    return daily


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION INTERFACE
# ═══════════════════════════════════════════════════════════════════════


def score_headlines(
    df: pd.DataFrame,
    text_column: str = "headline",
    date_column: str = "date",
    backend: BackendName = "vader",
    output_path: Optional[Union[str, Path]] = None,
    **backend_kwargs: Any,
) -> pd.DataFrame:
    """One-call function: score every headline in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Headlines with at least *text_column* and *date_column*.
    backend : str
        ``"vader"`` or ``"finbert"``.
    output_path : str | Path | None
        Save scored DataFrame to CSV if provided.
    **backend_kwargs
        Forwarded to ``SentimentAnalyzer`` (e.g. ``model_name``).

    Returns
    -------
    pd.DataFrame
        Input rows enriched with sentiment columns.
    """
    analyzer = SentimentAnalyzer(backend=backend, **backend_kwargs)
    scored = analyzer.score_dataframe(
        df, text_column=text_column, date_column=date_column
    )
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(p, index=False)
        logger.info(f"Scored headlines saved → {p}")
    return scored


def build_daily_sentiment(
    df: pd.DataFrame,
    text_column: str = "headline",
    date_column: str = "date",
    backend: BackendName = "vader",
    output_path: Optional[Union[str, Path]] = None,
    **backend_kwargs: Any,
) -> pd.DataFrame:
    """One-call function: score headlines **and** aggregate to daily features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw headlines DataFrame.
    backend : str
        ``"vader"`` or ``"finbert"``.
    output_path : str | Path | None
        Save daily features to CSV if provided.

    Returns
    -------
    pd.DataFrame
        Daily sentiment features indexed by date.

    Examples
    --------
    >>> daily = build_daily_sentiment(news_df, backend="finbert")
    >>> daily[["sent_mean", "sent_positive_pct"]].tail()
    """
    scored = score_headlines(
        df, text_column=text_column, date_column=date_column,
        backend=backend, **backend_kwargs,
    )
    daily = aggregate_daily_sentiment(scored, date_column=date_column)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(p)
        logger.info(f"Daily sentiment features saved → {p}")
    return daily


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score financial news headlines for sentiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.sentiment_analyzer --input data/raw/AAPL_news.csv --backend vader\n"
            "  python -m src.sentiment_analyzer --input data/raw/AAPL_news.csv --backend finbert --daily\n"
            "  python -m src.sentiment_analyzer --demo\n"
        ),
    )
    parser.add_argument("--input", type=str, default=None, help="Input CSV with headlines")
    parser.add_argument("--text-col", type=str, default="headline", help="Column name for text")
    parser.add_argument("--date-col", type=str, default="date", help="Column name for date")
    parser.add_argument("--backend", type=str, default="vader", choices=["vader", "finbert"])
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for scored headlines")
    parser.add_argument("--daily", action="store_true", help="Also produce daily aggregation")
    parser.add_argument("--daily-output", type=str, default=None, help="Output path for daily CSV")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo with sample headlines")
    args = parser.parse_args()

    if args.demo:
        # ── Demo mode ──
        print(f"\n{'═' * 70}")
        print(f"  Sentiment Analysis Demo  (backend: {args.backend})")
        print(f"{'═' * 70}\n")

        sample_headlines = pd.DataFrame(
            {
                "date": [
                    "2024-06-10", "2024-06-10", "2024-06-10",
                    "2024-06-11", "2024-06-11",
                    "2024-06-12", "2024-06-12", "2024-06-12",
                ],
                "headline": [
                    "Apple stock surges 8% after record iPhone sales",
                    "Fed signals potential rate cut — markets rally",
                    "Oil prices drop sharply amid global demand concerns",
                    "Tesla faces another recall affecting 500k vehicles",
                    "Microsoft Azure revenue beats analyst expectations",
                    "Markets close flat on mixed earnings signals",
                    "Goldman Sachs upgrades NVIDIA to strong buy",
                    "Inflation data comes in hotter than expected",
                ],
            }
        )

        analyzer = SentimentAnalyzer(backend=args.backend)
        scored = analyzer.score_dataframe(sample_headlines)

        print("Per-headline scores:")
        print("-" * 70)
        for _, row in scored.iterrows():
            print(
                f"  {row['label']:>8}  "
                f"(cmpd={row['compound']:+.4f}  "
                f"pos={row['positive']:.3f}  "
                f"neg={row['negative']:.3f}  "
                f"neu={row['neutral']:.3f})  "
                f"{row['headline'][:55]}…"
            )

        print(f"\n{'─' * 70}")
        print("Daily aggregation:")
        print("-" * 70)
        daily = analyzer.aggregate_daily(scored)
        print(daily.to_string())
        print()

    elif args.input:
        # ── File mode ──
        input_df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(input_df)} rows from {args.input}")

        scored = score_headlines(
            input_df,
            text_column=args.text_col,
            date_column=args.date_col,
            backend=args.backend,
            output_path=args.output or args.input.replace(".csv", "_scored.csv"),
        )

        if args.daily:
            daily = aggregate_daily_sentiment(scored, date_column=args.date_col)
            daily_path = args.daily_output or args.input.replace(".csv", "_daily.csv")
            daily.to_csv(daily_path)
            logger.info(f"Daily features saved → {daily_path}")
            print(f"\nDaily sentiment ({len(daily)} days):")
            print(daily.to_string())

    else:
        parser.print_help()
