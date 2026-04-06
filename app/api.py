"""
api.py
======
FastAPI service for stock prediction, sentiment scoring, and backtesting.

Endpoints
---------
- ``GET  /health``              -- Service health check
- ``GET  /predict/{ticker}``    -- Run full prediction pipeline for a ticker
- ``POST /sentiment``           -- Score arbitrary texts for sentiment
- ``GET  /backtest/{ticker}``   -- Quick backtest on a ticker
- ``GET  /models``              -- List available saved models

Run
---
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

Examples
--------
    curl http://localhost:8000/health
    curl "http://localhost:8000/predict/AAPL?period=6mo"
    curl "http://localhost:8000/backtest/AAPL?period=1y&commission=0.001"
    curl -X POST http://localhost:8000/sentiment \\
         -H "Content-Type: application/json" \\
         -d '{"texts": ["Apple stock surges 15%!"]}'
"""

from __future__ import annotations

import datetime as dt
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Stock Sentiment Predictor API",
    version="0.2.1",
    description=(
        "Predict stock price direction from news sentiment and technical "
        "features.  Supports real-time prediction, sentiment scoring, "
        "and backtesting."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS -- allow all origins in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
#  Pydantic Schemas
# ===================================================================


# -- Health ----------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    modules_available: Dict[str, bool]


# -- Prediction ------------------------------------------------------------

class SentimentSummary(BaseModel):
    """Aggregated sentiment stats for the recent window."""
    mean_score: float = Field(..., description="Mean compound sentiment")
    std_score: float = Field(0.0, description="Sentiment volatility")
    positive_pct: float = Field(0.0, description="% positive headlines")
    negative_pct: float = Field(0.0, description="% negative headlines")
    neutral_pct: float = Field(0.0, description="% neutral headlines")
    headline_count: int = Field(0, description="Headlines analyzed")
    top_headlines: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top scored recent headlines",
    )


class PerformanceMetrics(BaseModel):
    """Basic model/data quality metrics returned with predictions."""
    data_points: int = Field(0, description="Number of bars used")
    feature_count: int = Field(0, description="Number of features built")
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    recent_volatility: Optional[float] = Field(
        None, description="20-day rolling volatility"
    )
    recent_trend: Optional[str] = Field(
        None, description="Short-term trend label"
    )
    current_rsi: Optional[float] = None


class PredictionResponse(BaseModel):
    """Full prediction response with direction, confidence, and context."""
    ticker: str
    direction: str = Field(..., description="UP or DOWN")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Model confidence [0..1]")
    model_type: str = Field(..., description="Model used for prediction")
    features_used: int = Field(0, description="Number of features")
    current_price: Optional[float] = None
    prediction_date: str = Field(..., description="Date of prediction")
    sentiment: Optional[SentimentSummary] = None
    performance: Optional[PerformanceMetrics] = None
    pipeline_status: str = Field(
        "ok", description="ok | partial | error"
    )
    warnings: List[str] = Field(default_factory=list)


# -- Sentiment -------------------------------------------------------------

class SentimentRequest(BaseModel):
    texts: List[str]
    backend: str = "vader"


class SentimentResponse(BaseModel):
    results: List[Dict[str, Any]]
    backend_used: str


# -- Backtest ---------------------------------------------------------------

class BacktestResponse(BaseModel):
    ticker: str
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    hit_rate_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    exposure_pct: float
    buy_and_hold_return_pct: Optional[float] = None
    outperformance_pct: Optional[float] = None
    data_points: int = 0
    period: str = ""


# -- Models -----------------------------------------------------------------

class ModelInfo(BaseModel):
    filename: str
    model_type: Optional[str] = None
    test_accuracy: Optional[float] = None
    test_f1: Optional[float] = None
    n_features: Optional[int] = None
    file_size_kb: float = 0


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    count: int


# ===================================================================
#  Helper: Module availability check
# ===================================================================


def _check_modules() -> Dict[str, bool]:
    """Check which pipeline modules can be imported."""
    modules = {}
    for name in [
        "src.data_fetcher",
        "src.news_fetcher",
        "src.sentiment_analyzer",
        "src.feature_engineering",
        "src.model_training",
        "src.deep_learning",
        "src.hybrid_model",
        "src.backtesting",
    ]:
        try:
            __import__(name)
            modules[name.split(".")[-1]] = True
        except ImportError:
            modules[name.split(".")[-1]] = False
    return modules


# ===================================================================
#  Endpoints
# ===================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check with module availability."""
    return HealthResponse(
        status="ok",
        version=app.version,
        modules_available=_check_modules(),
    )


@app.get(
    "/predict/{ticker}",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(
    ticker: str,
    period: str = Query("6mo", description="Look-back period (1mo/3mo/6mo/1y/2y)"),
    model_path: Optional[str] = Query(
        None,
        description="Path to a saved model (.joblib or .pt). "
                    "Auto-detects best available if omitted.",
    ),
    news_source: str = Query(
        "rss",
        description="News source for sentiment (rss/gnews/newsapi/finnhub)",
    ),
    sentiment_backend: str = Query(
        "vader",
        description="Sentiment backend (vader/finbert)",
    ),
):
    """Predict next-day price direction for a ticker.

    Full pipeline:
    1. Fetch historical price data via yfinance.
    2. Fetch recent news headlines and score sentiment.
    3. Build feature matrix (technical + sentiment).
    4. Load or train a model and predict.
    5. Return direction, confidence, sentiment summary, and metrics.
    """
    ticker = ticker.strip().upper()
    warnings: List[str] = []
    pipeline_status = "ok"

    # ------------------------------------------------------------------
    # 1. Fetch price data
    # ------------------------------------------------------------------
    try:
        from src.data_fetcher import fetch_stock_data

        prices_df = fetch_stock_data(
            ticker=ticker,
            period=period,
            compute_returns=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch price data for {ticker}: {exc}",
        )

    if prices_df.empty or len(prices_df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"Insufficient price data for {ticker} "
                   f"(got {len(prices_df)} rows, need >= 30).",
        )

    current_price = float(prices_df["Close"].iloc[-1])
    date_start = str(prices_df.index.min().date())
    date_end = str(prices_df.index.max().date())

    # ------------------------------------------------------------------
    # 2. Fetch news + sentiment
    # ------------------------------------------------------------------
    sentiment_summary = None
    daily_sentiment = None
    top_headlines: List[Dict[str, Any]] = []

    try:
        from src.news_fetcher import fetch_news

        # Fetch last 30 days of news
        end_date = dt.date.today().isoformat()
        start_date = (dt.date.today() - dt.timedelta(days=30)).isoformat()

        news_df = fetch_news(
            ticker=ticker,
            start=start_date,
            end=end_date,
            source=news_source,
            max_results=50,
        )

        if not news_df.empty:
            from src.sentiment_analyzer import SentimentAnalyzer, aggregate_daily_sentiment

            analyzer = SentimentAnalyzer(backend=sentiment_backend)
            scored = analyzer.score_dataframe(
                news_df,
                text_column="headline",
                date_column="date",
            )

            # Build daily sentiment for feature engineering
            daily_sentiment = aggregate_daily_sentiment(scored)

            # Build summary for response
            mean_score = float(scored["compound"].mean())
            std_score = float(scored["compound"].std()) if len(scored) > 1 else 0.0
            pos_pct = float((scored["label"] == "POSITIVE").mean() * 100)
            neg_pct = float((scored["label"] == "NEGATIVE").mean() * 100)
            neu_pct = float((scored["label"] == "NEUTRAL").mean() * 100)

            # Top headlines by absolute compound score
            scored_sorted = scored.reindex(
                scored["compound"].abs().sort_values(ascending=False).index
            )
            for _, row in scored_sorted.head(5).iterrows():
                top_headlines.append({
                    "headline": str(row.get("headline", "")),
                    "label": str(row.get("label", "")),
                    "compound": round(float(row.get("compound", 0)), 4),
                    "date": str(row.get("date", "")),
                })

            sentiment_summary = SentimentSummary(
                mean_score=round(mean_score, 4),
                std_score=round(std_score, 4),
                positive_pct=round(pos_pct, 1),
                negative_pct=round(neg_pct, 1),
                neutral_pct=round(neu_pct, 1),
                headline_count=len(scored),
                top_headlines=top_headlines,
            )
        else:
            warnings.append("No news headlines found; prediction uses price features only.")

    except Exception as exc:
        warnings.append(f"Sentiment pipeline failed: {exc}")
        pipeline_status = "partial"

    # ------------------------------------------------------------------
    # 3. Build feature matrix
    # ------------------------------------------------------------------
    try:
        from src.feature_engineering import FeatureEngineer

        # Drop return columns that data_fetcher adds (they'd conflict)
        price_cols_to_use = [
            c for c in prices_df.columns
            if c in ("Open", "High", "Low", "Close", "Volume", "Adj Close")
        ]
        clean_prices = prices_df[price_cols_to_use].copy()

        eng = FeatureEngineer()
        features_df = eng.build(clean_prices, sentiment_df=daily_sentiment)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature engineering failed: {exc}",
        )

    n_features = len([
        c for c in features_df.columns if c not in ("target", "target_return")
    ])

    # Extract performance context from features
    recent_vol = None
    recent_trend = None
    current_rsi = None

    if "volatility_20d" in features_df.columns:
        recent_vol = round(float(features_df["volatility_20d"].iloc[-1]), 6)
    if "rsi" in features_df.columns:
        current_rsi = round(float(features_df["rsi"].iloc[-1]), 2)
    if "return_20d" in features_df.columns:
        r20 = float(features_df["return_20d"].iloc[-1])
        if r20 > 0.05:
            recent_trend = "Strong Uptrend"
        elif r20 > 0.01:
            recent_trend = "Mild Uptrend"
        elif r20 > -0.01:
            recent_trend = "Sideways"
        elif r20 > -0.05:
            recent_trend = "Mild Downtrend"
        else:
            recent_trend = "Strong Downtrend"

    # ------------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------------
    direction = "UP"
    confidence = 0.5
    model_type = "none"

    # Strategy: Try loading a saved model, else train a quick XGBoost on the fly
    predict_success = False

    # 4a. Try saved model
    if model_path:
        try:
            saved_path = Path(model_path)
            if saved_path.suffix == ".pt":
                # PyTorch model (deep learning or hybrid)
                meta_path = saved_path.with_suffix(".pt.meta.json")
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    mtype = meta.get("model_type", "")
                    if mtype == "hybrid":
                        from src.hybrid_model import HybridTrainer
                        loaded = HybridTrainer.load(saved_path)
                        proba = loaded.predict_proba(features_df)
                        if len(proba) > 0:
                            confidence = float(proba[-1])
                            direction = "UP" if confidence >= 0.5 else "DOWN"
                            model_type = "hybrid_" + loaded.cell_type
                            predict_success = True
                    else:
                        from src.deep_learning import DeepLearningTrainer
                        loaded = DeepLearningTrainer.load(saved_path)
                        proba = loaded.predict_proba(features_df)
                        if len(proba) > 0:
                            confidence = float(proba[-1])
                            direction = "UP" if confidence >= 0.5 else "DOWN"
                            model_type = loaded.cell_type
                            predict_success = True
            else:
                # sklearn/xgboost model
                from src.model_training import ModelTrainer
                loaded = ModelTrainer.load(saved_path)
                proba = loaded.predict_proba(features_df)
                if len(proba) > 0:
                    confidence = float(proba[-1, 1])
                    direction = "UP" if confidence >= 0.5 else "DOWN"
                    model_type = loaded.model_type
                    predict_success = True
        except Exception as exc:
            warnings.append(f"Failed to load model at {model_path}: {exc}")

    # 4b. Auto-detect saved model in models/ dir
    if not predict_success:
        models_dir = ROOT_DIR / "models"
        candidates = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            try:
                from src.model_training import ModelTrainer
                loaded = ModelTrainer.load(cand)
                proba = loaded.predict_proba(features_df)
                if len(proba) > 0:
                    confidence = float(proba[-1, 1])
                    direction = "UP" if confidence >= 0.5 else "DOWN"
                    model_type = loaded.model_type
                    predict_success = True
                    break
            except Exception:
                continue

    # 4c. Train quick XGBoost on the fly
    if not predict_success:
        try:
            from src.model_training import ModelTrainer

            trainer = ModelTrainer(
                model_type="xgboost",
                val_size=0.15,
                test_size=0.20,
            )
            trainer.train(features_df)
            proba = trainer.predict_proba(features_df)
            if len(proba) > 0:
                confidence = float(proba[-1, 1])
                direction = "UP" if confidence >= 0.5 else "DOWN"
                model_type = "xgboost (auto-trained)"
                predict_success = True
                warnings.append(
                    "No saved model found; trained XGBoost on-the-fly. "
                    "For better results, pre-train and save a model."
                )
        except Exception as exc:
            warnings.append(f"Auto-training failed: {exc}")
            pipeline_status = "partial"
            # Fall back to a simple momentum heuristic
            if "return_5d" in features_df.columns:
                r5 = float(features_df["return_5d"].iloc[-1])
                direction = "UP" if r5 > 0 else "DOWN"
                confidence = min(abs(r5) * 10, 0.95)
                model_type = "momentum_heuristic"
                warnings.append("Using momentum heuristic as fallback.")

    # Clip confidence
    confidence = round(max(0.0, min(1.0, confidence)), 4)
    # For DOWN predictions, express confidence as 1 - raw_proba
    if direction == "DOWN":
        confidence = round(1.0 - confidence, 4) if confidence < 0.5 else confidence

    prediction_date = str(dt.date.today())

    return PredictionResponse(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        model_type=model_type,
        features_used=n_features,
        current_price=round(current_price, 2),
        prediction_date=prediction_date,
        sentiment=sentiment_summary,
        performance=PerformanceMetrics(
            data_points=len(prices_df),
            feature_count=n_features,
            date_range_start=date_start,
            date_range_end=date_end,
            recent_volatility=recent_vol,
            recent_trend=recent_trend,
            current_rsi=current_rsi,
        ),
        pipeline_status=pipeline_status,
        warnings=warnings,
    )


@app.post(
    "/sentiment",
    response_model=SentimentResponse,
    tags=["Sentiment"],
)
async def score_sentiment(request: SentimentRequest):
    """Score a batch of texts for sentiment.

    Supports ``"vader"`` (fast, no download) and ``"finbert"``
    (transformer-based, more accurate for financial text).
    """
    try:
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer(backend=request.backend)
        results = analyzer.score_texts(request.texts)
        return SentimentResponse(
            results=results,
            backend_used=analyzer.backend_name,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Sentiment module not available: {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/backtest/{ticker}",
    response_model=BacktestResponse,
    tags=["Backtest"],
)
async def backtest(
    ticker: str,
    period: str = Query("1y", description="Historical look-back"),
    commission: float = Query(0.001, description="One-way commission rate"),
    initial_cash: float = Query(100_000, description="Starting capital"),
    model_type: str = Query(
        "xgboost",
        description="Model to train for signals (xgboost/random_forest/logistic_regression)",
    ),
):
    """Train a model on historical data, generate signals, and backtest.

    Returns detailed performance metrics including comparison against
    buy-and-hold.
    """
    ticker = ticker.strip().upper()

    # 1. Fetch data
    try:
        from src.data_fetcher import fetch_stock_data

        prices_df = fetch_stock_data(ticker=ticker, period=period)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data for {ticker}: {exc}",
        )

    if len(prices_df) < 60:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {ticker}: {len(prices_df)} bars (need >= 60).",
        )

    # 2. Build features
    try:
        from src.feature_engineering import FeatureEngineer

        price_cols = [c for c in prices_df.columns
                      if c in ("Open", "High", "Low", "Close", "Volume", "Adj Close")]
        eng = FeatureEngineer()
        features_df = eng.build(prices_df[price_cols])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature engineering failed: {exc}",
        )

    # 3. Train model and generate predictions
    try:
        from src.model_training import ModelTrainer

        trainer = ModelTrainer(model_type=model_type)
        trainer.train(features_df)

        # Generate signals on test set
        test_preds = trainer.predict(trainer.X_test)
        test_prices = prices_df["Close"].reindex(trainer.X_test.index)
        test_signals = pd.Series(test_preds, index=trainer.X_test.index)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {exc}",
        )

    # 4. Backtest
    try:
        from src.backtesting import SimpleBacktester

        bt = SimpleBacktester(
            initial_cash=initial_cash,
            commission_pct=commission,
        )
        result = bt.run(test_prices, test_signals)
        baselines = bt.compare_baselines(include_random=False)

        bah_return = None
        outperformance = None
        if "Buy-and-Hold" in baselines:
            bah_return = round(baselines["Buy-and-Hold"].total_return_pct, 4)
            outperformance = round(
                result.total_return_pct - bah_return, 4
            )

        return BacktestResponse(
            ticker=ticker,
            total_return_pct=round(result.total_return_pct, 4),
            annual_return_pct=round(result.annual_return_pct, 4),
            sharpe_ratio=round(result.sharpe_ratio, 4),
            sortino_ratio=round(result.sortino_ratio, 4),
            max_drawdown_pct=round(result.max_drawdown_pct, 4),
            hit_rate_pct=round(result.hit_rate_pct, 2),
            total_trades=result.total_trades,
            win_rate_pct=round(result.win_rate_pct, 1),
            profit_factor=round(result.profit_factor, 2),
            exposure_pct=round(result.exposure_pct, 1),
            buy_and_hold_return_pct=bah_return,
            outperformance_pct=outperformance,
            data_points=len(test_prices),
            period=period,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Backtesting failed: {exc}",
        )


@app.get("/models", response_model=ModelsResponse, tags=["System"])
async def list_models():
    """List all saved models in the models/ directory."""
    models_dir = ROOT_DIR / "models"
    infos: List[ModelInfo] = []

    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            if f.suffix in (".joblib", ".pt") and not f.name.endswith(".meta.json"):
                info = ModelInfo(
                    filename=f.name,
                    file_size_kb=round(f.stat().st_size / 1024, 1),
                )

                # Try reading metadata
                meta_path = f.with_suffix(f.suffix + ".meta.json")
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        info.model_type = meta.get("model_type")
                        info.n_features = meta.get("n_features")
                        test_m = meta.get("test_metrics", {})
                        info.test_accuracy = test_m.get("accuracy")
                        info.test_f1 = test_m.get("f1")
                    except Exception:
                        pass

                infos.append(info)

    return ModelsResponse(models=infos, count=len(infos))


# ===================================================================
#  Main
# ===================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
