# 📈 Stock Sentiment Predictor

A modular, production-ready pipeline for predicting stock price movements using sentiment analysis from news and social media sources.

---

## 🏗️ Project Structure

```
stock-sentiment-predictor/
├── app/                    # API server & dashboard
│   ├── api.py              # FastAPI REST endpoints
│   └── dashboard.py        # Streamlit/Dash interactive dashboard
├── config/                 # Configuration files
│   └── config.yaml         # Centralized project settings
├── data/                   # Data storage (gitignored)
│   ├── raw/                # Unprocessed source data
│   ├── processed/          # Cleaned & transformed data
│   └── external/           # Third-party reference datasets
├── models/                 # Serialized trained models
├── notebooks/              # Jupyter exploration notebooks
├── reports/                # Generated analysis & figures
├── src/                    # Core library modules
│   ├── data_fetcher.py     # Market data & news ingestion
│   ├── sentiment_analyzer.py  # NLP sentiment scoring
│   ├── feature_engineering.py # Feature construction
│   ├── model_training.py   # ML model training & evaluation
│   ├── deep_learning.py    # LSTM/GRU deep learning models
│   ├── hybrid_model.py     # Two-branch hybrid (price RNN + sentiment encoder)
│   └── backtesting.py      # Strategy backtesting engine
├── tests/                  # Unit & integration tests
├── requirements.txt        # Python dependencies
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/stock-sentiment-predictor.git
cd stock-sentiment-predictor

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config/config.yaml` to add your API keys and adjust parameters:

```yaml
api_keys:
  alpha_vantage: "YOUR_KEY"
  news_api: "YOUR_KEY"
```

## 📦 Modules

| Module | Description |
|---|---|
| `src/data_fetcher.py` | Pulls historical prices, news articles, and social media posts |
| `src/sentiment_analyzer.py` | Scores text sentiment using transformer-based NLP models |
| `src/feature_engineering.py` | Constructs technical indicators and sentiment features |
| `src/model_training.py` | Trains and evaluates baseline ML models (XGBoost, RF, LR) |
| `src/deep_learning.py` | LSTM/GRU recurrent models with sequence-based training |
| `src/hybrid_model.py` | Two-branch fusion: price RNN + sentiment encoder |
| `src/backtesting.py` | Simulates trading strategies on historical data |
| `app/api.py` | Exposes predictions via a RESTful API |
| `app/dashboard.py` | Interactive visualization dashboard |

## 🧪 Testing

```bash
pytest tests/ -v
```

## 🗺️ Roadmap

- [ ] Integrate real-time streaming data (WebSockets)
- [ ] Add Reddit / Twitter sentiment sources
- [ ] Implement ensemble model strategies
- [ ] Deploy via Docker + cloud CI/CD
- [ ] Add MLflow experiment tracking

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
