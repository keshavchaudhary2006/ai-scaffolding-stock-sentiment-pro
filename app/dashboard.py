"""
dashboard.py
============
Interactive Streamlit dashboard for the Stock Sentiment Predictor.

Sections
--------
1. **Price Chart** -- Candlestick + volume + moving averages
2. **Sentiment** -- Daily sentiment trend, distribution, top headlines
3. **Prediction** -- Model prediction card with confidence gauge
4. **Model Results** -- Classification metrics, confusion matrix, split comparison
5. **Backtest** -- Equity curve, drawdown, baseline comparison, trade log
6. **Feature Importance** -- Top features bar chart

Run
---
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is on sys.path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# =====================================================================
#  Page Config
# =====================================================================

st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
#  Custom CSS
# =====================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100, 180, 255, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stMetricLabel"] {
        color: #8899aa !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        color: #e0e6ed !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a1a2e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.35);
    }

    /* Header gradient */
    h1 {
        background: linear-gradient(90deg, #0ea5e9, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Status badges */
    .pred-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(100, 180, 255, 0.2);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    .pred-up {
        border-color: rgba(34, 197, 94, 0.5);
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1);
    }
    .pred-down {
        border-color: rgba(239, 68, 68, 0.5);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.1);
    }
    .pred-direction {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin: 8px 0;
    }
    .pred-direction-up { color: #22c55e; }
    .pred-direction-down { color: #ef4444; }
    .pred-label {
        font-size: 0.85rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .pred-conf {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e0e6ed;
    }
    .sent-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .sent-pos { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .sent-neg { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .sent-neu { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(100, 180, 255, 0.1);
        border-radius: 10px;
    }

    .footer-text {
        text-align: center;
        color: #556677;
        font-size: 0.8rem;
        padding: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
#  Plotly Theme
# =====================================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c0ccd8"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(100,180,255,0.07)"),
    yaxis=dict(gridcolor="rgba(100,180,255,0.07)"),
)


# =====================================================================
#  Sidebar
# =====================================================================

st.sidebar.markdown("## Stock Sentiment Predictor")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").strip().upper()
period = st.sidebar.selectbox(
    "Historical Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2
)

st.sidebar.markdown("#### Model Settings")
model_type = st.sidebar.selectbox(
    "ML Model",
    ["XGBoost", "Random Forest", "Logistic Regression"],
    index=0,
)
MODEL_MAP = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
}

st.sidebar.markdown("#### Sentiment Settings")
news_source = st.sidebar.selectbox(
    "News Source",
    ["RSS (no key)", "GNews", "NewsAPI", "Finnhub"],
    index=0,
)
NEWS_MAP = {
    "RSS (no key)": "rss",
    "GNews": "gnews",
    "NewsAPI": "newsapi",
    "Finnhub": "finnhub",
}
sentiment_backend = st.sidebar.selectbox(
    "Sentiment Model", ["VADER (fast)", "FinBERT (accurate)"], index=0,
)
SENT_MAP = {"VADER (fast)": "vader", "FinBERT (accurate)": "finbert"}

st.sidebar.markdown("#### Backtest Settings")
initial_cash = st.sidebar.number_input("Initial Cash ($)", value=100_000, step=10_000)
commission = st.sidebar.number_input(
    "Commission (bps)", value=10, step=5, help="One-way, in basis points"
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Full Pipeline", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly")


# =====================================================================
#  Session State Initialization
# =====================================================================

STATE_KEYS = [
    "prices_df", "features_df", "daily_sentiment", "scored_headlines",
    "trainer", "metrics", "backtest_result", "baselines",
    "prediction", "confidence", "avg_sentiment", "pipeline_ticker",
]
for key in STATE_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None


# =====================================================================
#  Header
# =====================================================================

st.title("Stock Sentiment Predictor")
st.markdown(
    "Combine **NLP sentiment analysis** with **technical indicators** "
    "to predict short-term stock price direction."
)

# Top-level metric cards
c1, c2, c3, c4, c5 = st.columns(5)

pred_val = st.session_state.get("prediction")
conf_val = st.session_state.get("confidence")
avg_sent = st.session_state.get("avg_sentiment")
model_acc = (
    st.session_state["metrics"].get("accuracy")
    if st.session_state.get("metrics")
    else None
)
pticker = st.session_state.get("pipeline_ticker") or "--"

c1.metric("Ticker", pticker)
c2.metric("Prediction", pred_val or "--")
c3.metric("Confidence", f"{conf_val:.1%}" if conf_val else "--")
c4.metric("Model Accuracy", f"{model_acc:.1%}" if model_acc else "--")
c5.metric("Avg Sentiment", f"{avg_sent:+.3f}" if avg_sent is not None else "--")

st.markdown("---")


# =====================================================================
#  Tabs
# =====================================================================

tab_price, tab_sentiment, tab_predict, tab_model, tab_backtest, tab_features = st.tabs(
    ["Price Chart", "Sentiment", "Prediction", "Model Results", "Backtest", "Features"]
)


# ----- Tab: Price Chart -----------------------------------------------

with tab_price:
    if st.session_state.get("prices_df") is not None:
        prices = st.session_state["prices_df"]
        st.subheader(f"{pticker} -- Price History ({len(prices)} bars)")

        # Candlestick + MAs
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=prices.index,
            open=prices["Open"], high=prices["High"],
            low=prices["Low"], close=prices["Close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ))

        for col, color, label in [
            ("sma_20", "#60a5fa", "SMA 20"),
            ("sma_50", "#a78bfa", "SMA 50"),
        ]:
            if col in prices.columns:
                fig.add_trace(go.Scatter(
                    x=prices.index, y=prices[col],
                    name=label, line=dict(color=color, width=1.5), opacity=0.7,
                ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{pticker} Price",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume
        vol_colors = [
            "#22c55e" if c >= o else "#ef4444"
            for c, o in zip(prices["Close"], prices["Open"])
        ]
        fig_vol = go.Figure(go.Bar(
            x=prices.index, y=prices["Volume"],
            marker_color=vol_colors, opacity=0.6, name="Volume",
        ))
        fig_vol.update_layout(
            **PLOTLY_LAYOUT, title="Volume", height=200, yaxis_title="Volume",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # Price statistics
        with st.expander("Price Statistics"):
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Current", f"${prices['Close'].iloc[-1]:.2f}")
            hi = prices["High"].max()
            lo = prices["Low"].min()
            sc2.metric("Period High", f"${hi:.2f}")
            sc3.metric("Period Low", f"${lo:.2f}")
            if "daily_return" in prices.columns:
                vol = prices["daily_return"].std() * np.sqrt(252) * 100
                sc4.metric("Ann. Volatility", f"{vol:.1f}%")
    else:
        st.info("Click **Run Full Pipeline** in the sidebar to load data.")


# ----- Tab: Sentiment --------------------------------------------------

with tab_sentiment:
    daily_sent = st.session_state.get("daily_sentiment")
    scored_hl = st.session_state.get("scored_headlines")

    if daily_sent is not None and not daily_sent.empty:
        st.subheader("Sentiment Analysis")

        # Summary badges
        sc1, sc2, sc3, sc4 = st.columns(4)
        mean_s = daily_sent["sent_mean"].mean()
        badge_cls = "sent-pos" if mean_s > 0.05 else ("sent-neg" if mean_s < -0.05 else "sent-neu")
        badge_txt = "Bullish" if mean_s > 0.05 else ("Bearish" if mean_s < -0.05 else "Neutral")
        sc1.markdown(
            f'<div class="pred-card" style="padding:16px">'
            f'<div class="pred-label">Overall Sentiment</div>'
            f'<span class="sent-badge {badge_cls}">{badge_txt}</span>'
            f'<div style="margin-top:8px;color:#e0e6ed;font-size:1.2rem">{mean_s:+.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if "sent_count" in daily_sent.columns:
            sc2.metric("Total Headlines", int(daily_sent["sent_count"].sum()))
        sc3.metric("Days Covered", len(daily_sent))
        if "sent_positive_pct" in daily_sent.columns:
            pos_avg = daily_sent["sent_positive_pct"].mean() * 100
            sc4.metric("Avg Positive %", f"{pos_avg:.1f}%")

        # Daily sentiment trend
        fig_sent = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            subplot_titles=["Daily Mean Sentiment", "Headline Count"],
            vertical_spacing=0.08,
        )

        # Sentiment line
        colors_sent = [
            "#22c55e" if v >= 0 else "#ef4444"
            for v in daily_sent["sent_mean"]
        ]
        fig_sent.add_trace(go.Bar(
            x=daily_sent.index, y=daily_sent["sent_mean"],
            marker_color=colors_sent, opacity=0.7, name="Sentiment",
        ), row=1, col=1)
        fig_sent.add_hline(y=0, line_dash="dash", line_color="#555", row=1, col=1)

        # Std band
        if "sent_std" in daily_sent.columns:
            fig_sent.add_trace(go.Scatter(
                x=daily_sent.index,
                y=daily_sent["sent_mean"] + daily_sent["sent_std"],
                mode="lines", line=dict(width=0), showlegend=False,
            ), row=1, col=1)
            fig_sent.add_trace(go.Scatter(
                x=daily_sent.index,
                y=daily_sent["sent_mean"] - daily_sent["sent_std"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(99,102,241,0.1)",
                showlegend=False,
            ), row=1, col=1)

        # Headline count
        if "sent_count" in daily_sent.columns:
            fig_sent.add_trace(go.Bar(
                x=daily_sent.index, y=daily_sent["sent_count"],
                marker_color="#6366f1", opacity=0.5, name="Count",
            ), row=2, col=1)

        fig_sent.update_layout(
            **PLOTLY_LAYOUT, height=500, showlegend=False,
            title_text="",
        )
        fig_sent.update_yaxes(title_text="Compound Score", row=1, col=1)
        fig_sent.update_yaxes(title_text="Count", row=2, col=1)
        st.plotly_chart(fig_sent, use_container_width=True)

        # Sentiment distribution
        if "sent_positive_pct" in daily_sent.columns:
            st.markdown("#### Sentiment Distribution (avg across days)")
            pos_m = daily_sent["sent_positive_pct"].mean()
            neg_m = daily_sent["sent_negative_pct"].mean()
            neu_m = daily_sent["sent_neutral_pct"].mean()

            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[pos_m, neg_m, neu_m],
                marker_colors=["#22c55e", "#ef4444", "#facc15"],
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(size=13),
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT, height=350,
                showlegend=False,
                annotations=[dict(
                    text=f"{mean_s:+.3f}", x=0.5, y=0.5,
                    font_size=22, font_color="#e0e6ed", showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Top headlines table
        if scored_hl is not None and not scored_hl.empty:
            st.markdown("#### Top Headlines by Impact")
            display_hl = scored_hl.copy()
            display_hl["abs_compound"] = display_hl["compound"].abs()
            display_hl = display_hl.sort_values("abs_compound", ascending=False).head(15)
            show_cols = ["date", "headline", "label", "compound"]
            show_cols = [c for c in show_cols if c in display_hl.columns]
            st.dataframe(
                display_hl[show_cols].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("Click **Run Full Pipeline** to fetch news and analyze sentiment.")


# ----- Tab: Prediction -------------------------------------------------

with tab_predict:
    if st.session_state.get("prediction") is not None:
        pred = st.session_state["prediction"]
        conf = st.session_state["confidence"]
        prices = st.session_state.get("prices_df")

        st.subheader("Model Prediction")

        pc1, pc2, pc3 = st.columns([1, 1.2, 1])

        # Prediction card
        dir_cls = "pred-up" if pred == "UP" else "pred-down"
        dir_color = "pred-direction-up" if pred == "UP" else "pred-direction-down"
        arrow = "&#9650;" if pred == "UP" else "&#9660;"

        with pc1:
            st.markdown(
                f'<div class="pred-card {dir_cls}">'
                f'<div class="pred-label">Next-Day Prediction</div>'
                f'<div class="pred-direction {dir_color}">{arrow} {pred}</div>'
                f'<div class="pred-label" style="margin-top:12px">Confidence</div>'
                f'<div class="pred-conf">{conf:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Confidence gauge
        with pc2:
            gauge_color = "#22c55e" if pred == "UP" else "#ef4444"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                number=dict(suffix="%", font=dict(size=36, color="#e0e6ed")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#8899aa")),
                    bar=dict(color=gauge_color),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40], color="rgba(239,68,68,0.15)"),
                        dict(range=[40, 60], color="rgba(250,204,21,0.15)"),
                        dict(range=[60, 100], color="rgba(34,197,94,0.15)"),
                    ],
                    threshold=dict(
                        line=dict(color="#e0e6ed", width=2),
                        thickness=0.8, value=conf * 100,
                    ),
                ),
                title=dict(text="Confidence", font=dict(size=16, color="#8899aa")),
            ))
            gauge_layout = {
                k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"
            }
            fig_gauge.update_layout(
                **gauge_layout, height=280,
                margin=dict(l=30, r=30, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Context card
        with pc3:
            if prices is not None:
                cur_price = prices["Close"].iloc[-1]
                prev_price = prices["Close"].iloc[-2] if len(prices) > 1 else cur_price
                day_chg = (cur_price - prev_price) / prev_price * 100

                avg_s = st.session_state.get("avg_sentiment")
                sent_text = f"{avg_s:+.4f}" if avg_s is not None else "N/A"
                sent_label = (
                    "Bullish" if (avg_s or 0) > 0.05
                    else "Bearish" if (avg_s or 0) < -0.05
                    else "Neutral"
                )

                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="pred-label">Current Price</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:#e0e6ed">'
                    f'${cur_price:.2f}</div>'
                    f'<div style="font-size:0.9rem;color:{"#22c55e" if day_chg>=0 else "#ef4444"}">'
                    f'{"+" if day_chg>=0 else ""}{day_chg:.2f}% today</div>'
                    f'<div class="pred-label" style="margin-top:16px">Sentiment</div>'
                    f'<div style="font-size:1.1rem;color:#e0e6ed">'
                    f'{sent_text} ({sent_label})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Disclaimer
        st.caption(
            "This prediction is for educational purposes only and does not "
            "constitute financial advice. Past performance does not guarantee future results."
        )
    else:
        st.info("Click **Run Full Pipeline** to generate a prediction.")


# ----- Tab: Model Results -----------------------------------------------

with tab_model:
    if st.session_state.get("metrics") is not None:
        metrics = st.session_state["metrics"]
        trainer = st.session_state["trainer"]

        st.subheader("Model Performance")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("Precision", f"{metrics['precision']:.4f}")
        m3.metric("Recall", f"{metrics['recall']:.4f}")
        m4.metric("F1 Score", f"{metrics['f1']:.4f}")
        m5.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else "N/A",
        )

        mc1, mc2 = st.columns([1, 1.2])

        # Confusion matrix
        with mc1:
            st.markdown("#### Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Pred DOWN", "Pred UP"],
                y=["Actual DOWN", "Actual UP"],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=20, color="white"),
                colorscale=[[0, "#1e293b"], [1, "#6366f1"]],
                showscale=False,
            ))
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_cm, use_container_width=True)

        # Split comparison
        with mc2:
            if trainer and trainer.train_metrics:
                st.markdown("#### Train / Val / Test")
                split_data = {
                    "Split": ["Train", "Validation", "Test"],
                    "Accuracy": [
                        trainer.train_metrics.accuracy,
                        trainer.val_metrics.accuracy,
                        trainer.test_metrics.accuracy,
                    ],
                    "F1": [
                        trainer.train_metrics.f1,
                        trainer.val_metrics.f1,
                        trainer.test_metrics.f1,
                    ],
                    "ROC-AUC": [
                        trainer.train_metrics.roc_auc or 0,
                        trainer.val_metrics.roc_auc or 0,
                        trainer.test_metrics.roc_auc or 0,
                    ],
                }
                fig_sp = go.Figure()
                bar_colors = ["#60a5fa", "#a78bfa", "#f472b6"]
                for i, mn in enumerate(["Accuracy", "F1", "ROC-AUC"]):
                    fig_sp.add_trace(go.Bar(
                        name=mn, x=split_data["Split"], y=split_data[mn],
                        marker_color=bar_colors[i],
                        text=[f"{v:.3f}" for v in split_data[mn]],
                        textposition="outside",
                    ))
                fig_sp.update_layout(
                    **PLOTLY_LAYOUT, barmode="group",
                    height=320, yaxis_range=[0, 1.15],
                )
                st.plotly_chart(fig_sp, use_container_width=True)

        with st.expander("Full Classification Report"):
            st.code(metrics["report"])
    else:
        st.info("Click **Run Full Pipeline** to train a model.")


# ----- Tab: Backtest ---------------------------------------------------

with tab_backtest:
    bt_result = st.session_state.get("backtest_result")
    if bt_result is not None:
        st.subheader("Backtest Results")

        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Total Return", f"{bt_result.total_return_pct:+.2f}%")
        b2.metric("Sharpe", f"{bt_result.sharpe_ratio:.3f}")
        b3.metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.2f}%")
        b4.metric("Hit Rate", f"{bt_result.hit_rate_pct:.1f}%")
        b5.metric("Win Rate", f"{bt_result.win_rate_pct:.1f}%")
        b6.metric("Trades", bt_result.total_trades)

        # Equity curve with baseline
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=bt_result.equity_curve.index,
            y=bt_result.equity_curve.values,
            name="Model Strategy",
            line=dict(color="#6366f1", width=2.5),
        ))

        baselines = st.session_state.get("baselines") or {}
        baseline_colors = {
            "Buy-and-Hold": "#22c55e",
            "Random": "#facc15",
        }
        for bname, bresult in baselines.items():
            if bname in baseline_colors and not bresult.equity_curve.empty:
                fig_eq.add_trace(go.Scatter(
                    x=bresult.equity_curve.index,
                    y=bresult.equity_curve.values,
                    name=bname,
                    line=dict(color=baseline_colors[bname], width=1.5, dash="dash"),
                    opacity=0.7,
                ))

        fig_eq.update_layout(
            **PLOTLY_LAYOUT,
            title="Equity Curve vs Baselines",
            yaxis_title="Portfolio Value ($)",
            height=420, legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown
        if not bt_result.drawdown_series.empty:
            fig_dd = go.Figure(go.Scatter(
                x=bt_result.drawdown_series.index,
                y=bt_result.drawdown_series.values * 100,
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.12)",
                line=dict(color="#ef4444", width=1.5),
                name="Drawdown",
            ))
            fig_dd.update_layout(
                **PLOTLY_LAYOUT, title="Drawdown",
                yaxis_title="Drawdown (%)", height=220,
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # Baseline comparison table
        if baselines:
            st.markdown("#### Strategy Comparison")
            rows = []
            all_strats = {"Model": bt_result}
            all_strats.update(baselines)
            for sname, sres in all_strats.items():
                rows.append({
                    "Strategy": sname,
                    "Return": f"{sres.total_return_pct:+.2f}%",
                    "Annual": f"{sres.annual_return_pct:+.2f}%",
                    "Sharpe": f"{sres.sharpe_ratio:.3f}",
                    "Max DD": f"{sres.max_drawdown_pct:.2f}%",
                    "Hit Rate": f"{sres.hit_rate_pct:.1f}%",
                    "Trades": sres.total_trades,
                })
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True,
            )

        # Trade log
        if bt_result.trades:
            with st.expander(f"Trade Log (last {min(20, len(bt_result.trades))})"):
                trade_rows = [
                    {
                        "Entry": t.entry_date[:10],
                        "Exit": t.exit_date[:10],
                        "Entry$": f"${t.entry_price:.2f}",
                        "Exit$": f"${t.exit_price:.2f}",
                        "P&L": f"${t.pnl:.2f}",
                        "Return": f"{t.return_pct:+.2f}%",
                        "Days": t.holding_days,
                    }
                    for t in bt_result.trades[-20:]
                ]
                st.dataframe(
                    pd.DataFrame(trade_rows),
                    use_container_width=True, hide_index=True,
                )
    else:
        st.info("Click **Run Full Pipeline** to run a backtest simulation.")


# ----- Tab: Feature Importance -----------------------------------------

with tab_features:
    if st.session_state.get("trainer") is not None:
        trainer = st.session_state["trainer"]
        try:
            fi = trainer.feature_importance(top_n=25)
            st.subheader("Top 25 Feature Importances")

            fig_fi = go.Figure(go.Bar(
                y=fi["feature"][::-1],
                x=fi["importance"][::-1],
                orientation="h",
                marker=dict(
                    color=fi["importance"][::-1],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#6366f1"], [1, "#a855f7"]],
                ),
                text=[f"{v:.4f}" for v in fi["importance"][::-1]],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig_fi.update_layout(
                **PLOTLY_LAYOUT, height=700,
                title="Feature Importance",
                xaxis_title="Importance",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            with st.expander("Feature Importance Table"):
                st.dataframe(fi, use_container_width=True, hide_index=True)
        except RuntimeError:
            st.warning("Feature importance not available for this model.")
    else:
        st.info("Click **Run Full Pipeline** to view features.")


# =====================================================================
#  Pipeline Execution
# =====================================================================

if run_button:
    progress = st.progress(0, text="Starting pipeline...")

    try:
        # ---- Step 1: Fetch prices ----------------------------------------
        progress.progress(5, text=f"Fetching {ticker} price data...")
        from src.data_fetcher import fetch_stock_data

        prices_df = fetch_stock_data(
            ticker=ticker, period=period, compute_returns=True,
        )
        st.session_state["prices_df"] = prices_df
        st.session_state["pipeline_ticker"] = ticker

        # ---- Step 2: Fetch news + sentiment ------------------------------
        progress.progress(20, text="Fetching news headlines...")
        daily_sentiment = None
        scored_headlines = None

        try:
            import datetime as dt
            from src.news_fetcher import fetch_news

            end_dt = dt.date.today().isoformat()
            start_dt = (dt.date.today() - dt.timedelta(days=30)).isoformat()

            news_df = fetch_news(
                ticker=ticker,
                start=start_dt, end=end_dt,
                source=NEWS_MAP[news_source],
                max_results=50,
            )

            if not news_df.empty:
                progress.progress(30, text="Scoring sentiment...")
                from src.sentiment_analyzer import (
                    SentimentAnalyzer, aggregate_daily_sentiment,
                )

                analyzer = SentimentAnalyzer(
                    backend=SENT_MAP[sentiment_backend],
                )
                scored_headlines = analyzer.score_dataframe(news_df)
                daily_sentiment = aggregate_daily_sentiment(scored_headlines)

                st.session_state["daily_sentiment"] = daily_sentiment
                st.session_state["scored_headlines"] = scored_headlines
                st.session_state["avg_sentiment"] = float(
                    daily_sentiment["sent_mean"].mean()
                )
            else:
                st.session_state["avg_sentiment"] = 0.0

        except Exception as exc:
            st.warning(f"Sentiment pipeline: {exc}")
            st.session_state["avg_sentiment"] = None

        # ---- Step 3: Build features --------------------------------------
        progress.progress(40, text="Engineering features...")
        from src.feature_engineering import FeatureEngineer

        price_cols = [
            c for c in prices_df.columns
            if c in ("Open", "High", "Low", "Close", "Volume", "Adj Close")
        ]
        eng = FeatureEngineer()
        features_df = eng.build(
            prices_df[price_cols],
            sentiment_df=daily_sentiment,
        )
        st.session_state["features_df"] = features_df

        # Push SMA columns back for charting
        for col in ["sma_20", "sma_50"]:
            if col in features_df.columns:
                prices_df.loc[features_df.index, col] = features_df[col]
        st.session_state["prices_df"] = prices_df

        # ---- Step 4: Train model -----------------------------------------
        progress.progress(55, text=f"Training {model_type}...")
        from src.model_training import ModelTrainer

        model_key = MODEL_MAP[model_type]
        trainer = ModelTrainer(model_type=model_key)
        trainer.train(features_df)
        metrics = trainer.evaluate()

        st.session_state["trainer"] = trainer
        st.session_state["metrics"] = metrics

        # ---- Step 5: Generate prediction ---------------------------------
        progress.progress(70, text="Generating prediction...")

        last_row = features_df.iloc[[-1]]
        proba = trainer.predict_proba(last_row)[0]
        pred_class = int(proba[1] > 0.5)

        st.session_state["prediction"] = "UP" if pred_class == 1 else "DOWN"
        st.session_state["confidence"] = float(max(proba))

        # ---- Step 6: Backtest --------------------------------------------
        progress.progress(85, text="Running backtest...")
        from src.backtesting import SimpleBacktester

        test_features = features_df.iloc[int(len(features_df) * 0.8):]
        predictions = trainer.predict(test_features)
        signals = pd.Series(
            predictions, index=test_features.index, name="signal",
        )
        test_prices = prices_df.loc[test_features.index, "Close"]

        bt = SimpleBacktester(
            initial_cash=initial_cash,
            commission_pct=commission / 10_000,  # bps to fraction
        )
        backtest_result = bt.run(test_prices, signals)
        baselines = bt.compare_baselines(include_random=True)

        st.session_state["backtest_result"] = backtest_result
        st.session_state["baselines"] = baselines

        # ---- Done --------------------------------------------------------
        progress.progress(100, text="Pipeline complete!")
        st.success(
            f"Pipeline completed for **{ticker}** using **{model_type}**. "
            f"Processed {len(prices_df)} bars, "
            f"{len(features_df)} feature rows, "
            f"{len(scored_headlines) if scored_headlines is not None else 0} headlines."
        )
        st.rerun()

    except Exception as exc:
        progress.empty()
        st.error(f"Pipeline failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


# =====================================================================
#  Footer
# =====================================================================

st.markdown("---")
st.markdown(
    '<p class="footer-text">Stock Sentiment Predictor v0.2.0 '
    '-- Built with Streamlit, Plotly, scikit-learn, PyTorch</p>',
    unsafe_allow_html=True,
)
