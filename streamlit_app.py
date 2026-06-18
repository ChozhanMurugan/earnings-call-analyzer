"""Thin Streamlit UI on top of the FastAPI service.

Run:
    uvicorn eca.api.main:app --reload      # in one shell
    streamlit run streamlit_app.py          # in another
"""
from __future__ import annotations

import os
from datetime import date

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.getenv("ECA_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")
st.title("Earnings Call Analyzer")
st.caption("FinBERT + hedging + forward-guidance features → directional classifier → backtest")

tab_analyze, tab_backtest = st.tabs(["Analyze a call", "Backtest"])

# ---------- Analyze ----------
with tab_analyze:
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.text_input("Ticker", value="AAPL").upper()
        call_date = st.date_input("Call date", value=date.today())
        fq = st.text_input("Fiscal quarter (optional)", value="")
    with col2:
        prepared = st.text_area(
            "Prepared remarks",
            height=180,
            placeholder="Paste the prepared-remarks section of the transcript…",
        )
        qa = st.text_area("Q&A section (optional)", height=120)

    if st.button("Analyze", type="primary"):
        if len(prepared) < 200:
            st.error("Prepared remarks must be at least 200 characters.")
        else:
            with st.spinner("Running FinBERT + feature extraction…"):
                r = httpx.post(
                    f"{API_BASE}/analyze",
                    json={
                        "ticker": ticker,
                        "call_date": call_date.isoformat(),
                        "fiscal_quarter": fq or None,
                        "prepared_remarks": prepared,
                        "qa_section": qa,
                    },
                    timeout=120,
                )
            if r.status_code != 200:
                st.error(f"{r.status_code}: {r.text}")
            else:
                data = r.json()
                pred = data.get("prediction")
                if pred:
                    cA, cB, cC = st.columns(3)
                    cA.metric("P(up)", f"{pred['prob_up']:.3f}")
                    cB.metric("Direction", "UP" if pred["direction"] == 1 else "DOWN")
                    cC.metric("Confidence", f"{pred['confidence']:.2f}")
                elif data.get("note"):
                    st.info(data["note"])
                feats = data["features"]
                st.subheader("Features")
                fdf = pd.DataFrame({"feature": list(feats.keys()), "value": list(feats.values())})
                st.dataframe(fdf, use_container_width=True, height=420)

# ---------- Backtest ----------
with tab_backtest:
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker_bt = st.text_input("Ticker (blank = all)", value="").upper()
        threshold = st.slider("Probability threshold (|p - 0.5|)", 0.0, 0.45, 0.0, 0.05)
        go = st.button("Run backtest", type="primary")

    if go:
        url = f"{API_BASE}/backtest/{ticker_bt}" if ticker_bt else f"{API_BASE}/backtest"
        with st.spinner("Computing…"):
            r = httpx.get(url, params={"threshold": threshold}, timeout=60)
        if r.status_code != 200:
            st.error(f"{r.status_code}: {r.text}")
        else:
            data = r.json()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", data["n_trades"])
            m2.metric("Hit-rate", f"{data['hit_rate']:.1%}" if data["hit_rate"] == data["hit_rate"] else "n/a")
            m3.metric("Ann. Sharpe", f"{data['annualised_sharpe']:.2f}" if data["annualised_sharpe"] == data["annualised_sharpe"] else "n/a")
            m4.metric("Max DD", f"{data['max_drawdown']:.1%}")

            curve = pd.DataFrame(data["equity_curve"])
            if not curve.empty:
                curve["date"] = pd.to_datetime(curve["date"])
                long = curve.melt(id_vars="date", value_vars=["strategy_equity", "benchmark_equity"],
                                  var_name="series", value_name="equity")
                fig = px.line(long, x="date", y="equity", color="series",
                              title="Equity curve: strategy vs benchmark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trades fired at this threshold.")
