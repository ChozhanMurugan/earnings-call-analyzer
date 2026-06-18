---
title: Earnings Call Analyzer
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Earnings Call Analyzer

NLP-driven directional classifier on S&P 500 earnings-call transcripts, served as a FastAPI + Streamlit app.

- Paste a transcript → get FinBERT sentiment, hedge-word density, forward-guidance score
- Long/short signal based on LightGBM probability (walk-forward CV trained on 33k calls)
- Backtest endpoint returns hit-rate, Sharpe, and max drawdown vs SPY

Source code: https://github.com/ChozhanMurugan/earnings-call-analyzer
