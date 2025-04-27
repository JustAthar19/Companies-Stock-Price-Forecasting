# 📈 Stock Price Forecasting and Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-orange?logo=streamlit)](https://stocks-dashboard-athar.streamlit.app/)

**Access the Live Dashboard here ➔ [stocks-dashboard-athar.streamlit.app](https://stocks-dashboard-athar.streamlit.app/)**


This project explores stock market data for major tech companies (Apple, Amazon, Google, and Microsoft) through two approaches:

1. **Exploratory Data Analysis and Stock Price Prediction with LSTM (Notebook)**
2. **Interactive Stock Forecasting Dashboard with Prophet (Web App)**

---

## 📊 1. Stock Analysis and Forecasting (Jupyter Notebook)

In the notebook `stock-price-prediction.ipynb`, we:

- Fetch stock data using **yFinance**
- Perform **Exploratory Data Analysis** (EDA) with **Seaborn** and **Matplotlib**
- Analyze stock risk based on historical returns
- Build and train an **LSTM** model to predict future stock prices

**Libraries used**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `yfinance`, `keras`, `tensorflow`

---

## 🖥️ 2. Stock Forecasting Dashboard (Streamlit App)

The dashboard `stocks_dashboard.py` provides:

- Real-time stock data fetching from **Yahoo Finance**
- Future price forecasting using **Facebook Prophet**
- Interactive charts (candlestick and line plots) using **Plotly**
- Key stock metrics (high, low, volume, % change)
- Easy-to-use web UI built with **Streamlit**

**Libraries used**: `streamlit`, `yfinance`, `pandas`, `prophet`, `plotly`

---