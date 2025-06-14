# 📈 Stock Price Forecasting and Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-orange?logo=streamlit)](https://stocks-dashboard-athar.streamlit.app/)

**Access the Live Dashboard here ➔ [stocks-dashboard-athar.streamlit.app](https://stocks-dashboard-athar.streamlit.app/)**


This project explores stock market data for major tech companies: **Apple (AAPL)**, **Amazon (AMZN)**, **Google (GOOG)**, and **Microsoft (MSFT)**—using two approaches:

1. **Exploratory Data Analysis & Price Prediction (Notebook)**
2. **Interactive Stock Forecasting Dashboard (Streamlit App)**

---

## 📊 1. Stock Analysis and Forecasting (Jupyter Notebook)

File: [`stock-price-prediction.ipynb`](/stock-price-prediction.ipynb)

In-depth analysis of stock market behavior and builds various forecasting models.

### Features:
- **Data Collection**: Real-time stock data via `yFinance`
- **Exploratory Data Analysis**:
  - Descriptive statistics
  - Closing Price trends
  - OHLC and Volume plots
  - Moving Averages (10, 20, 50-day)
  - Daily Returns
  - RSI (Relative Strength Index)
  - Volatility visualization
  - Correlation between stocks
  - Risk vs Return plots
- **Forecasting Models**:
  - **ARIMA**: ACF, PACF, differencing, stationarity checks (ADF test), STL decomposition
  - **LSTM and CNN + LSTM**: Deep learning models for sequential prediction
  - **Facebook Prophet**: Trend and seasonality decomposition forecasting
- **Model Evaluation**: RMSE comparison between models

### 🛠️ Libraries Used:
`pandas`, `numpy`, `seaborn`, `matplotlib`, `yfinance`, `keras`, `tensorflow`, `prophet`, `statsmodels`, `sklearn`

---

## 🖥️ 2. Stock Forecasting Dashboard (Streamlit App)

File: [`stocks_dashboard.py`](stocks_dashboard.py)

An interactive dashboard to monitor, analyze, and forecast stock prices in real time.

#### 📊 Overview Tab:
- Key metrics: latest price, high, low, volume, Sharpe ratio
- Interactive candlestick and line charts with volume
- Annotated historical events: Black Monday, Dot-com crash, 2008 Crisis, COVID-19
- Portfolio-level correlation analysis among AAPL, AMZN, GOOG, MSFT

#### 📈 Technical Indicators Tab:
- Moving Averages (10, 20, 50-day)
- RSI with overbought/oversold thresholds
- Volatility plot
- Anomaly detection using Z-score method

#### 🔮 Forecast Tab:
- Forecast stock prices with:
  - **Prophet**: additive model with seasonality
  - **SARIMA**: statistical modeling
- Forecast horizon selector (7 to 90 days)
- Model performance metrics: RMSE, MAE
- Confidence intervals visualization

#### 📄 Raw Data Tab:
- Interactive table with historical data
- Data quality metrics: missing values, outliers, completeness
- Export options:
  - CSV Download
  - PDF Summary Report via ReportLab

**Libraries used**: `streamlit`, `yfinance`, `pandas`, `plotly`, `prophet`, `statsmodels`, `sklearn`, `reportlab`

---

## Installation
```bash
git clone https://github.com/JustAthar19/Companies-Stock-Price-Forecasting.git
cd Companies-Stock-Price-Forecasting
pip install -r requirements.txt
```
---
## Running the Dashboard Locally
```bash
cd Companies-Stock-Price-Forecasting
streamlit run stocks_dashboard.oy
```