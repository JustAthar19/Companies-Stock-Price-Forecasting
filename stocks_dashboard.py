import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from prophet import Prophet


# get the data from yahoo finance 
@st.cache_resource
def fetch_stock_data(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=False)
    return data


# process data to make sure it's in the right format
def process_data(data):
    data = data.droplevel(1, axis=1)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data.rename(columns={'Date':'Datetime'}, inplace=True)
    return data

forecast_periods = {
    '1wk': 7,
    '1mo': 20,
    '1y': 365,
    'max': 365
}

# new forecasting method
@st.cache_data
def get_forecast(ticker, period, time_periods):
    raw_data = fetch_stock_data(ticker, period)
    data = process_data(raw_data)
    df = data[['Datetime', 'Close']].rename(columns={"Datetime": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=time_periods)
    forecast = model.predict(future)
    return forecast, data


def moving_average_plot(data, ticker):
    days = [10, 20, 50]
    for day in days:
        col = f"MA {day}"
        data[col] = data['Close'].rolling(window=day).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'],
                             mode='lines', name='Close'))

    for day in days:
        col = f"MA {day}"
        fig.add_trace(go.Scatter(x=data['Datetime'], y=data[col],
                                 mode='lines', name=f'MA {day} days'))

    fig.update_layout(
        title=f'{ticker} - Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        template='plotly_white',
        width=900,
        height=600
    )

    st.plotly_chart(fig)


def daily_return_plot(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    fig = px.line(data, x=data['Datetime'], y='Daily Return', title='Daily Returns Over Time')
    fig.update_layout(yaxis_title='Daily Return', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)
    


# calculate basic metric from the stock data
def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

# set up streamlit page layout
st.set_page_config(layout='wide')
st.title("Stock Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Indicators", "🔮 Forecast", "📄 Raw Data"])
st.sidebar.header("Configurations")
ticker = st.sidebar.selectbox("Select Company for Predictions", ['AAPL', 'GOOG', 'MSFT', 'AMZN'])
time_period = st.sidebar.selectbox("Time Period", ["1wk", "1mo", "1y", "max"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])

# Main content Area

data = fetch_stock_data(ticker, time_period)
data = process_data(data)
last_close, change, pct_change, high, low, volume = calculate_metrics(data)


with tab1: # Overview
    # display the main metrics
    st.metric(label=f"{ticker} last price", value=f"{round(last_close, 2)} USD", delta=f"{round(change,2)} ({round(pct_change)}%)")
    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{round(high,2)} USD")
    col2.metric("Low", f"{round(low,2)} USD")
    col3.metric("Volume", f"{volume}")

    # plot the stock chart
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(x=data['Datetime'],open=data['Open'],high=data['High'],
                                        low=data['Low'],close=data['Close']))

    else:
        fig = px.line(data, x='Datetime', y='Close')
    fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                      xaxis_title="Time", yaxis_title="Price (USD)", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab2: 
    st.subheader("Moving Average")
    if len(data) < 10 :
        st.caption("Data length is not sufficient enough to create moving average plot")
    else:
        moving_average_plot(data, ticker)
    st.subheader("Daily Return")
    daily_return_plot(data)

with tab3:
    if st.button("Forecast"):
        training_state = st.text("Training in Progress")
        # Prophet
        forecast_data, data = get_forecast(ticker, time_period, forecast_periods[time_period])
        training_state.text("Training Done")
        # Prophet Graph
        fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'][:len(data) + 4], name='predicted')) 
    fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                            xaxis_title="Time",
                            yaxis_title="Price (USD)",
                            height = 600)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # display historiacal data and technical indicator
    st.subheader("Historical Data")
    st.dataframe(data[['Datetime', 'Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']],  use_container_width=True)

# Sidebar Prices
# sidebar section for real-tine stock prices of selected symbol
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL', 'GOOG', 'AMZN', 'MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol, '1d')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        last_price = real_time_data['Close'].iloc[-1]
        change = last_price - real_time_data['Open'].iloc[0]
        pct_change = (change / real_time_data['Open'].iloc[0]) * 100
        st.sidebar.metric(f"{symbol}", f"{round(last_price,2)} USD", f"{round(change,2)} ({round(pct_change,2)}%)")




