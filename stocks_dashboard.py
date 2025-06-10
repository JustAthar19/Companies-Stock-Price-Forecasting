import yfinance as yf
import streamlit as st
import plotly.express as px
import pandas as pd
from plotly import graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Data Load & Preprocessing
forecast_periods = {
    '1wk': 7,
    '1mo': 30,
    '1y': 365,
    'max': 365
}
@st.cache_resource
def fetch_stock_data(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=False)
    return data

def process_data(data):
    data = data.droplevel(1, axis=1)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data.rename(columns={'Date':'Datetime'}, inplace=True)
    return data


# Forecasting Prophet & SARIMA for the next 30 days
def get_prophet_forecast(time_period):
    df = data[['Datetime', 'Close']].rename(columns={"Datetime": "ds", "Close": "y"})
    df.dropna(inplace=True)
    if time_period == "1wk":
        model = Prophet(daily_seasonality=True,weekly_seasonality=False, yearly_seasonality=False)
    elif time_period == "1mo":
        model = Prophet(daily_seasonality=True,weekly_seasonality=True, yearly_seasonality=False)
    else:
        model = Prophet(daily_seasonality=True,weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

def get_sarima_forecast():
    df = data[['Datetime', 'Close']]
    df.set_index('Datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    model = SARIMAX(df['Close'], order=(1, 1, 2), seasonal_order=(1, 1, 1, 21))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=30)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast.predicted_mean})
    return forecast_df

# Plotting
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
    st.plotly_chart(fig,use_container_width=True)


def daily_return_plot(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    fig = px.line(data, x=data['Datetime'], y='Daily Return', title='Daily Returns Over Time')
    fig.update_layout(yaxis_title='Daily Return', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)
    

# Basic Metric Calculation
def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume


st.set_page_config(layout='wide')
st.title("Stock Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Indicators", "🔮 Forecast", "📄 Raw Data"])

# Sidebar Configurations
st.sidebar.header("Configurations")
ticker = st.sidebar.selectbox("Select Company for Predictions", ['AAPL', 'GOOG', 'MSFT', 'AMZN'])
time_period = st.sidebar.selectbox("Time Period", ["1wk", "1mo", "1y", "max"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])

# Main content
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

    insight_text = f"As of {data['Datetime'].iloc[-1]}, {ticker} closed at ${last_close:.2f}, " \
                f"{'up' if change > 0 else 'down'} {abs(change):.2f} USD ({pct_change:.2f}%) " \
                f"from the beginning of the selected period."
    st.write(insight_text)

    # plot the stock chart
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(x=data['Datetime'],open=data['Open'],high=data['High'],
                                        low=data['Low'],close=data['Close']))
        if time_period == 'max':

            fig.add_vrect(x0="1987-01-01", x1="1988-01-01", fillcolor="gray", opacity=0.3, layer="below", line_width=0,
                      annotation_text="Black Monday", annotation_position="top left")

            fig.add_vrect(x0="2000-01-01", x1="2002-01-01", fillcolor="purple", opacity=0.3, layer="below", line_width=0,
                            annotation_text="Dot-com & 9/11", annotation_position="top left")

            fig.add_vrect(x0="2008-01-01", x1="2009-01-01", fillcolor="red", opacity=0.3, layer="below", line_width=0,
                            annotation_text="Financial Crisis", annotation_position="top left")

            fig.add_vrect(x0="2020-01-01", x1="2021-01-01", fillcolor="green", opacity=0.3, layer="below", line_width=0,
                            annotation_text="COVID 19", annotation_position="top left")

    

    else:
        fig = px.line(data, x='Datetime', y='Close')
    fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                      xaxis_title="Time", yaxis_title="Price (USD)", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volume")
    volume_fig = px.bar(data, x='Datetime', y='Volume', title="Trading Volume")
    st.plotly_chart(volume_fig, use_container_width=True)
    

with tab2: 
    st.subheader("Moving Average")
    if len(data) < 10 :
        st.caption("Data length is not sufficient enough to create moving average plot")
    else:
        moving_average_plot(data, ticker)
    st.subheader("Daily Return")
    daily_return_plot(data)
    st.subheader(f"Rolling Volatility {time_period}")
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=10).std()
    fig_vol = px.line(data, x='Datetime', y='Volatility', title=f'{forecast_periods[time_period]}-day Rolling Volatility')
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Cumulative Return")
    data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
    fig_cum = px.line(data, x='Datetime', y='Cumulative Return', title='Cumulative Return')
    st.plotly_chart(fig_cum, use_container_width=True)

with tab3:
    model_choice = st.radio("Select Forecasting Model", ['Prophet', 'SARIMA'])
    if st.button("Forecast"):
        # training_state = st.empty()
        # progress_bar = st.progress(0)
        training_state = st.text("Training in Progress")
        if model_choice == 'Prophet':
            # progress_bar.progress(20)
            forecast_data = get_prophet_forecast(time_period)
            # progress_bar.progress(100)
            # Confidence Interval Plot
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_upper'],
                mode='lines',
                name='Upper Confidence',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat_lower'],
                mode='lines',
                name='Lower Confidence',
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
        else: 
            # progress_bar.progress(20)
            forecast_data = get_sarima_forecast()
            # progress_bar.progress(100)
        training_state.text("Training Done ✅")

        fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], name='Forecast'))

    
    fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                            xaxis_title="Time",
                            yaxis_title="Price (USD)",
                            height = 600)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Historical Data")
    st.dataframe(data[['Datetime', 'Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']],  use_container_width=True)
    st.download_button("Download CSV", data.to_csv(index=False), file_name=f"{ticker}_historical.csv")

    st.subheader("Data Summary")
    st.dataframe(data.describe(), use_container_width=True)
    

# Sidebar 
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




