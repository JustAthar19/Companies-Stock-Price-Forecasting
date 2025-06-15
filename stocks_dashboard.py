import yfinance as yf
import streamlit as st
import plotly.express as px
import pandas as pd
from plotly import graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import zscore
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

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
def get_prophet_forecast():
    df = data[['Datetime', 'Close']].rename(columns={"Datetime": "ds", "Close": "y"})
    df.dropna(inplace=True)
    if time_period == "1wk":
        model = Prophet(daily_seasonality=True,weekly_seasonality=False, yearly_seasonality=False)
    elif time_period == "1mo":
        model = Prophet(daily_seasonality=True,weekly_seasonality=True, yearly_seasonality=False)
    else:
        model = Prophet(daily_seasonality=True,weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    return model, forecast, df

def get_sarima_forecast():
    df = data[['Datetime', 'Close']]
    df.set_index('Datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    model = SARIMAX(df['Close'], order=(1, 1, 2), seasonal_order=(1, 1, 1, 21))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=forecast_horizon)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast.predicted_mean,
                                'yhat_lower': forecast.conf_int()['lower Close'],
                                'yhat_upper': forecast.conf_int()['upper Close']})
    return results, forecast_df, df

def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_squared_error(actual, predicted)
    return rmse, mae

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
    daily_returns = data['Adj Close'].pct_change().dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252) * 100  # Assuming 252 trading days
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
    return last_close, change, pct_change, high, low, volume, annualized_volatility, sharpe_ratio

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss.where(loss != 0, 1)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def data_quality_insights(data):
    missing_values = data.isnull().sum()
    outliers = data[abs(zscore(data['Close'].dropna())) > 3].shape[0]
    completeness = (1 - missing_values['Close'] / len(data)) * 100
    return missing_values, outliers, completeness


def generate_summary_report(ticker, data, metrics, anomalies):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Stock Summary Report: {ticker}")
    c.drawString(100, 730, f"Date: {data['Datetime'].iloc[-1]}")
    c.drawString(100, 710, f"Last Close: ${metrics[0]:.2f}")
    c.drawString(100, 690, f"Change: ${metrics[1]:.2f} ({metrics[2]:.2f}%)")
    c.drawString(100, 670, f"Annualized Volatility: {metrics[6]:.2f}%")
    c.drawString(100, 650, f"Sharpe Ratio: {metrics[7]:.2f}")
    c.drawString(100, 630, f"Anomalies Detected: {len(anomalies)}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


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
last_close, change, pct_change, high, low, volume, annualized_volatility, sharpe_ratio = calculate_metrics(data)


with tab1:
    st.subheader(f"{ticker} Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Last Price", value=f"{round(last_close, 1)} USD", delta=f"{round(change, 1)} ({round(pct_change, 1)}%)")
    with col2:
        st.metric("High", f"{round(high, 1)} USD")
    with col3:
        st.metric("Low", f"{round(low, 1)} USD")
    with col4:
        st.metric("Sharpe Ratio", f"{round(sharpe_ratio, 1)}")
    st.metric("Volume", f"{volume:,}")
    
    
    # Combined Price and Volume Chart
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(x=data['Datetime'], open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'], name='Price'))
    else:
        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Bar(x=data['Datetime'], y=data['Volume'], name='Volume', yaxis='y2', opacity=0.3))
    if time_period == 'max':
        fig.add_vrect(x0="1987-01-01", x1="1988-01-01", fillcolor="gray", opacity=0.3, layer="below", line_width=0,
                      annotation_text="Black Monday", annotation_position="top left")
        fig.add_vrect(x0="2000-01-01", x1="2002-01-01", fillcolor="purple", opacity=0.3, layer="below", line_width=0,
                      annotation_text="Dot-com & 9/11", annotation_position="top left")
        fig.add_vrect(x0="2008-01-01", x1="2009-01-01", fillcolor="red", opacity=0.3, layer="below", line_width=0,
                      annotation_text="Financial Crisis", annotation_position="top left")
        fig.add_vrect(x0="2020-01-01", x1="2021-01-01", fillcolor="green", opacity=0.3, layer="below", line_width=0,
                      annotation_text="COVID 19", annotation_position="top left")
    fig.update_layout(
        title=f'{ticker} Price and Volume ({time_period.upper()})',
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(title="Volume", overlaying='y', side='right'),
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio Correlation
    st.subheader("Portfolio Correlation")
    stock_symbols = ['AAPL', 'GOOG', 'AMZN', 'MSFT']
    corr_data = pd.DataFrame()
    for symbol in stock_symbols:
        temp_data = fetch_stock_data(symbol, time_period)
        temp_data = process_data(temp_data)
        if not temp_data.empty:
            corr_data[symbol] = temp_data.set_index('Datetime')['Close']
    if not corr_data.empty:
        corr_matrix = corr_data.pct_change().corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu',
                             title='Correlation of Daily Returns')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.write(f"{ticker} has {['low', 'moderate', 'high'][int(corr_matrix[ticker].mean() / 0.33)]} correlation with other stocks, suggesting {'diversification potential' if corr_matrix[ticker].mean() < 0.5 else 'similar market behavior'}.")
    else:
        st.write("Correlation data not available.")

    # Collapsible Insight Summary
    with st.expander("Performance Summary"):
        st.write(f"As of {data['Datetime'].iloc[-1]}, {ticker} closed at ${last_close:.2f}, "
                 f"{'up' if change > 0 else 'down'} {abs(change):.2f} USD ({pct_change:.2f}%) "
                 f"from the start of the period. Annualized volatility: {annualized_volatility:.2f}%.")

with tab2:
    st.subheader("Technical Analysis")
    # Indicator Selection
    indicators = st.multiselect(
        "Select Indicators to Display",
        ["Moving Averages", "RSI", "Volatility", "Anomalies"],
        default=["Moving Averages", "RSI"]
    )
    data['RSI'] = calculate_rsi(data)

    daily_return = data['Adj Close'].pct_change()
    z_score = zscore(daily_return.dropna())
    data['Daily Return'] = daily_return
    data['Z-Score'] = z_score
    anomalies = data[abs(data['Z-Score']) > 3]

    fig_tech = go.Figure()
    fig_tech.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name='Close'))
    
    summary_metrics = {}
    if "Moving Averages" in indicators and len(data) >= 10:
        days = [10, 20, 50]
        for day in days:
            col = f"MA {day}"
            data[col] = data['Close'].rolling(window=day).mean()
            fig_tech.add_trace(go.Scatter(x=data['Datetime'], y=data[col], mode='lines', name=f'MA {day}'))
        summary_metrics["Latest MA (50-day)"] = f"{data['MA 50'].iloc[-1]:.2f} USD"

    if "RSI" in indicators and len(data) >= 14:
        fig_tech.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI'], name='RSI', yaxis='y2'))
        fig_tech.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", yref='y2')
        fig_tech.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", yref='y2')
        summary_metrics["Latest RSI"] = f"{data['RSI'].iloc[-1]:.2f}"
    
    if "Volatility" in indicators:
        data['Volatility'] = data['Daily Return'].rolling(window=10).std()
        fig_tech.add_trace(go.Scatter(x=data['Datetime'], y=data['Volatility'], name='Volatility', yaxis='y3'))
        summary_metrics["Latest Volatility"] = f"{data['Volatility'].iloc[-1]:.4f}"
    
    if "Anomalies" in indicators: 
        fig_tech.add_trace(go.Scatter(x=anomalies['Datetime'], y=anomalies['Close'],
                                      mode='markers', name='Anomalies', marker=dict(size=10, color='red')))
        summary_metrics["Anomalies Detected"] = f"{len(anomalies)}"
    
    fig_tech.update_layout(
        title=f'{ticker} Technical Indicators ({time_period.upper()})',
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100], showgrid=False),
        yaxis3=dict(title="Volatility", overlaying='y', side='right', anchor='free', position=0.95, showgrid=False),
        height=600,
        template='plotly_white',
        showlegend=True
    )
    st.plotly_chart(fig_tech, use_container_width=True)
    
    if summary_metrics:
        st.subheader("Indicator Summary")
        cols = st.columns(len(summary_metrics))
        for i, (label, value) in enumerate(summary_metrics.items()):
            cols[i].metric(label, value)

with tab3:
    st.subheader("Stock Price Forecast")
     # Model Explanation
    with st.expander("About the Models"):
        st.write("""
                **Prophet**: A flexible forecasting model by Meta AI, suitable for daily, weekly, and yearly seasonality. It handles missing data and trends well but may overfit on short periods.\n
                **SARIMA**: A statistical model capturing autoregressive, differencing, and moving average components with seasonal patterns. It’s robust for stable time series but sensitive to parameter choice.
            """)
    # Configuration Panel
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            model_choice = st.radio("Select Model", ['Prophet', 'SARIMA', 'Both'])
        with col2:
            forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
    
    if st.button("Forecast"):
        training_state = st.text("Training in Progress")
        fig_forecast = go.Figure()
        metrics = {}
        period_days = forecast_periods.get(time_period, 365)
        test_size = max(3, int(period_days * 0.2))
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        
        
        if model_choice in ['Prophet', 'Both']:
            prophet_model, prophet_forecast, prophet_train = get_prophet_forecast()
            fig_forecast.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'],
                                             mode='lines', name='Prophet Forecast'))
            if time_period not in ["1wk", "1mo"]:
                fig_forecast.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'],
                                                mode='lines', name='Prophet Upper CI', line=dict(width=0), showlegend=False))
                fig_forecast.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'],
                                                mode='lines', name='Prophet Lower CI', fill='tonexty',
                                                fillcolor='rgba(0, 100, 80, 0.2)', line=dict(width=0), showlegend=False))
            if not test_data.empty:
                prophet_test_pred = prophet_model.predict(test_data[['Datetime', 'Close']].rename(columns={"Datetime": "ds", "Close": "y"}))
                rmse, mae = evaluate_forecast(test_data['Close'], prophet_test_pred['yhat'][-test_size:])
                metrics['Prophet'] = {'RMSE': rmse, 'MAE': mae}
       
        if model_choice in ['SARIMA', 'Both']:
            sarima_model, sarima_forecast, sarima_train = get_sarima_forecast()
            fig_forecast.add_trace(go.Scatter(x=sarima_forecast['ds'], y=sarima_forecast['yhat'],
                                             mode='lines', name='SARIMA Forecast'))
            if time_period not in ["1wk", "1mo"]:
                fig_forecast.add_trace(go.Scatter(x=sarima_forecast['ds'], y=sarima_forecast['yhat_upper'],
                                                mode='lines', name='SARIMA Upper CI', line=dict(width=0), showlegend=False))
                fig_forecast.add_trace(go.Scatter(x=sarima_forecast['ds'], y=sarima_forecast['yhat_lower'],
                                                mode='lines', name='SARIMA Lower CI', fill='tonexty',
                                                fillcolor='rgba(255, 99, 132, 0.2)', line=dict(width=0), showlegend=False))
            if not test_data.empty:
                sarima_test_pred = sarima_model.get_forecast(steps=test_size).predicted_mean
                rmse, mae = evaluate_forecast(test_data['Close'], sarima_test_pred[-test_size:])
                metrics['SARIMA'] = {'RMSE': rmse, 'MAE': mae}
    
        fig_forecast.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'],
                                         mode='lines', name='Historical Close'))
        fig_forecast.update_layout(title=f'{ticker} Forecast ({time_period.upper()})',
                                  xaxis_title="Date", yaxis_title="Price (USD)", height=600, template='plotly_white')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Model Performance
        if metrics:
            st.subheader("Model Performance")
            
            metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], name='RMSE', marker_color='#36A2EB'))
            fig_compare.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'], name='MAE', marker_color='#FF6384'))
            fig_compare.update_layout(title='Model Comparison', xaxis_title='Model', yaxis_title='Error',
                                         barmode='group', template='plotly_white', height=400)
            st.plotly_chart(fig_compare, use_container_width=True)
        

       
        
        training_state.text("Training Done ✅")


with tab4:
    st.subheader("Data Exploration")
    
    # Data Preview
    st.write("**Historical Data**")
    st.dataframe(
        data[['Datetime', 'Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']],
        use_container_width=True,
        column_config={
            "Datetime": st.column_config.DateColumn("Date"),
            "Adj Close": st.column_config.NumberColumn(format="$%.2f"),
            "Open": st.column_config.NumberColumn(format="$%.2f"),
            "High": st.column_config.NumberColumn(format="$%.2f"),
            "Low": st.column_config.NumberColumn(format="$%.2f"),
            "Close": st.column_config.NumberColumn(format="$%.2f"),
            "Volume": st.column_config.NumberColumn(format="%,d")
        }
    )
    
    # Data Quality Insights
    st.subheader("Data Quality")
    missing_values, outliers, completeness = data_quality_insights(data)
    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values (Close)", missing_values['Close'])
    col2.metric("Outliers in Close", outliers)
    col3.metric("Data Completeness", f"{completeness:.2f}%")
    
    # Download Options
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download CSV", data.to_csv(index=False), file_name=f"{ticker}_historical.csv")
        st.caption("Full dataset in CSV format.")
    with col2:
        report_buffer = generate_summary_report(ticker, data, calculate_metrics(data), anomalies)
        st.download_button("Download PDF Report", report_buffer, file_name=f"{ticker}_summary.pdf", mime="application/pdf")
        st.caption("Summary report with key metrics and anomalies.")
    
    # Data Summary
    with st.expander("Statistical Summary"):
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