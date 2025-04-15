import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from plotly import graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly


end = '2015-01-01'
start = datetime.now().strftime("%Y-%m-%d")

# end = datetime.today().strftime("%Y-%m-%d")
# start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d") 

st.title('Stock Prediction App')
stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN')
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Year of Prediction:", 1,4)
period = n_years * 365 # convert it into number of days


@st.cache_data # this will cache the file and avoid redundant computation or api calls
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True) #put the date into the index column
    return data
 
data_load_state = st.text("load the data....")
data = load_data(selected_stocks) 
data_load_state.text("Loading data ... done")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df = data[['Date', 'Close']]
df = df.rename(columns={
    "Date":"ds",
    "Close" : "y"
    })
ph = Prophet()


ph.fit(df)
future = ph.make_future_dataframe(periods=period)
forecast = ph.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.subheader("Forecast Data")
fig1 = plot_plotly(ph, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = ph.plot_components(forecast)
st.write(fig2)