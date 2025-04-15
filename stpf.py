import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM




st.title('Apple/Google/MSFT/AMZN Stock Price Forecasting')
stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN')
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Year of Prediction:", 1,4)
period = n_years * 365 # convert it into number of days


end = datetime.now().strftime("%Y-%m-%d")
start = '2015-01-01'
st.write(end)
st.write(start)
# start = '2015-01-01'
# end = '2025-01-01'
@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, start, end)
    # data = data.reset_index()
    # data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    return data

""" 
DATA PREPROCESSING
"""    

def lstm_data_load():
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:len_train_data, :]
    test_data = scaled_data[len_train_data - 60:, :]

    X_train, y_train = [], []
    X_test, y_test = [], dataset[len_train_data:] 

    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i,0])

    X_train, y_train,  = np.array(X_train), np.array(y_train)    
    X_test = np.array(X_test)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, X_test, y_test



""" 
MODEL TRAINING
"""


@st.cache_resource
def train(X_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train,epochs=5)    
    return model

def predict(X_test, y_test, model):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(rmse)
    return predictions





""" 
PLOTTING
"""
def get_stock_data_plot(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'][f'{selected_stocks}'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'][f'{selected_stocks}'], name="Stock Close"))
    fig.layout.update(title_text=f"{selected_stocks} Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)




def get_daily_return(data):
    data['Daily Return'] = data['Close'].pct_change()
    fig, ax = plt.subplots()
    ax.hist(data['Daily Return'].dropna(), bins=50)  # dropna to avoid NaNs
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Counts')
    ax.set_title(f'{selected_stocks} Daily Returns')
    st.pyplot(fig)


def get_close_price_history(data):
    fig, ax = plt.subplots(figsize=(16,6))
    ax.set_title(f'{selected_stocks} Close Price History')
    ax.plot(data['Close'])
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price USD ($)', fontsize=18)
    st.pyplot(fig)

def plot_predictions(data, predictions, len_train_data):
    plt.style.use("fivethirtyeight")
    train = data[:len_train_data]
    valid = data[len_train_data:].copy()
    valid['Predictions'] = predictions

    # Create the figure
    fig = plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    # Show plot in Streamlit
    st.pyplot(fig)



def plot_pred(data, predictions, len_train_data):
    # data = data.reset_index()
    train = data[:len_train_data].copy()
    valid = data[len_train_data:].copy()
    valid['Predictions'] = predictions
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data.index, y=data['Open'][f'{selected_stocks}'], name="Stock Open"))

    fig.add_trace(go.Scatter(x=data.index, y=train['Close'][f'{selected_stocks}'], name="Close Price"))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'][f'{selected_stocks}'], name="Validations"))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name="Predictions"))
    
    fig.layout.update(title_text=f"{selected_stocks} Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


    
data_load_state = st.text("load the data....")
data = load_data(selected_stocks)
data_load_state.text("Loading data ... done")
st.subheader('Raw Data')
st.write(data.tail())


scaler  = MinMaxScaler(feature_range=(0,1))

dataset = data['Close'].values
len_train_data = int(np.ceil(len(dataset) * .95))


scaled_data = scaler.fit_transform(dataset)

X_train, y_train, X_test, y_test = lstm_data_load()


models = train(X_train, y_train)
predictions = predict(X_test, y_test, models)
# # get_stock_data_plot(data)
# # get_close_price_history(data)
plot_pred(data, predictions, len_train_data)
get_daily_return(data)




# st.write(valid[['Close', 'Volume']])
# st.write(valid)

# plot_predictions(data, predictions, len_train_data)



## Add Button ##
valid = data[len_train_data:].copy()
valid['Predictions'] = predictions
st.write(valid)
st.write(valid['Predictions'])
st.write(valid['Close'][f'{selected_stocks}'])
