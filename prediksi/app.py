import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

from itertools import cycle
import plotly.express as px
import streamlit as st 
import plotly.graph_objs as go
from plotly import graph_objs as go
from datetime import date
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

START = st.date_input('Enter Start Date')
#TODAY = date.today().strftime("%Y-%m-%d")

st.title("Nilai Tukar Mata uang") 
stocks = ("IDR=X")
#period = 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(stocks)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.head()) 

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data() 

#Forecasting
# Forecasting
df_train = data[['Date', 'Close']]
df_train.set_index('Date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(df_train).reshape(-1, 1))

training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size
train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(10, input_shape=(time_step, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, verbose=1)

# Forecasting for the next 30 days
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []
n_steps = time_step
i = 0
pred_days = 30

while (i < pred_days):
    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

lst_output = scaler.inverse_transform(lst_output)
forecast_dates = pd.date_range(data['Date'].iloc[-1], periods=pred_days + 1).tolist()

# Create a plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
fig.add_trace(go.Scatter(x=forecast_dates, y=np.concatenate([data['Close'].values[-1:], lst_output.flatten()]), name='Forecast'))

fig.update_layout(title_text='LSTM Forecasting',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  legend_title_text='Series',
                  plot_bgcolor='white',
                  font_size=15,
                  font_color='black')

st.plotly_chart(fig)
st.subheader('Predicted Close Price for Next 30 Days')
st.write(forecast_dates)