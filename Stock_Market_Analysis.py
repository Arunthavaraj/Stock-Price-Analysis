import streamlit as st
import subprocess
import os
import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#  Streamlit App Title
st.title("📈 Live Stock Price Prediction using LSTM")
st.write("Select a stock and predict the next closing price using an LSTM model.")

#  User Input: Select a Stock Ticker
stocks = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL']
ticker = st.selectbox("Select a stock:", stocks)

#  Calculate Date Range: Last 365 days
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=365)

#  Fetch Historical Data for the last 365 days
st.write(f"Fetching data for **{ticker}**...")
data = yf.download(ticker, start=start_date, end=end_date)

#  Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#  Create Sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    # Ensure there is enough data to create sequences
    if len(data) <= seq_length:
        return np.array([]), np.array([])  # Return empty arrays if not enough data
    
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(scaled_data, seq_length)

# Check if data is not empty before reshaping
if X_train.size > 0:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM
else:
    st.warning(f"Not enough data to create sequences. You need at least {seq_length} data points.")
    X_train, y_train = np.array([]), np.array([])

#  Build LSTM Model (only if data is available)
if X_train.size > 0:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    #  Train Model
    st.write("Training the LSTM model... This may take a few minutes.")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    st.success("✅ Model training complete!")
else:
    st.warning("Model training not completed due to insufficient data.")

# 📡 Live Stock Price Prediction
def predict_next_close(ticker):
    # Fetch last 365 days of data for live prediction
    live_data = yf.download(ticker, period='365d', interval='1d')['Close'].values
    if len(live_data) < 365:
        return None
    
    # Scale and reshape the data
    live_data_scaled = scaler.transform(live_data.reshape(-1, 1))
    live_input = np.array([live_data_scaled])
    live_input = np.reshape(live_input, (live_input.shape[0], live_input.shape[1], 1))

    # Predict the next closing price
    predicted_price = model.predict(live_input)
    return scaler.inverse_transform(predicted_price)[0][0]

#  Predict Button
if st.button("🔮 Predict Next Closing Price"):
    if X_train.size > 0:
        predicted_price = predict_next_close(ticker)
        if predicted_price:
            st.success(f"📊 Predicted Next Closing Price for **{ticker}**: **${predicted_price:.2f}**")
        else:
            st.warning("Not enough recent data for prediction.")
    else:
        st.warning("Model not trained due to insufficient data.")

# Display Stock Price Chart
st.subheader(f"📉 {ticker} Stock Price History")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data['Close'], label=f'{ticker} Closing Price', color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{ticker} Closing Price Over Time")
ax.legend()
st.pyplot(fig)
