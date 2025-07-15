import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model #type: ignore
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

now=datetime.now()
# === Configuration ===
ticker = "^NSEI"  # Nifty 50 Index
start_date = "2018-01-01"
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
seq_len = 60
steps_ahead = 1
target_column = "Open"

# === 1. Download data ===
df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)

print(df.tail())


# === 2. Calculate Technical Indicators ===
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Open'])

# Calculate MACD and MACD Signal (12, 26, 9 EMA)
ema_12 = df['Open'].ewm(span=12, adjust=False).mean()
ema_26 = df['Open'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calculate ATR (14-day)
high_low = df['High'] - df['Low']
high_close_prev = abs(df['High'] - df['Open'].shift(1))
low_close_prev = abs(df['Low'] - df['Open'].shift(1))
tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
df['ATR'] = tr.rolling(window=14).mean()

# Drop NaNs after indicator calculations
df.dropna(inplace=True)

# Select only relevant features
feature_columns = ["Open", "Volume", "RSI", "MACD", "MACD_Signal", "ATR"]
df = df[feature_columns]




# === 3. Normalize ===
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=feature_columns, index=df.index)



# === 4. Sequence preparation ===
def create_sequences(data, time_steps):
    X = []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i-time_steps:i].values)
    return np.array(X)

X = create_sequences(scaled_df, seq_len)
initial_sequence = X[-1].reshape(1, seq_len, len(feature_columns))



# === 5. Load model ===
model = load_model("market value/Market Value.keras")



# === 6. Predict future values ===
def predict_future_values(model, initial_sequence, steps_ahead, scaler, feature_columns, target_column, seq_len):
    predicted_values = []
    current_input = initial_sequence

    for _ in range(steps_ahead):
        predicted_value = model.predict(current_input, verbose=0)[0][0]
        predicted_values.append(predicted_value)

        next_input = np.roll(current_input, shift=-1, axis=1)
        next_input[0, -1, feature_columns.index(target_column)] = predicted_value
        current_input = next_input

    dummy_input = np.zeros((len(predicted_values), len(feature_columns)))
    dummy_input[:, feature_columns.index(target_column)] = predicted_values
    predicted_values_inv = scaler.inverse_transform(dummy_input)[:, feature_columns.index(target_column)]

    return predicted_values_inv

predicted_values = predict_future_values(
    model,
    initial_sequence,
    steps_ahead,
    scaler,
    feature_columns,
    target_column,
    seq_len
)



# === 7. Plot predictions ===
print(f"Predicted 'Open' prices for next {steps_ahead} days:")
print(predicted_values)

plt.plot(range(1, steps_ahead + 1), predicted_values, marker='o', label='Predicted Open Prices')
plt.title(f"Predicted Open Prices for {ticker}")
plt.xlabel("Days Ahead")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()
