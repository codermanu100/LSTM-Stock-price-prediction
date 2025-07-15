import yfinance as yf
import pandas as pd

df = yf.download("^NSEI", start="2014-01-01", end="2025-01-01", interval="1d")
df.dropna(inplace=True)

# Calculating RSI (14-day)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Open'])

# Calculating MACD (12, 26, 9 EMA)
ema_12 = df['Open'].ewm(span=12, adjust=False).mean()
ema_26 = df['Open'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calculating ATR (14-day)
high_low = df['High'] - df['Low']
high_close_prev = abs(df['High'] - df['Open'].shift(1))
low_close_prev = abs(df['Low'] - df['Open'].shift(1))

tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
df['ATR'] = tr.rolling(window=14).mean()

df = df.dropna()
df.to_csv("market value/Nifty50 Dataset.csv")

print(df.head())
