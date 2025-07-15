import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df= pd.read_csv("market value/Nifty50 Dataset.csv")
df = df.dropna()
df=df.drop(columns=["Price","Close","High","Low"])
print(df.head())

# Min max Normqalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

value_target_column = 'Open'
features = scaled_df.columns.tolist()

def create_sequences(data, target_col, time_steps=60):
    X, Y = [], []
    for i in range(time_steps, len(data)):
        X.append(data.iloc[i-time_steps:i].values)
        Y.append(data.iloc[i][target_col])
    return np.array(X), np.array(Y)

X_value, Y_value = create_sequences(scaled_df, value_target_column)

# Splitting the data into train and test sets
# Using 80% for training and 20% for testing
train_size = int(len(X_value) * 0.8)
X_value_train, X_value_test = X_value[:train_size], X_value[train_size:]
Y_value_train, Y_value_test = Y_value[:train_size], Y_value[train_size:]


SEQ_LEN = X_value.shape[1]  # Number of time steps in the sequence
print(SEQ_LEN)
print("X_train shape:", X_value_train.shape)  # (samples, time_steps, features)
print("y_train shape:", Y_value_train.shape)  # (samples,)