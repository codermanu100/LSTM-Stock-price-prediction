import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Bidirectional,BatchNormalization, LSTM, Dense, Dropout #type: ignore
import Datapreprocessing as dp
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, Adagrad, Nadam, AdamW #type: ignore
from sklearn.metrics import classification_report, confusion_matrix

value_model= Sequential([
    LSTM(units=152, input_shape=(dp.SEQ_LEN, dp.X_value_train.shape[2])),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])
value_model.compile(optimizer=Adagrad(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
value_model.summary()
history = value_model.fit(dp.X_value_train, dp.Y_value_train, epochs=52, batch_size=8, validation_data=(dp.X_value_test, dp.Y_value_test), verbose=1)
y_pred = value_model.predict(dp.X_value_test)


# Create dummy array to inverse transform just the 'Open' column
# Append predictions to same number of features for proper shape
dummy_input = np.zeros((len(y_pred), dp.scaled_df.shape[1]))
dummy_input[:, dp.df.columns.get_loc("Open")] = y_pred.flatten()

y_pred_inv = dp.scaler.inverse_transform(dummy_input)[:, dp.df.columns.get_loc("Open")]

dummy_input[:, dp.df.columns.get_loc("Open")] = dp.Y_value_test.flatten()
y_test_inv = dp.scaler.inverse_transform(dummy_input)[:, dp.df.columns.get_loc("Open")]


# print(classification_report(dp.Y_value_test, preds))
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title("Actual vs Predicted Close Price")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Save the models
value_model.save("Market value.keras")




