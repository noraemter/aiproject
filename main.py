import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
data = pd.read_csv('dataset/urban_mobility_data_past_year.csv')
data = data.dropna()

# One-hot encode categorical features

encoder = OneHotEncoder(sparse_output=False, drop=None) 
weather_encoded = encoder.fit_transform(data[['weather_conditions']])
weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather_conditions']))
temperature_df = data[['temperature']].reset_index(drop=True)

X = pd.concat([weather_df, temperature_df], axis=1)
y = data['congestion_level']

# Scale the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build the feedforward neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(4,))) 
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping])
# Evaluate the model
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)

print(f'Train Loss: {train_loss}')
print(f'Test Loss: {test_loss}')

# Make predictions
predictions = model.predict(X_test)
predictions_rescaled = scaler_y.inverse_transform(predictions)
y_test_rescaled = scaler_y.inverse_transform(y_test)

#Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual')
plt.plot(predictions_rescaled, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Index')
plt.ylabel('Congestion Level')
plt.legend()
plt.show()

mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, predictions_rescaled)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

# Save the trained model
model.save('traffic_congestion_model.keras')
