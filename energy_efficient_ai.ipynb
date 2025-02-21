import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.lite as tflite
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Simulating IoT Sensor Data
def generate_synthetic_data(samples=5000):
    np.random.seed(42)
    cpu_usage = np.random.uniform(10, 90, samples)  # CPU usage in %
    network_latency = np.random.uniform(5, 200, samples)  # Latency in ms
    power_draw = np.random.uniform(0.5, 10, samples)  # Power in Watts
    workload = np.random.uniform(1, 100, samples)  # Workload intensity
    energy_consumption = 0.3 * cpu_usage + 0.4 * network_latency + 0.2 * power_draw + 0.1 * workload + np.random.normal(0, 5, samples) 
    
    data = pd.DataFrame({
        'CPU_Usage': cpu_usage,
        'Network_Latency': network_latency,
        'Power_Draw': power_draw,
        'Workload': workload,
        'Energy_Consumption': energy_consumption
    })
    return data

# Generate Data
data = generate_synthetic_data()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
df_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Splitting Data
X = df_scaled.drop(columns=['Energy_Consumption'])
y = df_scaled['Energy_Consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a Lightweight AI Model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train Model
model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, verbose=1)

# Convert to TensorFlow Lite Model
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model converted to TensorFlow Lite format!")

convert_to_tflite(model)

# Visualizing Results
plt.plot(history.history['mae'], label='MAE (Train)')
plt.plot(history.history['val_mae'], label='MAE (Test)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Model Performance')
plt.show()
