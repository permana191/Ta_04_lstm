import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import joblib
import os

# 1. Persiapan Folder
os.makedirs('models', exist_ok=True)
os.makedirs('app/static/images', exist_ok=True) # Untuk menyimpan grafik (figures)

print("1. Memuat dan Membersihkan Data...")
df = pd.read_csv('dataset/Foreign_Exchange_Rates.csv')

# Menggunakan mata uang EURO sebagai target prediksi
target_col = 'EURO AREA - EURO/US$'

# Pembersihan Data: Mengubah 'ND' menjadi NaN dan memaksakan tipe numerik
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# Interpolasi untuk mengisi nilai kosong (NaN) agar urutan historis tidak terputus
df[target_col] = df[target_col].interpolate(method='linear')
# PERBAIKAN: Menggunakan .bfill() langsung untuk kompatibilitas Pandas versi terbaru
df[target_col] = df[target_col].bfill() 

# Konversi kolom waktu ke tipe datetime dan menjadikannya index
df['Time Serie'] = pd.to_datetime(df['Time Serie'])
df.set_index('Time Serie', inplace=True)

print("2. Membuat Visualisasi Eksplorasi Data (EDA)...")
plt.figure(figsize=(14, 6))
plt.plot(df.index, df[target_col], label='Harga Aktual (EURO/USD)', color='blue')
plt.title('Grafik Historis Nilai Tukar EURO terhadap USD (2000 - 2019)')
plt.xlabel('Tanggal')
plt.ylabel('Nilai Tukar')
plt.legend()
plt.grid(True)
plt.savefig('app/static/images/eda_plot.png')
plt.close()
print(" -> Grafik EDA berhasil disimpan di app/static/images/eda_plot.png")

print("3. Pra-pemrosesan Data (Scaling & Windowing)...")
dataset = df[target_col].values.reshape(-1, 1)

# Normalisasi menggunakan MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Memisahkan Data Latih (80%) dan Data Uji (20%)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:] # Mundur 60 hari untuk memprediksi data uji pertama

# Membuat Sekuens (Windowing) dengan timestep = 60 hari
timesteps = 60
X_train, y_train = [], []
for i in range(timesteps, len(train_data)):
    X_train.append(train_data[i-timesteps:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Penyiapan Data Uji
X_test, y_test_scaled = [], []
for i in range(timesteps, len(test_data)):
    X_test.append(test_data[i-timesteps:i, 0])
    y_test_scaled.append(test_data[i, 0])
X_test, y_test_scaled = np.array(X_test), np.array(y_test_scaled)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Data aktual untuk perbandingan hasil evaluasi
y_test_actual = dataset[train_size:]

print("4. Membangun Arsitektur Model LSTM...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

print("5. Melatih Model (Training akan memakan waktu beberapa saat)...")
# Melatih model (Epoch disesuaikan menjadi 25)
history = model.fit(
    X_train, y_train, 
    batch_size=32, 
    epochs=25, 
    validation_data=(X_test, y_test_scaled)
)

print("6. Menyimpan Model dan Objek Scaler...")
model.save('models/lstm_forex_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')

print("7. Evaluasi Model dan Prediksi...")
predictions = model.predict(X_test)
# Mengembalikan nilai prediksi ke skala mata uang aslinya
predictions = scaler.inverse_transform(predictions)

# Menghitung metrik Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f"\n>>> Nilai Root Mean Squared Error (RMSE): {rmse:.4f} <<<\n")

print("8. Membuat Visualisasi Hasil Prediksi (Actual vs Predicted)...")
# Pemisahan frame untuk plotting
train_plot = df.iloc[:train_size]
valid_plot = df.iloc[train_size:].copy()
valid_plot['Predictions'] = predictions

plt.figure(figsize=(14, 6))
# plt.plot(train_plot.index, train_plot[target_col], label='Data Latih (Train)')
plt.plot(valid_plot.index, valid_plot[target_col], label='Harga Aktual (Data Uji)', color='blue')
plt.plot(valid_plot.index, valid_plot['Predictions'], label='Harga Prediksi LSTM', color='orange')
plt.title(f'Prediksi Nilai Tukar EURO/USD Menggunakan LSTM (RMSE: {rmse:.4f})')
plt.xlabel('Tanggal')
plt.ylabel('Nilai Tukar (USD)')
plt.legend()
plt.grid(True)
plt.savefig('app/static/images/prediction_plot.png')
plt.close()
print(" -> Grafik Prediksi berhasil disimpan di app/static/images/prediction_plot.png")

print("Selesai! Seluruh proses pelatihan model berhasil.")