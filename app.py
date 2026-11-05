from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta
import os

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Konfigurasi Path
MODEL_PATH = 'models/lstm_forex_model.h5'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'dataset/Foreign_Exchange_Rates.csv'

# Variabel Global
model = None
scaler = None
last_60_days = None
last_date = None

def init_model_data():
    """Fungsi untuk meload model dan 60 data observasi terakhir saat server Flask menyala."""
    global model, scaler, last_60_days, last_date
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Load dataset untuk mengambil memori 60 hari terakhir
        df = pd.read_csv(DATA_PATH)
        target_col = 'EURO AREA - EURO/US$'
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df[target_col] = df[target_col].interpolate(method='linear').bfill()
        df['Time Serie'] = pd.to_datetime(df['Time Serie'])
        df = df.sort_values('Time Serie')
        
        last_60_days = df[target_col].values[-60:]
        last_date = df['Time Serie'].iloc[-1]

# Panggil fungsi inisialisasi
init_model_data()

@app.route('/')
def index():
    return render_template('index.html', title="Dashboard Utama")

@app.route('/data')
def data():
    return render_template('data.html', title="Data Historis Forex")

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    result = None
    if request.method == 'POST':
        days = int(request.form.get('days'))
        
        if model is None or scaler is None:
            result = {"error": "Model LSTM belum dilatih. Silakan jalankan train_model.py terlebih dahulu."}
            return render_template('forecast.html', title="Prediksi Masa Depan", result=result)

        # 1. Siapkan input 60 hari terakhir
        current_input = last_60_days.reshape(-1, 1)
        current_input_scaled = scaler.transform(current_input)
        current_batch = current_input_scaled.reshape(1, 60, 1)
        
        predictions_scaled = []
        
        # 2. Lakukan prediksi secara iteratif (Sliding Window)
        for _ in range(days):
            next_pred = model.predict(current_batch, verbose=0)
            predictions_scaled.append(next_pred[0, 0])
            
            # Geser window: hapus hari pertama, masukkan hasil prediksi terbaru ke memori
            next_pred_reshaped = next_pred.reshape(1, 1, 1)
            current_batch = np.append(current_batch[:, 1:, :], next_pred_reshaped, axis=1)
            
        # 3. Kembalikan skala prediksi ke nilai mata uang USD asli (Inverse Transform)
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
        
        # 4. Format hasil untuk ditampilkan ke tabel HTML
        result = []
        for i in range(days):
            pred_date = last_date + timedelta(days=i+1)
            result.append({
                'hari': i + 1,
                'tanggal': pred_date.strftime('%d %B %Y'),
                'nilai': round(float(predictions[i][0]), 4)
            })
            
    return render_template('forecast.html', title="Prediksi Masa Depan", result=result)

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html', title="Evaluasi Model LSTM")

@app.route('/about')
def about():
    return render_template('about.html', title="Tentang Proyek TA-04")

if __name__ == '__main__':
    app.run(debug=True)