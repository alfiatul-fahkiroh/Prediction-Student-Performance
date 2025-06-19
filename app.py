# app.py

import streamlit as st
import joblib
import numpy as np

# 1. Load model
with open('model_graduation.pkl', 'rb') as file:
    model = joblib.load(file)

# 2. Judul aplikasi
st.title('Prediksi Kelulusan Mahasiswa')

# 3. Form input
st.header('Masukkan Data Mahasiswa')
ACT = st.number_input('Nilai ACT composite score', min_value=0.0)
SAT = st.number_input('Nilai SAT total score', min_value=0.0)
GPA = st.number_input('Nilai rata-rata SMA', min_value=0.0)
income = st.number_input('Pendapatan orang tua', min_value=0.0)

# 4. Tombol prediksi
if st.button('Prediksi'):
    # Format input sebagai array
    input_data = np.array([[ACT, SAT, GPA, income]])
    prediction = model.predict(input_data)
    st.success(f'Prediksi Kelulusan: {prediction[0]}')

# 5. Jalankan di terminal: streamlit run app.py
