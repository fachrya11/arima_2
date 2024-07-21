import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# Muat model ARIMA
with open('model/arima_model.pkl', 'rb') as file:
    model_ARIMA = pickle.load(file)

# Aplikasi Streamlit
st.title("Prediksi Harga Saham")
st.write("Aplikasi ini menggunakan model ARIMA untuk memprediksi harga saham.")

# Input tanggal
start_date = st.date_input("Tanggal Mulai", datetime.today() - timedelta(days=7))
end_date = st.date_input("Tanggal Akhir", datetime.today())

if st.button('Prediksi'):
    try:
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Gunakan model ARIMA untuk memprediksi harga saham
        predictions_diff = model_ARIMA.predict(start=len(date_range), end=len(date_range) + (end_date - start_date).days - 1)
        predictions_diff_cumsum = predictions_diff.cumsum()
        last_value = model_ARIMA.fittedvalues[-1]
        predictions = last_value + predictions_diff_cumsum

        # Siapkan hasil
        results = pd.DataFrame({'Tanggal': date_range, 'Harga Prediksi': predictions})
        
        # Tampilkan hasil
        st.write(results)

        # Plot hasil
        st.line_chart(results.set_index('Tanggal'))
    except Exception as e:
        st.error(f"Error: {str(e)}")
