from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

# Muat model ARIMA
with open('model/arima_model.pkl', 'rb') as file:
    model_ARIMA = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        
        # Generate date range for prediction
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Gunakan model ARIMA untuk memprediksi harga saham
        predictions_diff = model_ARIMA.predict(start=len(date_range), end=len(date_range) + (end_date - start_date).days - 1)
        predictions_diff_cumsum = predictions_diff.cumsum()
        last_value = model_ARIMA.fittedvalues[-1]
        predictions = last_value + predictions_diff_cumsum

        # Siapkan hasil
        results = {'date': date_range.strftime('%Y-%m-%d').tolist(), 'predictions': predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
