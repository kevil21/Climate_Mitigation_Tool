from flask import Flask, render_template, request, jsonify
import datetime
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model and scaler
model = load_model('model.h5')  
scaler = pickle.load(open('scaler.pkl', 'rb'))  

@app.route('/')
def index():
    # Get today's date to set it as default in the calendar
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    return render_template('index.html', today=today)

@app.route('/predict', methods=['POST'])
def predict():
    selected_date = request.form['date']
    selected_date = datetime.datetime.strptime(selected_date, '%Y-%m-%d')
    
    # Extract features for prediction
    features = {
        'Year': selected_date.year,
        'Month': selected_date.month,
        'Day': selected_date.day,
    }
    
    input_data = pd.DataFrame([features])

    # Scale the features before making a prediction
    scaled_data = scaler.transform(input_data)
    
    # Make the prediction using the model
    prediction = model.predict(scaled_data)
    
    # Return the prediction
    return jsonify({'prediction': prediction[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)