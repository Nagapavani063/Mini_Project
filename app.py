import pickle
import numpy as np
from flask import Flask, render_template, request

# Load your model dictionary
with open('house_pridiction.pkl','rb') as f:
    model_dict = pickle.load(f)

intercept = model_dict['intercept']
coefficients = model_dict['coefficients']  # Should have shape (15,)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Safe float conversion
        def get_float(value):
            return float(value) if value else 0.0

        # Only 15 features (remove city/country)
        features = [
            get_float(request.form.get('bedrooms')),
            get_float(request.form.get('bathrooms')),
            get_float(request.form.get('sqft_living')),
            get_float(request.form.get('sqft_lot')),
            get_float(request.form.get('floors')),
            get_float(request.form.get('waterfront')),
            get_float(request.form.get('view')),
            get_float(request.form.get('condition')),
            get_float(request.form.get('sqft_above')),
            get_float(request.form.get('sqft_basement')),
            get_float(request.form.get('yr_built')),
            get_float(request.form.get('yr_renovated')),
            get_float(request.form.get('city')),
            get_float(request.form.get('country')),
            get_float(request.form.get('year')),
            get_float(request.form.get('month')),
            get_float(request.form.get('day'))
        ]

        input_array = np.array(features)
        prediction = intercept + np.dot(coefficients, input_array)

        return render_template('index.html',
                               prediction_text=f"Predicted Price: ₹ {prediction:,.2f}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
