from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Wine Quality Prediction API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting {"features": [values...]}
    features = np.array(data["features"]).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return jsonify({"predicted_quality": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
