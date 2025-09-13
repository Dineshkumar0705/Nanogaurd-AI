from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import socket
import traceback
import numpy as np

# Optional ML import
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Use relative path for model (works on Render)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoencoder.h5")
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, custom_objects={"mse": MeanSquaredError()})
        print("✅ Model loaded and compiled successfully.")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Continuing without ML model.")

# Health check route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "API is running"}), 200

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Prediction unavailable."}), 500
    try:
        data = request.json
        if "input" not in data:
            return jsonify({"error": "Missing 'input' key in request JSON"}), 400

        user_input = data["input"]  # list of floats
        x = np.array(user_input, dtype=np.float32).reshape(1, -1)

        # Run prediction
        reconstructed = model.predict(x)
        mse = np.mean(np.square(x - reconstructed))

        # Simple anomaly threshold
        threshold = 0.01  # Adjust based on your dataset
        is_anomaly = mse > threshold

        return jsonify({
            "input": user_input,
            "mse": float(mse),
            "anomaly": bool(is_anomaly)
        }), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Catch-all error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Function to find a free port automatically
def find_free_port(start_port=5000):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('0.0.0.0', port)) != 0:
                return port
            port += 1

# Run the app
if __name__ == "__main__":
    port = find_free_port()
    print(f"🚀 Starting Flask app on port {port}")
    app.run(debug=True, host="0.0.0.0", port=port)
