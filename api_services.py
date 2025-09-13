from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Nanoguard AI - Cybersecurity API")

# 🔹 Load your trained ML model (adjust path if needed)
MODEL_PATH = "backend/models/anomaly_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None  # fallback if model is missing

@app.get("/")
async def root():
    return {"message": "Nanoguard AI API is running 🚀"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_logs(file: UploadFile = File(...)):
    """
    Upload a CSV log file, run anomaly detection, return results.
    """
    try:
        # Read CSV into dataframe
        df = pd.read_csv(file.file)

        if model is None:
            return JSONResponse(
                content={"error": "Model not found on server."},
                status_code=500
            )

        # Run prediction (1 = anomaly, 0 = normal)
        preds = model.predict(df)

        anomalies = int(np.sum(preds == 1))
        total = len(preds)

        return {
            "total_samples": total,
            "anomalies_detected": anomalies,
            "anomaly_ratio": round(anomalies / total, 4)
        }
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/predict")
async def predict(data: dict):
    """
    Predict on a single JSON log entry.
    Example:
    {
        "duration": 1.23,
        "protocol": 6,
        "src_bytes": 300,
        "dst_bytes": 200
    }
    """
    try:
        if model is None:
            return {"error": "Model not loaded"}

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]

        return {"prediction": int(pred), "label": "anomaly" if pred == 1 else "normal"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
