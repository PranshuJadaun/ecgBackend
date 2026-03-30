import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 🔥 Load model once
model = load_model("ecg_sliding_model.keras")

@app.get("/")
def home():
    return {"status": "ECG API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # ===== STEP 1: Extract =====
        arr = np.array(data["data"])

        # ===== STEP 2: Validate =====
        if len(arr) != 200:
            return {"error": "Input must be 200 values"}

        # ===== STEP 3: Normalize safely =====
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return {"error": "Invalid ECG signal"}

        arr = (arr - mean) / std

        # ===== STEP 4: Reshape =====
        arr = arr.reshape(1, 200, 1)

        # ===== STEP 5: Predict =====
        pred = model.predict(arr, verbose=0)[0][0]

        # ===== STEP 6: Decision =====
        threshold = 0.4
        result = "Abnormal" if pred > threshold else "Normal"

        return {
            "result": result,
            "confidence": float(pred)
        }

    except Exception as e:
        return {"error": str(e)}