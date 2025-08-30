import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import xgboost as xgb
from contextlib import asynccontextmanager

# --- 1. Lifespan Management (Startup Logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- API Starting Up ---")
    try:
        app.state.model = xgb.XGBClassifier()
        app.state.model.load_model("disaster_model.json")
        app.state.country_encoder = joblib.load("country_encoder.joblib")
        app.state.region_encoder = joblib.load("region_encoder.joblib")
        app.state.continent_encoder = joblib.load("continent_encoder.joblib")
        app.state.disaster_encoder = joblib.load("disaster_encoder.joblib")
        print("✅ Predictive model and encoders loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: {e}. Please run the corrected train_model.py first.")
        exit()
    yield
    print("--- API Shutting Down ---")

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="Truly Predictive Natural Disaster API",
    description="Calculates disaster risk based on location and time.",
    version="5.0.0",
    lifespan=lifespan
)

# --- 3. Define the User Input Model ---
class PredictionInput(BaseModel):
    year: int = Field(..., example=2025)
    country: str = Field(..., example="Bangladesh")
    region: str = Field(..., example="Southern Asia")
    continent: str = Field(..., example="Asia")
    month: int = Field(..., gt=0, lt=13, example=9)
    day: int = Field(..., gt=0, lt=32, example=29)

# --- 4. Create the Prediction Endpoint ---
@app.post("/predict/risk")
async def predict_risk(data: PredictionInput):
    # --- Data Validation ---
    if data.country not in app.state.country_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"Country '{data.country}' not found.")
    if data.region not in app.state.region_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"Region '{data.region}' not found.")
    if data.continent not in app.state.continent_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"Continent '{data.continent}' not found.")

    # --- Feature Engineering ---
    country_encoded = app.state.country_encoder.transform([data.country])[0]
    region_encoded = app.state.region_encoder.transform([data.region])[0]
    continent_encoded = app.state.continent_encoder.transform([data.continent])[0]
    month_sin = np.sin(2 * np.pi * data.month/12)
    month_cos = np.cos(2 * np.pi * data.month/12)
    day_sin = np.sin(2 * np.pi * data.day/31)
    day_cos = np.cos(2 * np.pi * data.day/31)
    
    # --- Prediction ---
    input_data = np.array([[
        data.year, country_encoded, region_encoded, continent_encoded,
        month_sin, month_cos, day_sin, day_cos
    ]])
    
    probabilities = app.state.model.predict_proba(input_data)[0]

    # --- Format and Return Response ---
    disaster_probabilities = {}
    for i, prob in enumerate(probabilities):
        disaster_type = app.state.disaster_encoder.inverse_transform([i])[0]
        disaster_probabilities[disaster_type] = f"{prob*100:.2f}%"

    return disaster_probabilities

