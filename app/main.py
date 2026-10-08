from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import httpx
import os

app = FastAPI(title="Crop Recommendation System with Auto-Weather")

# Path setup for model and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Set your OpenWeatherMap API Key here or via environment variable
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# 1. Serve Frontend UI
@app.get("/", response_class=HTMLResponse)
def render_frontend(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

# 2. Automated Weather Capture Endpoint
@app.get("/weather")
async def get_weather(city: str):
    if WEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
        raise HTTPException(status_code=400, detail="OpenWeather API key is not configured.")
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="City not found or weather service unavailable.")
        
        data = response.json()
        return {
            "city_name": data["name"],
            "temperature": round(data["main"]["temp"], 2),
            "humidity": round(data["main"]["humidity"], 2)
        }

# 3. Model Prediction Endpoint
@app.post("/predict")
def predict_crop(data: CropInput):
    if model is None:
        raise HTTPException(status_code=500, detail="ML model file missing.")
    
    features = np.array([[
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]])

    prediction = model.predict(features)[0]
    confidence = float(np.max(model.predict_proba(features)[0])) if hasattr(model, "predict_proba") else None

    return {
        "recommended_crop": str(prediction),
        "confidence_score": round(confidence, 4) if confidence else None
    }