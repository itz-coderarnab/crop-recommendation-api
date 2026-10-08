import os
import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Crop Recommendation API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

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


# ==========================================
# 1. API ROUTES MUST COME FIRST
# ==========================================

@app.get("/weather", response_class=JSONResponse)
async def get_weather(city: str = Query(..., min_length=1)):
    if not WEATHER_API_KEY or WEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
        raise HTTPException(status_code=400, detail="OpenWeather API key is not configured.")

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=8.0)
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Unable to connect to OpenWeather service.")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"City '{city}' not found.")

        data = response.json()
        return {
            "city_name": data.get("name", city),
            "temperature": round(data["main"]["temp"], 2),
            "humidity": round(data["main"]["humidity"], 2)
        }


@app.post("/predict", response_class=JSONResponse)
def predict_crop(data: CropInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file missing.")

    features = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    prediction = model.predict(features)[0]
    confidence = float(np.max(model.predict_proba(features)[0])) if hasattr(model, "predict_proba") else None

    return {
        "recommended_crop": str(prediction),
        "confidence_score": round(confidence, 4) if confidence else None
    }


# ==========================================
# 2. HTML FRONTEND ROUTE MUST COME LAST
# ==========================================

@app.get("/", response_class=HTMLResponse)
def render_frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")