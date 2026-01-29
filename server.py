import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from main import MatchPredictor  # Твоя программа

app = FastAPI()

class AnalysisRequest(BaseModel):
    home_team: str
    away_team: str
    user_id: str

@app.post("/analyze")
async def analyze_match(data: AnalysisRequest):
    try:
        # 1. Запускаем твой расчет из main.py
        predictor = MatchPredictor()
        # Вызываем метод из твоего класса
        prediction = predictor.predict(data.home_team, data.away_team)
        
        # 2. Возвращаем результат в приложение
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Rocket V5.3 API is running!"}