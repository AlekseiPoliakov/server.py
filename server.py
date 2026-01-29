import os
import sqlite3
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from main import MatchPredictor  # Твоя программа

app = FastAPI()

class AnalysisRequest(BaseModel):
    home_team: str
    away_team: str
    user_id: str
    # Поля для коэффициентов от пользователя
    odds_p1: float = 0.0
    odds_x: float = 0.0
    odds_p2: float = 0.0
    odds_tb25: float = 0.0
    odds_tm25: float = 0.0
    odds_btts_yes: float = 0.0

@app.post("/analyze")
async def analyze_match(data: AnalysisRequest):
    try:
        # Инициализируем твой MatchPredictor
        predictor = MatchPredictor(data_dir=".")
        
        # Вызываем расчет (Python подхватит данные)
        prediction = predictor.predict(data.home_team, data.away_team)

        return {"status": "success", "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "Rocket V5.3 API is running!"}