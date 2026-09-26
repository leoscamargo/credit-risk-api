import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

MODELS_DIR = Path("models")
LOG_PATH = Path("logs/predictions.csv")
model = joblib.load(MODELS_DIR / "model.joblib")
metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
app = FastAPI(title="Credit Risk API", version=metadata["version"])
class Cliente(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Limite de crédito")
    EDUCATION: int = Field(..., ge=1, le=4)
    MARRIAGE: int = Field(..., ge=1, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_1: int = Field(..., ge=-2, le=9, description="Status de pagamento no mês mais recente")
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

def faixa_risco(p: float) -> str:
    if p < 0.3:
        return "BAIXO"
    if p < 0.6:
        return "MEDIO"
    return "ALTO"

def log_prediction(features: dict, proba: float) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(),
           "model_version": metadata["version"], "proba": proba, **features}
    new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")

def health():
    return {"status": "ok", "model": metadata["model"], "version": metadata["version"]}

@app.get("/model-info")

def model_info():
    return metadata

@app.post("/predict")

def predict(cliente: Cliente):
    data = cliente.model_dump()
    proba = float(model.predict_proba(pd.DataFrame([data]))[0, 1])
    log_prediction(data, proba)
    return {
        "probabilidade_default": round(proba, 4),
        "score": round((1 - proba) * 1000),
        "faixa_risco": faixa_risco(proba),
        "model_version": metadata["version"],
    }