from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
CLIENTE = {
    "LIMIT_BAL": 50000, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
    "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_ok():
    r = client.post("/predict", json=CLIENTE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probabilidade_default"] <= 1
    assert 0 <= body["score"] <= 1000
    assert body["faixa_risco"] in {"BAIXO", "MEDIO", "ALTO"}
    
def test_predict_invalido():
    r = client.post("/predict", json={**CLIENTE, "AGE": 10})
    assert r.status_code == 422