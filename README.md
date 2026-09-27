# Credit Risk API

[![CI](https://github.com/leoscamargo/credit-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/leoscamargo/credit-risk-api/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Docker](https://img.shields.io/badge/Docker-container-2496ED)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900)

End-to-end machine learning project that predicts the **probability of default (PD)** of credit card
customers and serves it through a REST API running in a Docker container on AWS, with CI on
GitHub Actions, experiment tracking in MLflow and drift monitoring with PSI.

**Live API docs:** http://98.92.172.24:8000/docs
<sub>(demo instance — it may be offline outside of review periods)</sub>

---

## Business problem

In credit concession, knowing which customers are likely to miss next month's payment allows the
business to adjust credit limits, pricing and collection strategy before losses happen.

For each customer, the API returns:

- **Probability of default** (0 to 1)
- **Score** from 0 (highest risk) to 1000 (lowest risk)
- **Risk band**: `BAIXO` (low), `MEDIO` (medium) or `ALTO` (high)
- **Model version**, so every prediction is traceable

## Results

Evaluated on a stratified hold-out set of 6,000 customers.

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic Regression (baseline) | 0.744 | 0.488 | 0.388 |
| **LightGBM (production)** | **0.777** | **0.554** | **0.424** |

- **AUC 0.777**: given one defaulter and one non-defaulter, the model ranks the defaulter as
  riskier ~78% of the time.
- **KS 0.424**: strong separation between good and bad payers (above 0.40 is considered good
  in credit scoring).
- LightGBM beats the interpretable baseline by +6.6 Gini points by capturing non-linear
  effects, such as risk jumping sharply after two months of delay.

### Experiment tracking (MLflow)

![MLflow runs](docs/img/captura-runs.jpeg)

![Model comparison](docs/img/compare-runs.jpeg)

## Key findings from the EDA

Full analysis in [`notebooks/`](notebooks/).

- **Recent payment behavior is the strongest signal.** Default rate goes from ~13% for customers
  who paid on time last month to ~34% with one month of delay and ~69% with two months.
- **Frequency matters, not only recency.** The more months a customer was late in the last six,
  the higher the default rate — this motivated the `meses_em_atraso` feature.
- **Trend matters.** Customers who *started* being late recently behave differently from those
  who were late in the past and recovered.
- **Credit limit utilization** adds signal on top of payment history.

## Architecture

```
UCI dataset ─► EDA notebook ─► train.py ─► MLflow (experiments)
                                   │
                                   ▼
                           models/model.joblib
                                   │
GitHub ─► GitHub Actions (lint + tests + Docker build)
   │
   └─► AWS EC2 ─► Docker ─► FastAPI  /predict  /health  /model-info
                                   │
                                   ▼
                       logs/predictions.csv ─► PSI drift report
```

## Technical decisions

- **Feature engineering inside the scikit-learn `Pipeline`**: training and serving run the exact
  same transformations, avoiding training–serving skew.
- **Interpretable baseline first**: logistic regression is the historical standard for credit
  scorecards and sets the bar the complex model has to beat.
- **Credit metrics (KS, Gini) in addition to AUC**: accuracy is misleading on an imbalanced target
  (~22% default rate).
- **`class_weight="balanced"` + stratified split with fixed seed**: handles class imbalance and
  keeps results reproducible.
- **`SEX` removed from the model**: non-discrimination principle of Brazil's data protection law
  (LGPD, art. 6, IX).
- **Input validation with Pydantic**: invalid requests are rejected with HTTP 422.
- **Every prediction is logged** with timestamp and model version, which feeds drift monitoring.

## API usage

![API documentation (Swagger)](docs/img/swagger.png)

**Request**

```bash
curl -X POST http://98.92.172.24:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
       "PAY_1": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
       "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 45000,
       "BILL_AMT4": 30000, "BILL_AMT5": 28000, "BILL_AMT6": 25000,
       "PAY_AMT1": 0, "PAY_AMT2": 1000, "PAY_AMT3": 1500,
       "PAY_AMT4": 1500, "PAY_AMT5": 1500, "PAY_AMT6": 1500}'
```

**Response**

```json
{
  "probabilidade_default": 0.8711,
  "score": 129,
  "faixa_risco": "ALTO",
  "model_version": "202609260437"
}
```

A customer with two months of recent delay, a bill close to the credit limit and no payment last
month is correctly classified as high risk.

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Probability of default, score and risk band |
| `/health` | GET | Service status and model version |
| `/model-info` | GET | Model metadata and test metrics |
| `/docs` | GET | Interactive documentation (Swagger) |

## Monitoring

Credit models degrade when the customer base changes (e.g. an economic downturn). The
**Population Stability Index (PSI)** compares the score distribution in production with the
training reference, flagging shifts before actual defaults are observed (which takes 30–90 days).

| PSI | Status |
|---|---|
| < 0.10 | Stable |
| 0.10 – 0.25 | Attention |
| > 0.25 | Drift — re-evaluate the model |

```bash
# Send real customers to the API (use --shift to simulate a crisis scenario)
python -m monitoring.simulate_traffic --url http://localhost:8000 --n 300
python -m monitoring.drift
```

## Project structure

```
credit-risk-api/
├── .github/workflows/ci.yml   # CI: lint, tests and Docker build
├── app/main.py                # FastAPI service
├── src/
│   ├── data.py                # loading and cleaning
│   ├── features.py            # feature engineering
│   ├── metrics.py             # KS, Gini, PSI
│   └── train.py               # training + MLflow tracking
├── monitoring/
│   ├── drift.py               # PSI report
│   └── simulate_traffic.py    # traffic simulator
├── notebooks/                 # exploratory data analysis
├── models/                    # trained model, metadata, reference scores
├── tests/test_api.py
└── Dockerfile
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download the dataset into data/raw/ (see Data section)
python -m src.train                # trains, logs to MLflow, saves the best model
mlflow ui --backend-store-uri sqlite:///mlflow.db

uvicorn app.main:app --reload      # http://localhost:8000/docs
pytest -q
ruff check src app tests monitoring
```

**With Docker**

```bash
docker build -t credit-risk-api .
docker run -d --name api -p 8000:8000 credit-risk-api
```

## Deployment

The API runs on an **AWS EC2** instance (Amazon Linux 2023, t3.micro) inside a Docker container
with `--restart unless-stopped`, and prediction logs are persisted through a mounted volume.

## Next steps

- Explainability with **SHAP** to justify individual decisions to customers and regulators
- Model registry with approval workflow and automated deployment (CD) via Amazon ECR
- Centralized logs in S3 / CloudWatch with scheduled drift alerts
- API authentication and HTTPS
- Out-of-time validation, hyperparameter tuning and probability calibration

## Tech stack

Python · pandas · scikit-learn · LightGBM · MLflow · FastAPI · Pydantic · pytest · Ruff ·
Docker · GitHub Actions · AWS EC2

## Data

Yeh, I. C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
30,000 credit card customers with six months of payment history.
