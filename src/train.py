import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import TARGET, clean, load_raw
from src.features import RAW_FEATURES, add_features
from src.metrics import gini, ks

MODELS_DIR = Path("models")
SEED = 42

def build_pipelines() -> dict:
    fe = ("features", FunctionTransformer(add_features))
    return {
      "logistic": Pipeline([fe, ("scaler", StandardScaler()),
                              ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "lightgbm": Pipeline([fe, ("model", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight="balanced", random_state=SEED, verbose=-1))]),
    }

def main():
    df = clean(load_raw())
    X, y = df[RAW_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk")
    results = {}
    for name, pipe in build_pipelines().items():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            m = {"auc": roc_auc_score(y_test, proba), "gini": gini(y_test, proba),
                 "ks": ks(y_test, proba)}
            mlflow.log_params({"model": name, **pipe.named_steps["model"].get_params()})
            mlflow.log_metrics(m)
            results[name] = (pipe, m)
            print(f"{name:10s} AUC={m['auc']:.4f}  Gini={m['gini']:.4f}  KS={m['ks']:.4f}")
    best = max(results, key=lambda k: results[k][1]["auc"])
    pipe, m = results[best]
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    np.save(MODELS_DIR / "reference_scores.npy", pipe.predict_proba(X_test)[:, 1])
    meta = {"model": best, "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M"),
            "metrics": {k: round(v, 4) for k, v in m.items()},
            "features": RAW_FEATURES, "n_train": len(X_train), "n_test": len(X_test)}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Melhor modelo: {best} -> models/model.joblib")

if __name__ == "__main__":
    main()