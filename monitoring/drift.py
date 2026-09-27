from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import psi

reference = np.load(Path("models/reference_scores.npy"))
logs = pd.read_csv(Path("logs/predictions.csv"))
valor = psi(reference, logs["proba"].values)
status = "ESTÁVEL" if valor < 0.1 else "ATENÇÃO" if valor < 0.25 else "DRIFT - reavaliar modelo"
print(f"Predições analisadas: {len(logs)}")
print(f"PSI = {valor:.4f} -> {status}")