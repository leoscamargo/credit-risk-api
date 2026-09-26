import numpy as np
import pandas as pd

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
# SEX fica de fora de propósito: princípio da não discriminação (LGPD, art. 6º, IX)
RAW_FEATURES = (
    ["LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[RAW_FEATURES].copy()
    limit = df["LIMIT_BAL"].replace(0, np.nan)
    df["utilizacao_limite"] = (df["BILL_AMT1"] / limit).fillna(0)
    df["atraso_maximo"] = df[PAY_COLS].max(axis=1)
    df["meses_em_atraso"] = (df[PAY_COLS] > 0).sum(axis=1)
    bill_total = df[BILL_COLS].sum(axis=1).replace(0, np.nan)
    df["taxa_pagamento"] = (df[PAYAMT_COLS].sum(axis=1) / bill_total).fillna(1).clip(0, 5)
    return df