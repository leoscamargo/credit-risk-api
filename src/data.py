from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/default of credit card clients.xls")
TARGET = "default"

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    return df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["ID"], errors="ignore").copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df