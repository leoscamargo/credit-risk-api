import argparse

import requests

from src.data import clean, load_raw
from src.features import PAY_COLS, RAW_FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--shift", action="store_true", help="piora os atrasos para gerar drift")
args = parser.parse_args()

df = clean(load_raw())[RAW_FEATURES].sample(args.n, random_state=7)
if args.shift:
    df[PAY_COLS] = (df[PAY_COLS] + 2).clip(upper=9)

for _, row in df.iterrows():
    payload = {k: (int(v) if k in PAY_COLS + ["EDUCATION", "MARRIAGE", "AGE"] else float(v))
               for k, v in row.items()}
    requests.post(f"{args.url}/predict", json=payload, timeout=10).raise_for_status()
print(f"{args.n} requisições enviadas para {args.url}")