from __future__ import annotations
import argparse, yaml, pandas as pd, numpy as np, mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import load
from src.utils.io import read_csv
from src.features.transformers import build_preprocessor
from src.models.train import pick_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    df = read_csv(cfg["data"]["raw_csv"])
    y = df[cfg["target"]].values
    X = df.drop(columns=[cfg["target"]])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=cfg["val_size"], random_state=cfg["random_state"], stratify=y_temp)

    cat = cfg.get("categorical", [])
    num = cfg.get("numerical", [])
    preproc = build_preprocessor(num, cat)
    model = pick_model(cfg)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("preproc", preproc), ("model", model)])
    pipe.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
