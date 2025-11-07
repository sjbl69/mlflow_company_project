from __future__ import annotations
import argparse, pandas as pd, mlflow

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_uri", required=True, help="ex: runs:/<run_id>/model ou models:/<name>/Production")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default="data/processed/predictions.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    model = mlflow.pyfunc.load_model(args.model_uri)
    df = pd.read_csv(args.input_csv)
    preds = model.predict(df)
    out = df.copy()
    out["proba_positive"] = preds
    out.to_csv(args.output_csv, index=False)
    print(f"Predictions écrites: {args.output_csv}")

if __name__ == "__main__":
    main()
