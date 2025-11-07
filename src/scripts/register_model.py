from __future__ import annotations
import argparse, mlflow

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--name", required=True, help="Nom du modèle dans le registry")
    return ap.parse_args()

def main():
    args = parse_args()
    result = mlflow.register_model(f"runs:/{args.run_id}/model", args.name)
    print(f"Version créée: name={result.name}, version={result.version}")

if __name__ == "__main__":
    main()
