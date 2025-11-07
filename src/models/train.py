from __future__ import annotations
import argparse, json, os
import yaml, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.io import read_csv, ensure_dir
from src.features.transformers import build_preprocessor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    return ap.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_model(cfg):
    algo = cfg["model"]["algorithm"]
    if algo == "logreg":
        p = cfg["model"]["logreg"]
        return LogisticRegression(C=p["C"], penalty=p["penalty"], max_iter=p["max_iter"], class_weight=p["class_weight"], n_jobs=None, solver="lbfgs")
    elif algo == "rf":
        p = cfg["model"]["rf"]
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            class_weight=p["class_weight"],
            n_jobs=-1,
            random_state=cfg["random_state"],
        )
    elif algo == "xgb":
        if not HAS_XGB:
            raise RuntimeError("XGBoost non installé. Changez l'algo ou installez xgboost.")
        p = cfg["model"]["xgb"]
        return XGBClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"],
            objective="binary:logistic",
            eval_metric=p.get("eval_metric","aucpr"),
            random_state=cfg["random_state"],
            n_jobs=-1
        )
    else:
        raise ValueError(f"Algorithme inconnu: {algo}")

def choose_threshold(y_true, y_proba, cost_fp: float, cost_fn: float):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    # On recherche le seuil qui minimise le coût total attendu
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost

def plot_and_log(fig, name):
    path = f"artifacts/{name}.png"
    ensure_dir("artifacts")
    fig.savefig(path, bbox_inches="tight")
    mlflow.log_artifact(path)
    plt.close(fig)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    mlflow.set_experiment(cfg["experiment_name"])
    run_name = cfg.get("run_name","run")
    with mlflow.start_run(run_name=run_name) as run:
        # Log params
        mlflow.log_params({
            "algorithm": cfg["model"]["algorithm"],
            "random_state": cfg["random_state"],
            "test_size": cfg["test_size"],
            "val_size": cfg["val_size"],
            "cv_folds": cfg["cv_folds"]
        })
        mlflow.log_dict(cfg, "config_used.yaml")

        # Load data
        df = read_csv(cfg["data"]["raw_csv"])
        target = cfg["target"]
        y = df[target].values
        X = df.drop(columns=[target])

        id_col = cfg.get("id_column")
        if id_col and id_col in X.columns:
            X = X.drop(columns=[id_col])

        cat = cfg.get("categorical", [])
        num = cfg.get("numerical", [])
        preproc = build_preprocessor(num, cat)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=cfg["val_size"], random_state=cfg["random_state"], stratify=y_temp)

        model = pick_model(cfg)

        # Pipeline
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("preproc", preproc), ("model", model)])

        # CV oof predictions (sur train) pour AUC/PR robustes
        skf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
        y_train_proba = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")[:,1]

        mlflow.log_metric("auc_roc_train_oof", roc_auc_score(y_train, y_train_proba))
        mlflow.log_metric("auc_pr_train_oof", average_precision_score(y_train, y_train_proba))

        # Fit sur train+val puis évaluation sur test
        pipe.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))
        y_test_proba = pipe.predict_proba(X_test)[:,1]

        auc_roc = roc_auc_score(y_test, y_test_proba)
        auc_pr = average_precision_score(y_test, y_test_proba)
        mlflow.log_metric("auc_roc_test", auc_roc)
        mlflow.log_metric("auc_pr_test", auc_pr)

        # Seuil basé coûts métier
        cfp = float(cfg["costs"]["false_positive"])
        cfn = float(cfg["costs"]["false_negative"])
        best_t, best_cost = choose_threshold(y_test, y_test_proba, cfp, cfn)
        y_pred = (y_test_proba >= best_t).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics @threshold
        tn, fp, fn, tp = cm.ravel()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(tn + fp + fn + tp, 1)
        mlflow.log_metrics({
            "threshold_optimal": best_t,
            "cost_total_test": best_cost,
            "precision_test": precision,
            "recall_test": recall,
            "accuracy_test": accuracy
        })

        # Courbes
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc_roc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC (test)")
        plt.legend()
        plot_and_log(fig, "roc_test")

        prec, rec, thr = precision_recall_curve(y_test, y_test_proba)
        fig = plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR (test)")
        plot_and_log(fig, "pr_test")

        # Matrice de confusion
        import seaborn as sns
        fig = plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion matrix @ t={best_t:.2f}")
        plot_and_log(fig, "confusion_matrix_test")

        # Log model
        # signature & input example
        import mlflow.models.signature as sig
        from mlflow.types.schema import Schema, ColSpec
        cols = [ColSpec("double", c) for c in num] + [ColSpec("string", c) for c in cat]
        input_schema = Schema(cols) if cols else None
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X.head(3).to_dict(orient="list"),
            signature=sig.infer_signature(X.head(3), pipe.predict_proba(X.head(3))[:,1]),
        )
        print(f"Run ID: {run.info.run_id}")
        print("Terminé.")

if __name__ == "__main__":
    main()
