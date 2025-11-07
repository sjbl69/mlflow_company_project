import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, f1_score
import numpy as np
import pandas as pd

# Charger les données
df = pd.read_csv("data/raw/dataset.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

# Gérer le déséquilibre entre les classes
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {0: weights[0], 1: weights[1]}
print("Poids de classes :", class_weights)

# Définir le modèle
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

# Validation croisée stratifiée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score)

mlflow.set_experiment("OC-MLflow-Churn")

with mlflow.start_run(run_name="GradientBoosting_CV"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    mlflow.log_metric("mean_f1", mean_score)
    mlflow.log_metric("std_f1", std_score)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f" Validation croisée terminée — F1 moyen : {mean_score:.3f} ± {std_score:.3f}")
