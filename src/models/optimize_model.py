import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

#  Chargement des données

data_path = "data/raw/dataset.csv"
target = "churn"

print(" Chargement du dataset...")
df = pd.read_csv(data_path)

if target not in df.columns:
    raise ValueError(f"La colonne cible '{target}' est introuvable. Colonnes disponibles : {df.columns.tolist()}")

X = df.drop(columns=[target])
y = df[target]
print(f" Données chargées ({X.shape[0]} lignes, {X.shape[1]} variables)")

# Découpage Train/Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Recherche d’hyperparamètres

print(" Optimisation des hyperparamètres...")

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4]
}

gb = GradientBoostingClassifier(random_state=42)
grid = GridSearchCV(gb, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

print(f" Meilleurs paramètres : {grid.best_params_}")
print(f" Meilleur AUC en CV : {grid.best_score_:.3f}")

best_model = grid.best_estimator_

# Évaluation sur le test set

y_proba = best_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f" AUC sur test set : {auc:.3f}")

# Optimisation du seuil de décision

print("\n Ajustement du seuil de décision selon le coût métier...")

cost_fp = 10   # coût faux positif
cost_fn = 100  # coût faux négatif

thresholds = np.linspace(0, 1, 101)
costs = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    costs.append(total_cost)

best_threshold = thresholds[np.argmin(costs)]
min_cost = np.min(costs)

print(f" Seuil optimal : {best_threshold:.2f}")
print(f" Coût total minimal : {min_cost:.2f} €")

# Visualisation du coût selon le seuil

os.makedirs("data/outputs", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, label="Coût total (€)")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Seuil optimal ({best_threshold:.2f})")
plt.xlabel("Seuil de décision")
plt.ylabel("Coût total (€)")
plt.title("Optimisation du seuil de décision (coût métier)")
plt.legend()
plt.tight_layout()
plt.savefig("data/outputs/cost_vs_threshold.png")
plt.close()

print(" Graphique enregistré dans data/outputs/cost_vs_threshold.png")

# Évaluation finale

y_pred_final = (y_proba >= best_threshold).astype(int)
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)

print("\n Résumé final :")
print(f"AUC : {auc:.3f}")
print(f"Précision : {precision:.3f}")
print(f"Rappel : {recall:.3f}")
print(f"F1-score : {f1:.3f}")

print("\n Étape 4 terminée : modèle optimisé et seuil ajusté !")
