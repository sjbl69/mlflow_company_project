import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier

# Configuration

data_path = "data/raw/dataset.csv"
target = "churn"
model_path = "data/outputs/local_model.pkl"  # Modèle sauvegardé manuellement

#  Chargement du dataset

print(" Chargement du dataset...")
df = pd.read_csv(data_path)
if target not in df.columns:
    raise ValueError(f"La colonne cible '{target}' est introuvable. Colonnes disponibles : {df.columns.tolist()}")

X = df.drop(columns=[target])
y = df[target]
print(f" Données chargées ({X.shape[0]} lignes, {X.shape[1]} variables)")

# Si le modèle n’existe pas, on l’entraîne et le sauvegarde

if not os.path.exists(model_path):
    print(" Entraînement d’un modèle local GradientBoosting...")
    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    os.makedirs("data/outputs", exist_ok=True)
    joblib.dump(model, model_path)
    print(f" Modèle entraîné et sauvegardé dans {model_path}")
else:
    print(f" Chargement du modèle local depuis {model_path}...")
    model = joblib.load(model_path)
    print(" Modèle chargé avec succès.")

#  Calcul des valeurs SHAP

print(" Calcul des valeurs SHAP (TreeExplainer)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#  Graphique des importances globales

print(" Création du graphique d'importance des variables...")
output_path = "data/outputs/feature_importance.png"
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(output_path)
plt.close()
print(f" Graphique enregistré dans {output_path}")

# Rapport texte

import numpy as np
importance_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

report_path = "data/outputs/feature_importance_report.txt"
top_features = importance_df.head(10)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(" Rapport d'importance des variables (modèle local SHAP)\n")
    f.write("=" * 50 + "\n\n")
    f.write(top_features.to_string(index=False))

print(f" Rapport sauvegardé dans {report_path}")
print(" Explicabilité terminée avec succès !")
