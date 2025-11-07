import pandas as pd
import numpy as np
import os

# Créer le dossier data/raw s'il n'existe pas
os.makedirs("data/raw", exist_ok=True)

# Nombre d'exemples
n = 2000
np.random.seed(42)

# Génération de données clients
ages = np.random.randint(18, 70, n)
revenus = np.random.normal(3000, 800, n).round(0)
anciennete = np.random.randint(1, 10, n)
satisfaction = np.clip(np.random.normal(3.5, 1.0, n), 1, 5).round(1)
achats_moyens = np.random.normal(80, 25, n).round(2)
contact_service_client = np.random.randint(0, 10, n)

# Probabilité de churn (client qui quitte)
prob_churn = (
    0.3 * (5 - satisfaction) / 5
    + 0.3 * (1 - anciennete / 10)
    + 0.2 * (contact_service_client / 10)
    + 0.2 * (1 - revenus / revenus.max())
)
churn = np.random.binomial(1, np.clip(prob_churn, 0, 1))

# Création du DataFrame
df = pd.DataFrame({
    "age": ages,
    "revenu": revenus,
    "anciennete": anciennete,
    "satisfaction": satisfaction,
    "achats_moyens": achats_moyens,
    "contact_service_client": contact_service_client,
    "churn": churn
})

df.to_csv("data/raw/dataset.csv", index=False)
print(" Dataset OC généré avec succès :", df.shape)
print(df.head())
