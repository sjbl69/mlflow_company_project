#  MLflow Company Project

Suivi complet d'expérimentations Machine Learning avec **MLflow**  

---

##  Étapes du projet

1️⃣ **Génération du dataset client**  
→ Création de données simulées pour la fidélisation et le churn client.

2️⃣ **Entraînement et suivi MLflow**  
→ Enregistrement automatique des expériences, hyperparamètres et métriques.

3️⃣ **Explicabilité avec SHAP**  
→ Analyse des variables clés influençant la prédiction de churn.

4️⃣ **Optimisation et seuil métier**  
→ Ajustement des hyperparamètres et du seuil de décision pour minimiser le coût métier.

---

##  Résultats principaux

| Élément | Valeur |
|----------|--------|
| **Modèle retenu** | GradientBoostingClassifier |
| **AUC (test)** | 0.613 |
| **F1-score** | 0.590 |
| **Seuil optimal** | 0.20 |
| **Coût métier minimal** | 2 280 € |
| **Hyperparamètres optimaux** | learning_rate = 0.05, max_depth = 2, n_estimators = 100 |

 Le modèle détecte efficacement les clients à risque de départ (*rappel = 1.00*) tout en limitant les coûts d’alerte inutiles.

---

##  Installation

```bash
# Cloner le projet
git clone https://github.com/sjbl69/mlflow_company_project.git
cd mlflow_company_project

# Créer et activer un environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

mlflow_company_project/
│
├── data/
│   ├── raw/                # Données brutes simulées
│   ├── outputs/            # Graphiques et rapports générés
│
├── src/
│   ├── models/             # Scripts d'entraînement, d'explicabilité, d'optimisation
│   ├── utils/              # Fonctions utilitaires (lecture, écriture, etc.)
│
├── config.yaml             # Configuration du projet
├── environment.yml          # Dépendances Conda
├── requirements.txt         # Dépendances pip
├── README.md
└── Makefile

🧰 Outils utilisés

Python 3.12

scikit-learn

MLflow

SHAP

Matplotlib

Pandas / NumPy
