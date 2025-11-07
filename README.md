#  MLflow Company Project

Suivi complet d'expérimentations Machine Learning avec **MLflow**  

---
#  Contexte du projet

L’entreprise fictive pour laquelle ce projet est réalisé souhaite réduire le taux de départ de ses clients (appelé churn).
En effet, conserver un client existant coûte souvent 5 à 7 fois moins cher que d’en acquérir un nouveau.
L’objectif est donc de prédire les clients susceptibles de quitter afin d’anticiper et personnaliser les actions de fidélisation.

L’équipe Data Science de l’entreprise a donc entrepris de construire un modèle de classification supervisée capable de distinguer les clients “fidèles” des clients “à risque”.
Pour cela, plusieurs tâches ont été menées :

Génération de données clients simulées, représentant des variables réelles : âge, revenu, ancienneté, satisfaction, fréquence d’achat, contacts avec le service client, etc.

Exploration et préparation des données, afin d’assurer la qualité et la cohérence des informations.

Entraînement de plusieurs modèles de Machine Learning (régression logistique, forêt aléatoire, gradient boosting) pour comparer leurs performances.

Suivi des expérimentations avec MLflow, permettant de conserver automatiquement les hyperparamètres, métriques et versions des modèles.

Analyse d’explicabilité avec SHAP, pour comprendre les variables influentes dans les décisions du modèle.

Optimisation du modèle final selon les contraintes métier, notamment le coût associé aux erreurs de prédiction (faux positifs et faux négatifs).

# Objectif du projet

Mettre en place une pipeline de Machine Learning traçable et réplicable.

Comparer et justifier le modèle final selon des critères techniques (AUC, F1-score) et économiques (coût métier minimal).

Documenter et versionner les expérimentations à l’aide de MLflow, un outil de MLOps permettant de suivre toutes les phases d’un projet ML.

Fournir un modèle robuste, explicable et optimisé prêt à être déployé.

# Enjeux métier

Identifier les clients présentant un risque de départ élevé.

Minimiser les pertes financières en optimisant le seuil de décision.

Permettre à l’équipe marketing de concentrer les actions de fidélisation sur les clients prioritaires.

Rendre le processus de modélisation reproductible, traçable et transparent pour les futures itérations du projet.

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
