# Projet ML d'entreprise avec MLflow (Classification binaire)

Ce dépôt est un **starter kit prêt à l'emploi** pour mener un projet de classification binaire en entreprise
avec **MLflow** : suivi d'expériences, registry de modèles et déploiement.

## ⚙️ Démarrage rapide

1) Installez l'environnement :

```bash
# Option conda
conda env create -f environment.yml
conda activate mlops-mlflow

# Ou via pip
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

2) Renseignez la config dans `config.yaml` (chemin du CSV, nom de la cible binaire, features...).
3) Déposez vos données dans `data/raw/` et mettez à jour `conf/business_context.md` et `conf/data_dictionary.csv`.
4) (Optionnel) Lancez un serveur MLflow local (voir `mlflow_server/docker-compose.yml`) ou utilisez le tracking local par défaut.
5) Exécutez un entraînement :

```bash
make train
# ou
python -m src.models.train --config config.yaml
```

6) Évaluez/comparez :

```bash
make evaluate
```

7) Enregistrez dans le **Model Registry** (si vous utilisez un serveur MLflow) :

```bash
python -m src.scripts.register_model --run_id <RUN_ID> --name <MODEL_NAME>
```

8) Servez le modèle :

```bash
make serve  # mlflow models serve
```

## 📁 Arborescence

```
.
├── README.md
├── .gitignore
├── Makefile
├── requirements.txt
├── environment.yml
├── config.yaml
├── .env.example
├── conf/
│   ├── business_context.md
│   ├── privacy_statement.md
│   └── data_dictionary.csv
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
├── mlflow_server/
│   └── docker-compose.yml
└── src/
    ├── utils/io.py
    ├── features/transformers.py
    ├── models/train.py
    ├── models/evaluate.py
    ├── models/predict.py
    └── scripts/register_model.py
```

## 🧪 Bonnes pratiques incluses

- Suivi complet avec MLflow (params, metrics, artefacts, modèle).
- Séparation train/val/test, **cross-validation** et **reproductibilité** (seeds).
- Gestion des données manquantes, encodage catégoriel, standardisation numérique.
- **Seuil de décision** optimisé selon vos **coûts métier** (FN vs FP).
- **Courbes ROC/PR**, matrice de confusion et rapport de classification loggés dans MLflow.
- Support de plusieurs algorithmes (LogReg, RandomForest, XGBoost si installé).
- Enregistrement du **signature schema** pour la sérialisation MLflow.
- Scripts pour **Model Registry** et **serving**.

## 🔒 Confidentialité

Voir `conf/privacy_statement.md`. **Ne poussez jamais de données sensibles** dans le dépôt distant.
