from __future__ import annotations
from typing import List, Optional
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor(numerical: List[str], categorical: List[str]) -> ColumnTransformer:
    num_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    cat_pipeline = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
    transformers = []
    if numerical:
        from sklearn.pipeline import Pipeline
        transformers.append(("num", Pipeline(num_pipeline), numerical))
    if categorical:
        from sklearn.pipeline import Pipeline
        transformers.append(("cat", Pipeline(cat_pipeline), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")
