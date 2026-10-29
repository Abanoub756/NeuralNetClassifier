"""Leakage-free preprocessing.

The same pipeline is used for both models within a dataset:
  * continuous / ordinal-scale numeric features: median imputation -> standardisation
  * categorical features: most-frequent imputation -> one-hot encoding
Every step is *fitted on training data only*; test/validation data is only transformed.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    """Create an UNFITTED preprocessing pipeline (call .fit on training data only)."""
    parts = []
    if numeric:
        parts.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(numeric)))
    if categorical:
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical)))
    if not parts:
        raise ValueError("At least one numeric or categorical feature is required.")
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False, sparse_threshold=0.0)


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Apply a FITTED preprocessor and return a dense float array."""
    out = preprocessor.transform(X)
    if hasattr(out, "toarray"):
        out = out.toarray()
    return np.asarray(out, dtype=float)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Names of the encoded features produced by a fitted preprocessor."""
    return [str(n) for n in preprocessor.get_feature_names_out()]


def stratified_split_indices(y, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Positional train/test indices. Stratified when possible.

    Returns (train_idx, test_idx, was_stratified). Falls back to a plain random split
    (reported through the flag) if a class is too small to stratify.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    try:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
        return np.sort(tr), np.sort(te), True
    except ValueError:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
        return np.sort(tr), np.sort(te), False
