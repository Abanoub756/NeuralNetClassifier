"""Small numerical helpers shared by the RBF and PNN implementations."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def check_X(X, name: str = "X") -> np.ndarray:
    """Validate a feature matrix and return it as a 2-D float array."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features); got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values. Impute/clean features first.")
    return arr


def check_X_y(X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate (X, y). Returns X, y, sorted unique classes and integer class indices."""
    X = check_X(X)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {y.shape}.")
    if len(y) != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)} labels.")
    classes, y_idx = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Training data must contain at least two classes.")
    return X, y, classes, y_idx.ravel()


def squared_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (len(A), len(B))."""
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
