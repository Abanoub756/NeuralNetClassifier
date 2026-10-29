"""Radial Basis Function (RBF) network classifier, implemented from scratch.

Architecture
------------
    input x --> [m Gaussian hidden units + bias] --> linear output layer (one unit per class)

Mathematics
-----------
Hidden unit j (centre c_j, width sigma_j):

    phi_j(x) = exp( -||x - c_j||^2 / (2 sigma_j^2) )

Design matrix Phi = [phi_1(x) ... phi_m(x), 1]  (last column = bias).
Targets are one-hot encoded (T, shape n x K). Output weights solve the
ridge-regularised least-squares problem (bias is NOT penalised):

    W = argmin ||Phi W - T||^2 + lambda * ||W_without_bias||^2
      = (Phi^T Phi + lambda * I') ^-1 Phi^T T

Class scores are Phi(x) W (one real number per class). The predicted class is
the argmax of the scores. Scores are *regression outputs*, not probabilities;
``predict_proba`` offers a clipped-and-normalised version for display only.

Implementation choices
----------------------
* Centres: K-Means on the TRAINING features (``select_centers``).
* Width ("global"): sigma = scale * d_max / sqrt(2 m), d_max = largest distance
  between centres (classic heuristic). Width ("local"): sigma_j = scale * mean distance
  from centre j to its ``n_neighbors`` nearest other centres.
* Everything (centres, widths, weights) uses training data only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

from src.mathutils import check_X, check_X_y, squared_euclidean

WIDTH_METHODS = ("global", "local")


def select_centers(X, n_centers: int, random_state: Optional[int] = 42) -> np.ndarray:
    """Choose RBF centres with K-Means (n_centers is clipped to the number of samples)."""
    X = check_X(X)
    k = int(min(n_centers, X.shape[0]))
    if k < 1:
        raise ValueError("n_centers must be >= 1.")
    km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
    km.fit(X)
    return km.cluster_centers_


class RBFNetwork(ClassifierMixin, BaseEstimator):
    """RBF network with K-Means centres, Gaussian units and ridge least-squares output layer.

    Parameters
    ----------
    n_centers : int, number of hidden Gaussian units (>= 1).
    width_method : {"global", "local"}, rule used to set the Gaussian widths.
    width_scale : float > 0, multiplier applied to the width produced by the rule.
    ridge : float >= 0, L2 regularisation strength lambda of the output layer.
    n_neighbors : int >= 1, neighbours used by the "local" width rule.
    random_state : int or None, seed for K-Means.
    """

    def __init__(self, n_centers: int = 20, width_method: str = "global", width_scale: float = 1.0,
                 ridge: float = 1e-2, n_neighbors: int = 2, random_state: Optional[int] = 42):
        self.n_centers = n_centers
        self.width_method = width_method
        self.width_scale = width_scale
        self.ridge = ridge
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    # ------------------------------------------------------------ validation
    def _validate_params(self) -> None:
        if isinstance(self.n_centers, bool) or not isinstance(self.n_centers, (int, np.integer)) or self.n_centers < 1:
            raise ValueError(f"n_centers must be an integer >= 1; got {self.n_centers!r}.")
        if self.width_method not in WIDTH_METHODS:
            raise ValueError(f"width_method must be one of {WIDTH_METHODS}; got {self.width_method!r}.")
        if not np.isfinite(self.width_scale) or self.width_scale <= 0:
            raise ValueError(f"width_scale must be a finite number > 0; got {self.width_scale!r}.")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError(f"ridge must be a finite number >= 0; got {self.ridge!r}.")
        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1; got {self.n_neighbors!r}.")

    def _check_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise NotFittedError("RBFNetwork is not fitted yet. Call fit(X, y) first.")

    def _check_input(self, X) -> np.ndarray:
        self._check_fitted()
        X = check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features but the model was fitted with {self.n_features_in_}.")
        return X

    # --------------------------------------------------------------- fitting
    def _estimate_widths(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        m = centers.shape[0]
        fallback = float(np.sqrt(np.mean(np.var(X, axis=0))))
        if not fallback > 0:
            fallback = 1.0
        if m == 1:
            base = np.array([fallback])
        else:
            D = np.sqrt(squared_euclidean(centers, centers))
            if self.width_method == "global":
                sigma = float(D.max()) / np.sqrt(2.0 * m)
                base = np.full(m, sigma if sigma > 1e-12 else fallback)
            else:
                k = int(min(self.n_neighbors, m - 1))
                nearest = np.sort(D, axis=1)[:, 1:k + 1].mean(axis=1)
                base = np.where(nearest > 1e-12, nearest, fallback)
        return self.width_scale * base

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        sq = squared_euclidean(X, self.centers_)
        hidden = np.exp(-sq / (2.0 * self.widths_[None, :] ** 2))
        return np.hstack([hidden, np.ones((X.shape[0], 1))])

    def fit(self, X, y, centers: Optional[np.ndarray] = None) -> "RBFNetwork":
        """Fit on training data. ``centers`` may be passed to reuse K-Means centres
        computed (on the same training data) elsewhere, e.g. during hyper-parameter search."""
        self._validate_params()
        X, y, classes, y_idx = check_X_y(X, y)
        if centers is None:
            centers = select_centers(X, self.n_centers, self.random_state)
        else:
            centers = check_X(centers, "centers")
            if centers.shape[1] != X.shape[1]:
                raise ValueError("centers must have the same number of features as X.")
        self.centers_ = centers
        self.widths_ = self._estimate_widths(X, centers)
        Phi = self._design_matrix(X)
        T = np.eye(len(classes))[y_idx]
        reg = self.ridge * np.eye(Phi.shape[1])
        reg[-1, -1] = 0.0                      # do not penalise the bias
        A, B = Phi.T @ Phi + reg, Phi.T @ T
        try:
            W = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(A, B, rcond=None)[0]
        self.weights_ = W
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.n_centers_ = centers.shape[0]
        return self

    # ------------------------------------------------------------ prediction
    def class_scores(self, X) -> np.ndarray:
        """Raw output-layer scores, shape (n_samples, n_classes)."""
        X = self._check_input(X)
        return self._design_matrix(X) @ self.weights_

    def decision_function(self, X) -> np.ndarray:
        """Binary: score_1 - score_0 (shape (n,)). Multi-class: raw class scores."""
        s = self.class_scores(X)
        return s[:, 1] - s[:, 0] if s.shape[1] == 2 else s

    def positive_class_score(self, X) -> np.ndarray:
        """Continuous score for the positive class (binary problems), usable for ROC-AUC."""
        s = self.class_scores(X)
        if s.shape[1] != 2:
            raise ValueError("positive_class_score is only defined for binary problems.")
        return s[:, 1] - s[:, 0]

    def predict_proba(self, X) -> np.ndarray:
        """Display-only pseudo-probabilities: scores clipped at 0 and normalised.
        NOT calibrated probabilities; uniform when all clipped scores are zero."""
        s = np.clip(self.class_scores(X), 0.0, None)
        tot = s.sum(axis=1, keepdims=True)
        uniform = np.full_like(s, 1.0 / s.shape[1])
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), uniform)

    def predict(self, X) -> np.ndarray:
        """Predicted class labels (argmax of the output-layer scores)."""
        scores = self.class_scores(X)          # raises NotFittedError first if unfitted
        return self.classes_[np.argmax(scores, axis=1)]
