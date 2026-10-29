"""Probabilistic Neural Network (PNN) classifier, implemented from scratch.

Layers
------
1. Pattern layer   : one Gaussian kernel per training sample x_i
                     K(x, x_i) = exp( -||x - x_i||^2 / (2 h^2) )
2. Summation layer : per class k, the Parzen estimate of the class-conditional density
                     f_k(x) = 1/(n_k (2 pi h^2)^(d/2)) * sum_{i in class k} K(x, x_i)
3. Output layer    : posterior score  P(k | x)  proportional to  prior_k * f_k(x); predict argmax.

Documented policies
-------------------
* Densities are normalised by 1/n_k and by the Gaussian constant, so they are proper densities.
* Priors: ``"empirical"`` (n_k / n, default) or ``"uniform"`` (1 / K). Priors ARE included in the
  posterior scores returned by ``predict_proba`` / used by ``predict``.
* Everything is computed in log space (log-sum-exp) so that high-dimensional kernels do not underflow.
* The bandwidth h is a hyper-parameter; it must be tuned with training data only.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from src.mathutils import check_X, check_X_y, squared_euclidean

PRIOR_POLICIES = ("empirical", "uniform")
_CHUNK = 512  # rows of X processed at once (bounds memory)


class PNN(ClassifierMixin, BaseEstimator):
    """Probabilistic Neural Network with a single Gaussian bandwidth.

    Parameters
    ----------
    bandwidth : float > 0, Gaussian kernel bandwidth h (on standardised features).
    priors : {"empirical", "uniform"}, class-prior policy.
    """

    def __init__(self, bandwidth: float = 1.0, priors: str = "empirical"):
        self.bandwidth = bandwidth
        self.priors = priors

    def _validate_params(self) -> None:
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be a finite number > 0; got {self.bandwidth!r}.")
        if self.priors not in PRIOR_POLICIES:
            raise ValueError(f"priors must be one of {PRIOR_POLICIES}; got {self.priors!r}.")

    def _check_fitted(self) -> None:
        if not hasattr(self, "patterns_"):
            raise NotFittedError("PNN is not fitted yet. Call fit(X, y) first.")

    def fit(self, X, y) -> "PNN":
        """Store the training samples (pattern layer) grouped by class."""
        self._validate_params()
        X, y, classes, y_idx = check_X_y(X, y)
        self.classes_ = classes
        self.patterns_ = [X[y_idx == k] for k in range(len(classes))]
        self.class_counts_ = np.array([len(p) for p in self.patterns_], dtype=float)
        if self.priors == "empirical":
            priors = self.class_counts_ / self.class_counts_.sum()
        else:
            priors = np.full(len(classes), 1.0 / len(classes))
        self.class_priors_ = priors
        self.log_priors_ = np.log(priors)
        self.n_features_in_ = X.shape[1]
        return self

    def _check_input(self, X) -> np.ndarray:
        self._check_fitted()
        X = check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features but the model was fitted with {self.n_features_in_}.")
        return X

    def log_class_densities(self, X) -> np.ndarray:
        """log f_k(x) for every sample and class, shape (n_samples, n_classes)."""
        X = self._check_input(X)
        h2 = float(self.bandwidth) ** 2
        const = 0.5 * self.n_features_in_ * np.log(2.0 * np.pi * h2)
        out = np.empty((X.shape[0], len(self.classes_)))
        for k, P in enumerate(self.patterns_):
            for s in range(0, X.shape[0], _CHUNK):
                sq = squared_euclidean(X[s:s + _CHUNK], P)
                out[s:s + _CHUNK, k] = logsumexp(-sq / (2.0 * h2), axis=1) - np.log(len(P)) - const
        return out

    def class_densities(self, X) -> np.ndarray:
        """Class-conditional density estimates f_k(x) (may underflow to 0 in high dimension)."""
        return np.exp(self.log_class_densities(X))

    def predict_proba(self, X) -> np.ndarray:
        """Posterior class probabilities  prior_k f_k(x) / sum_j prior_j f_j(x)."""
        joint = self.log_class_densities(X) + self.log_priors_
        return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))

    def class_scores(self, X) -> np.ndarray:
        """Class scores = posterior probabilities, shape (n_samples, n_classes)."""
        return self.predict_proba(X)

    def positive_class_score(self, X) -> np.ndarray:
        """Posterior probability of the positive (second) class, usable for ROC-AUC."""
        p = self.predict_proba(X)
        if p.shape[1] != 2:
            raise ValueError("positive_class_score is only defined for binary problems.")
        return p[:, 1]

    def predict(self, X) -> np.ndarray:
        """Predicted class labels (argmax of the posterior)."""
        joint = self.log_class_densities(X) + self.log_priors_
        return self.classes_[np.argmax(joint, axis=1)]
