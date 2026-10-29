import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.pnn import PNN


def test_fit_predict_and_shapes(blobs):
    (Xtr, ytr), (Xte, yte) = blobs
    m = PNN(bandwidth=1.0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == yte.shape and (pred == yte).mean() > 0.95
    p = m.predict_proba(Xte)
    assert p.shape == (len(Xte), 2) and np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1)
    assert m.positive_class_score(Xte).shape == (len(Xte),)
    assert np.array_equal(m.classes_[np.argmax(p, axis=1)], pred)


def test_density_matches_manual_formula(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    h = 1.3
    m = PNN(bandwidth=h).fit(Xtr, ytr)
    x = Xte[:2]
    d = Xtr.shape[1]
    for k in (0, 1):
        P = Xtr[ytr == k]
        manual = np.array([np.mean(np.exp(-np.sum((P - xi) ** 2, axis=1) / (2 * h * h))) / (2 * np.pi * h * h) ** (d / 2) for xi in x])
        assert np.allclose(m.class_densities(x)[:, k], manual, rtol=1e-8)


def test_priors_policy(blobs):
    (Xtr, ytr), _ = blobs
    idx = np.r_[np.where(ytr == 0)[0][:50], np.where(ytr == 1)[0][:10]]
    emp = PNN(priors="empirical").fit(Xtr[idx], ytr[idx])
    uni = PNN(priors="uniform").fit(Xtr[idx], ytr[idx])
    assert np.allclose(emp.class_priors_, [50 / 60, 10 / 60]) and np.allclose(uni.class_priors_, [0.5, 0.5])


def test_extreme_bandwidth_is_numerically_stable(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    for h in (1e-3, 1e3):
        p = PNN(bandwidth=h).fit(Xtr, ytr).predict_proba(Xte)
        assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1)


def test_reproducible_and_validation(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    assert np.array_equal(PNN(0.8).fit(Xtr, ytr).predict_proba(Xte), PNN(0.8).fit(Xtr, ytr).predict_proba(Xte))
    for kw in ({"bandwidth": 0}, {"bandwidth": -1}, {"bandwidth": float("nan")}, {"priors": "bad"}):
        with pytest.raises(ValueError):
            PNN(**kw).fit(Xtr, ytr)
    with pytest.raises(NotFittedError):
        PNN().predict(Xte)
    with pytest.raises(ValueError):
        PNN().fit(Xtr, np.ones(len(Xtr)))
    with pytest.raises(ValueError):
        PNN().fit(Xtr, ytr).predict(Xte[:, :2])
