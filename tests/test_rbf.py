import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.rbf import RBFNetwork, select_centers


def test_fit_predict_shapes_and_accuracy(blobs):
    (Xtr, ytr), (Xte, yte) = blobs
    m = RBFNetwork(n_centers=6, random_state=0).fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == yte.shape and (pred == yte).mean() > 0.95
    assert m.class_scores(Xte).shape == (len(Xte), 2)
    assert np.all(np.isfinite(m.class_scores(Xte)))
    assert m.centers_.shape == (6, 3) and m.widths_.shape == (6,) and np.all(m.widths_ > 0)
    assert m.positive_class_score(Xte).shape == (len(Xte),)


def test_pseudo_probabilities_valid(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    p = RBFNetwork(n_centers=4).fit(Xtr, ytr).predict_proba(Xte)
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1), 1)


def test_reproducible(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    a = RBFNetwork(n_centers=5, random_state=3).fit(Xtr, ytr).class_scores(Xte)
    b = RBFNetwork(n_centers=5, random_state=3).fit(Xtr, ytr).class_scores(Xte)
    assert np.array_equal(a, b)


def test_local_width_and_single_center_and_too_many_centers(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    assert RBFNetwork(n_centers=5, width_method="local").fit(Xtr, ytr).predict(Xte).shape == (len(Xte),)
    assert np.all(RBFNetwork(n_centers=1).fit(Xtr, ytr).widths_ > 0)
    idx = np.arange(0, len(Xtr), 12)                         # 10 samples, both classes
    assert RBFNetwork(n_centers=10_000).fit(Xtr[idx], ytr[idx]).n_centers_ == len(idx)


def test_precomputed_centers_match_internal_kmeans(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    c = select_centers(Xtr, 5, 7)
    a = RBFNetwork(n_centers=5, random_state=7).fit(Xtr, ytr, centers=c).class_scores(Xte)
    b = RBFNetwork(n_centers=5, random_state=7).fit(Xtr, ytr).class_scores(Xte)
    assert np.allclose(a, b)


@pytest.mark.parametrize("kwargs", [{"n_centers": 0}, {"n_centers": 2.5}, {"width_method": "x"}, {"width_scale": 0},
                                    {"width_scale": -1}, {"ridge": -0.1}, {"n_neighbors": 0}])
def test_invalid_parameters(blobs, kwargs):
    (Xtr, ytr), _ = blobs
    with pytest.raises(ValueError):
        RBFNetwork(**kwargs).fit(Xtr, ytr)


def test_input_validation(blobs):
    (Xtr, ytr), (Xte, _) = blobs
    with pytest.raises(NotFittedError):
        RBFNetwork().predict(Xte)
    m = RBFNetwork(n_centers=3).fit(Xtr, ytr)
    with pytest.raises(ValueError):
        m.predict(Xte[:, :2])
    with pytest.raises(ValueError):
        m.predict(Xte[0])                                  # 1-D
    bad = Xte.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.predict(bad)
    with pytest.raises(ValueError):
        RBFNetwork().fit(Xtr, ytr[:-1])
    with pytest.raises(ValueError):
        RBFNetwork().fit(Xtr, np.zeros(len(Xtr)))          # one class only
