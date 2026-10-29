import numpy as np
import pytest

from src.data_loader import load_heart, load_student
from src.preprocessing import build_preprocessor, get_feature_names, stratified_split_indices, transform_features


def test_preprocessor_fit_on_train_only_and_dimensions(heart_csv):
    ds = load_heart(heart_csv)
    tr, te, strat = stratified_split_indices(ds.y, 0.2, 42)
    pre = build_preprocessor(ds.numeric_features, ds.categorical_features).fit(ds.X.iloc[tr])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], ds.X.iloc[tr]["age"].mean())            # fitted on train rows only
    assert not np.isclose(scaler.mean_[0], ds.X["age"].mean())                   # not on all rows
    Xt_tr, Xt_te = transform_features(pre, ds.X.iloc[tr]), transform_features(pre, ds.X.iloc[te])
    assert Xt_tr.shape[1] == Xt_te.shape[1] == len(get_feature_names(pre))
    assert np.all(np.isfinite(Xt_tr)) and np.all(np.isfinite(Xt_te))             # NaNs imputed
    assert np.allclose(Xt_tr[:, :5].mean(axis=0), 0, atol=1e-8)                  # scaled numeric block


def test_unseen_category_in_test_does_not_crash(student_csv):
    ds = load_student(student_csv)
    pre = build_preprocessor(ds.numeric_features, ds.categorical_features).fit(ds.X.iloc[:100])
    X_new = ds.X.iloc[100:105].copy()
    X_new["Mjob"] = "astronaut"
    out = transform_features(pre, X_new)
    assert out.shape == (5, len(get_feature_names(pre)))


def test_student_features_have_no_grades(student_csv):
    ds = load_student(student_csv)
    pre = build_preprocessor(ds.numeric_features, ds.categorical_features).fit(ds.X)
    assert not any(n.startswith(("G1", "G2", "G3")) for n in get_feature_names(pre))


def test_split_is_stratified_disjoint_reproducible():
    y = np.array([0] * 80 + [1] * 20)
    tr, te, strat = stratified_split_indices(y, 0.2, 42)
    assert strat and len(te) == 20 and len(tr) == 80
    assert set(tr).isdisjoint(te) and set(tr) | set(te) == set(range(100))
    assert y[te].sum() == 4                                                       # class ratio kept
    tr2, te2, _ = stratified_split_indices(y, 0.2, 42)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)


def test_split_falls_back_when_class_too_small():
    y = np.array([0] * 20 + [1])
    _, _, strat = stratified_split_indices(y, 0.2, 42)
    assert strat is False


def test_empty_feature_lists_rejected():
    with pytest.raises(ValueError):
        build_preprocessor([], [])
