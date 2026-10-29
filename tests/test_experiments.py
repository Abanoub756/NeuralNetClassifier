import json

import numpy as np
import pytest

from src.experiments import (best_model, make_cv_splits, predict_samples, run_experiment)


def test_heart_pipeline_end_to_end(heart_csv, small_settings):
    r = run_experiment("heart", path=heart_csv, settings=small_settings)
    assert set(r.results) == {"RBF", "PNN"} and set(r.models) == {"RBF", "PNN"}
    assert set(r.train_idx).isdisjoint(r.test_idx) and len(r.train_idx) + len(r.test_idx) == 160
    assert len(r.test_idx) == 32 and r.stratified_split
    for res in r.results.values():                                          # same test rows for both models
        assert len(res.y_pred) == len(r.test_idx) and np.all(np.isfinite(res.y_score))
        assert 0 <= res.metrics["accuracy"] <= 1 and res.train_time_s >= 0 and res.inference_ms_per_sample >= 0
        assert sum(map(sum, res.metrics["confusion_matrix"])) == len(r.test_idx)
    assert r.results["RBF"].best_params["n_centers"] in (3, 6)
    assert r.results["PNN"].best_params["bandwidth"] in (0.5, 1.0)
    assert r.results["PNN"].metrics["accuracy"] > 0.6                       # synthetic data is learnable


def test_preprocessing_fitted_on_training_rows_only(heart_csv, small_settings):
    r = run_experiment("heart", path=heart_csv, settings=small_settings)
    mean_age = r.preprocessor.named_transformers_["num"].named_steps["scale"].mean_[0]
    assert np.isclose(mean_age, r.dataset.X.iloc[r.train_idx]["age"].mean())
    assert not np.isclose(mean_age, r.dataset.X["age"].mean())


def test_student_pipeline_has_no_grade_features(student_csv, small_settings):
    r = run_experiment("student", path=student_csv, settings=small_settings)
    assert not any(c in r.dataset.X.columns for c in ("G1", "G2", "G3"))
    assert not any(n.startswith(("G1", "G2", "G3")) for n in r.feature_names)
    assert r.results["RBF"].metrics["accuracy"] > 0.5


def test_reproducible_with_fixed_seed(heart_csv, small_settings):
    a = run_experiment("heart", path=heart_csv, settings=small_settings)
    b = run_experiment("heart", path=heart_csv, settings=small_settings)
    assert np.array_equal(a.test_idx, b.test_idx)
    for n in ("RBF", "PNN"):
        assert a.results[n].best_params == b.results[n].best_params
        assert np.allclose(a.results[n].y_score, b.results[n].y_score)


def test_predict_samples_uses_trained_models(heart_csv, small_settings):
    r = run_experiment("heart", path=heart_csv, settings=small_settings)
    out = predict_samples(r, r.X_test.iloc[:5])
    for n in ("RBF", "PNN"):
        assert out[n]["pred"].shape == (5,) and out[n]["scores"].shape == (5, 2)
        assert np.array_equal(out[n]["pred"], r.results[n].y_pred[:5])        # identical to pipeline predictions
    name, why = best_model(r)
    assert name in (None, "RBF", "PNN") and why


def test_results_saved_as_json(heart_csv, small_settings, tmp_path):
    s = {**small_settings, "save_results": True, "results_dir": str(tmp_path / "out")}
    r = run_experiment("heart", path=heart_csv, settings=s)
    data = json.loads(r.results_path.read_text(encoding="utf-8"))
    assert data["dataset"] == "heart" and set(data["models"]) == {"RBF", "PNN"}


def test_invalid_settings_and_missing_file(heart_csv, small_settings, tmp_path):
    from src.data_loader import DatasetNotFoundError
    with pytest.raises(ValueError):
        run_experiment("heart", path=heart_csv, settings={**small_settings, "pnn_bandwidths": []})
    with pytest.raises(DatasetNotFoundError):
        run_experiment("heart", path=tmp_path / "missing.data", settings=small_settings)


def test_cv_splits_stratified_and_fallback():
    y = np.array([0] * 40 + [1] * 10)
    splits, strat = make_cv_splits(y, 5, 42)
    assert strat and len(splits) == 5 and all(y[va].sum() == 2 for _, va in splits)
    splits2, strat2 = make_cv_splits(np.array([0] * 9 + [1]), 5, 42)
    assert strat2 is False and len(splits2) == 5
