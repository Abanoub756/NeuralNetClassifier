import numpy as np
import pytest

from src.evaluation import (compute_metrics, cv_score, measure_inference_time, roc_curve_points, time_call,
                            wilson_interval)


def test_known_metric_values():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    score = np.array([.9, .8, .4, .2, .1, .3, .6, .05])
    m = compute_metrics(y_true, y_pred, score)
    assert m["confusion_matrix"] == [[4, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["roc_auc"] == pytest.approx(14 / 15)
    assert cv_score(y_true, y_pred) == pytest.approx(m["balanced_accuracy"])


def test_undefined_metrics_are_none_not_invented():
    m = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert m["precision"] is None and m["f1"] == 0.0 and m["recall"] == 0.0
    assert any("Precision undefined" in n for n in m["notes"])
    one = compute_metrics([0, 0, 0], [0, 0, 1], [0.1, 0.2, 0.9])
    assert one["recall"] is None and one["balanced_accuracy"] is None and one["roc_auc"] is None
    assert roc_curve_points([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert compute_metrics([0, 1], [0, 1])["roc_auc"] is None             # no scores supplied
    assert compute_metrics([0, 1], [0, 1], [np.nan, 1.0])["roc_auc"] is None
    assert cv_score([0, 0, 0], [0, 0, 0]) == 1.0                          # falls back to accuracy


def test_input_validation():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        compute_metrics([0, 2], [0, 1])


def test_roc_points_and_wilson():
    c = roc_curve_points([0, 0, 1, 1], [.1, .4, .35, .8])
    assert c["fpr"][0] == 0 and c["tpr"][-1] == 1
    lo, hi = wilson_interval(8, 10)
    assert 0 < lo < 0.8 < hi < 1
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_timing_helpers():
    out, sec = time_call(lambda: sum(range(1000)))
    assert out == 499500 and sec >= 0
    assert measure_inference_time(lambda X: X.sum(), np.ones((10, 3)), repeats=2) >= 0
    with pytest.raises(ValueError):
        measure_inference_time(lambda X: X, np.empty((0, 3)))
