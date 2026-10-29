"""Metrics and timing helpers.

Binary metrics treat label ``labels[1]`` (default 1) as the POSITIVE class.
Undefined metrics (e.g. precision with no positive predictions) are returned as ``None``
together with an explanatory note; no value is invented.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _safe_div(num: float, den: float) -> Optional[float]:
    return float(num / den) if den > 0 else None


def compute_metrics(y_true, y_pred, y_score=None, labels: Sequence[int] = (0, 1)) -> Dict[str, Any]:
    """Compute binary classification metrics from real predictions.

    Returns a dict with accuracy, precision, recall, f1, balanced_accuracy, roc_auc,
    confusion_matrix (rows = true [neg, pos], cols = predicted [neg, pos]) and notes.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.ndim != 1 or yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must be 1-D arrays of equal length.")
    if len(yt) == 0:
        raise ValueError("Cannot compute metrics on an empty set.")
    neg, pos = labels
    allowed = set(labels)
    if not set(np.unique(yt)) <= allowed or not set(np.unique(yp)) <= allowed:
        raise ValueError(f"Labels must be within {sorted(allowed)}.")
    notes: List[str] = []
    cm = np.array([[np.sum((yt == t) & (yp == p)) for p in (neg, pos)] for t in (neg, pos)], dtype=int)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    accuracy = (tp + tn) / len(yt)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    specificity = _safe_div(tn, tn + fp)
    if precision is None:
        notes.append("Precision undefined: the model made no positive predictions.")
    if recall is None:
        notes.append("Recall undefined: no positive samples in this set.")
    if f1 is None:
        notes.append("F1 undefined: no positive samples and no positive predictions.")
    if recall is not None and specificity is not None:
        balanced = (recall + specificity) / 2.0
    else:
        balanced = None
        notes.append("Balanced accuracy undefined: only one true class present.")

    auc = None
    if y_score is None:
        notes.append("ROC-AUC unavailable: no class scores supplied.")
    else:
        ys = np.asarray(y_score, dtype=float)
        if ys.shape != yt.shape:
            raise ValueError("y_score must have the same length as y_true.")
        if not np.all(np.isfinite(ys)):
            notes.append("ROC-AUC unavailable: scores contain NaN/inf.")
        elif len(np.unique(yt)) < 2:
            notes.append("ROC-AUC undefined: only one true class present.")
        else:
            auc = float(roc_auc_score((yt == pos).astype(int), ys))
    return {
        "n_samples": int(len(yt)), "accuracy": float(accuracy), "precision": precision, "recall": recall,
        "f1": f1, "balanced_accuracy": balanced, "roc_auc": auc,
        "confusion_matrix": cm.tolist(), "notes": notes,
    }


def cv_score(y_true, y_pred) -> float:
    """Score used for model selection: balanced accuracy (accuracy if that is undefined)."""
    m = compute_metrics(y_true, y_pred)
    return float(m["balanced_accuracy"] if m["balanced_accuracy"] is not None else m["accuracy"])


def roc_curve_points(y_true, y_score, pos_label: int = 1) -> Optional[Dict[str, List[float]]]:
    """FPR/TPR arrays for plotting, or None if the ROC curve is undefined."""
    yt = (np.asarray(y_true) == pos_label).astype(int)
    ys = np.asarray(y_score, dtype=float)
    if len(np.unique(yt)) < 2 or not np.all(np.isfinite(ys)):
        return None
    fpr, tpr, _ = roc_curve(yt, ys)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95 % interval for a proportion (e.g. test accuracy)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def time_call(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Run fn(), returning (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def measure_inference_time(predict_fn: Callable[[np.ndarray], Any], X: np.ndarray, repeats: int = 5) -> float:
    """Median wall-clock inference time per sample, in MILLISECONDS."""
    if len(X) == 0:
        raise ValueError("X must contain at least one sample.")
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) / len(X) * 1000.0)
