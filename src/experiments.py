"""End-to-end experiment pipeline: load -> split -> preprocess -> tune (CV on train) -> fit -> evaluate once.

Both models share the SAME train/test indices and the SAME fitted preprocessing pipeline.
During cross-validation the preprocessing is re-fitted inside every fold, so validation
folds never influence imputers, scalers or encoders.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import config
from src.data_loader import LoadedDataset, PathLike, load_dataset
from src.evaluation import compute_metrics, cv_score, measure_inference_time, roc_curve_points, time_call
from src.pnn import PNN
from src.preprocessing import build_preprocessor, get_feature_names, stratified_split_indices, transform_features
from src.rbf import RBFNetwork, select_centers

MODEL_NAMES = ("RBF", "PNN")


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    cv_mean: float
    cv_std: float
    table: List[Dict[str, Any]]          # best configurations first (top 15)


@dataclass
class ModelResult:
    name: str
    best_params: Dict[str, Any]
    cv_mean: float
    cv_std: float
    cv_table: List[Dict[str, Any]]
    train_time_s: float
    inference_ms_per_sample: float
    metrics: Dict[str, Any]
    roc: Optional[Dict[str, List[float]]]
    y_pred: np.ndarray
    y_score: np.ndarray


@dataclass
class ExperimentResult:
    dataset: LoadedDataset
    settings: Dict[str, Any]
    train_idx: np.ndarray
    test_idx: np.ndarray
    stratified_split: bool
    stratified_cv: bool
    preprocessor: Any
    feature_names: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    models: Dict[str, Any]
    results: Dict[str, ModelResult]
    warnings: List[str] = field(default_factory=list)
    results_path: Optional[Path] = None

    def summary_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary (no model objects)."""
        rep = self.dataset.report
        return {
            "dataset": self.dataset.key, "label": self.dataset.label,
            "n_records": rep["n_records"], "n_features_raw": rep["n_features"],
            "n_features_encoded": len(self.feature_names),
            "n_train": int(len(self.train_idx)), "n_test": int(len(self.test_idx)),
            "stratified_split": self.stratified_split, "stratified_cv": self.stratified_cv,
            "target_counts": rep["target_counts"], "settings": self.settings,
            "models": {
                n: {"best_params": r.best_params, "cv_mean": r.cv_mean, "cv_std": r.cv_std,
                    "train_time_s": r.train_time_s, "inference_ms_per_sample": r.inference_ms_per_sample,
                    "metrics": r.metrics}
                for n, r in self.results.items()
            },
            "warnings": self.warnings,
        }


# ------------------------------------------------------------------ CV tools
def make_cv_splits(y: Sequence[int], n_splits: int, seed: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], bool]:
    """Stratified K-fold splits when each class has enough samples, else plain K-fold (flag False)."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    min_count = int(np.unique(y, return_counts=True)[1].min())
    k = min(n_splits, min_count)
    if k >= 2:
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y)), True
    if len(y) < 2:
        raise ValueError("Not enough training samples for cross-validation.")
    return list(KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed).split(idx)), False


def prepare_fold_data(X_train: pd.DataFrame, y_train: np.ndarray, splits, preprocessor_factory):
    """Fit a fresh preprocessor on each fold's training part only; transform both parts."""
    folds = []
    for tr, va in splits:
        pre = preprocessor_factory()
        Xtr = transform_features(pre.fit(X_train.iloc[tr]), X_train.iloc[tr])
        Xva = transform_features(pre, X_train.iloc[va])
        folds.append((Xtr, y_train[tr], Xva, y_train[va]))
    return folds


def _summarise(scores: Dict[tuple, List[float]], names: Sequence[str], fixed: Dict[str, Any]) -> TuningResult:
    rows = []
    for combo, vals in scores.items():
        row = dict(zip(names, combo))
        row.update(fixed)
        row["cv_mean"], row["cv_std"] = float(np.mean(vals)), float(np.std(vals))
        rows.append(row)
    rows.sort(key=lambda r: -r["cv_mean"])      # stable: earlier (simpler) grid entries win ties
    best = {k: v for k, v in rows[0].items() if k not in ("cv_mean", "cv_std")}
    return TuningResult(best, rows[0]["cv_mean"], rows[0]["cv_std"], rows[:15])


def tune_rbf(folds, settings: Dict[str, Any], seed: int) -> TuningResult:
    """Grid search for the RBF network using cross-validation on training folds only."""
    scores: Dict[tuple, List[float]] = {}
    for Xtr, ytr, Xva, yva in folds:
        for m in settings["rbf_n_centers"]:
            centers = select_centers(Xtr, m, seed)          # K-Means once per (fold, m)
            for method in settings["rbf_width_methods"]:
                for scale in settings["rbf_width_scales"]:
                    for ridge in settings["rbf_ridges"]:
                        model = RBFNetwork(n_centers=m, width_method=method, width_scale=scale, ridge=ridge,
                                           n_neighbors=settings["rbf_n_neighbors"], random_state=seed)
                        model.fit(Xtr, ytr, centers=centers)
                        scores.setdefault((m, method, scale, ridge), []).append(cv_score(yva, model.predict(Xva)))
    return _summarise(scores, ["n_centers", "width_method", "width_scale", "ridge"],
                      {"n_neighbors": settings["rbf_n_neighbors"]})


def tune_pnn(folds, settings: Dict[str, Any]) -> TuningResult:
    """Search the PNN bandwidth using cross-validation on training folds only."""
    scores: Dict[tuple, List[float]] = {}
    for Xtr, ytr, Xva, yva in folds:
        for h in settings["pnn_bandwidths"]:
            model = PNN(bandwidth=h, priors=settings["pnn_priors"]).fit(Xtr, ytr)
            scores.setdefault((h,), []).append(cv_score(yva, model.predict(Xva)))
    return _summarise(scores, ["bandwidth"], {"priors": settings["pnn_priors"]})


# ------------------------------------------------------------------- helpers
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Cannot serialise {type(o)}")


def _validate_settings(s: Dict[str, Any]) -> None:
    for k in ("rbf_n_centers", "rbf_width_methods", "rbf_width_scales", "rbf_ridges", "pnn_bandwidths"):
        if not s.get(k):
            raise ValueError(f"Setting '{k}' must be a non-empty list.")
    if not 0 < s["test_size"] < 1:
        raise ValueError("test_size must be between 0 and 1.")


def run_experiment(dataset_key: str, path: Optional[PathLike] = None,
                   settings: Optional[Dict[str, Any]] = None) -> ExperimentResult:
    """Run the complete experiment for one dataset and return a structured result."""
    s = {**config.default_settings(), **(settings or {})}
    _validate_settings(s)
    seed = int(s["seed"])
    warns: List[str] = []

    ds = load_dataset(dataset_key, path)                       # 1-2. load + binary target
    warns.extend(ds.warnings)
    y_all = ds.y.to_numpy()
    train_idx, test_idx, strat = stratified_split_indices(y_all, s["test_size"], seed)   # 3. split
    if not strat:
        warns.append("Stratified split was impossible; a plain random split was used.")
    X_train, X_test = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    if len(np.unique(y_test)) < 2:
        warns.append("The test split contains only one class: some metrics are undefined.")

    def factory():
        return build_preprocessor(ds.numeric_features, ds.categorical_features)

    splits, strat_cv = make_cv_splits(y_train, int(s["cv_folds"]), seed)                 # 6. tune (train only)
    if not strat_cv:
        warns.append("Stratified cross-validation was impossible; plain K-fold was used.")
    folds = prepare_fold_data(X_train, y_train, splits, factory)
    tuned = {"RBF": tune_rbf(folds, s, seed), "PNN": tune_pnn(folds, s)}

    pre = factory().fit(X_train)                                                         # 4-5. fit on train only
    Xtr_t, Xte_t = transform_features(pre, X_train), transform_features(pre, X_test)
    feature_names = get_feature_names(pre)

    models: Dict[str, Any] = {}
    results: Dict[str, ModelResult] = {}
    for name in MODEL_NAMES:
        params = dict(tuned[name].best_params)
        if name == "RBF":
            model = RBFNetwork(random_state=seed, **params)
        else:
            model = PNN(**params)
        _, fit_s = time_call(lambda: model.fit(Xtr_t, y_train))                          # 7. final fit
        y_pred = model.predict(Xte_t)                                                    # 8. evaluate ONCE
        y_score = model.positive_class_score(Xte_t)
        metrics = compute_metrics(y_test, y_pred, y_score)
        infer_ms = measure_inference_time(model.predict, Xte_t, int(s["timing_repeats"]))   # 9. timing
        models[name] = model
        results[name] = ModelResult(
            name=name, best_params=params, cv_mean=tuned[name].cv_mean, cv_std=tuned[name].cv_std,
            cv_table=tuned[name].table, train_time_s=float(fit_s), inference_ms_per_sample=infer_ms,
            metrics=metrics, roc=roc_curve_points(y_test, y_score), y_pred=np.asarray(y_pred), y_score=np.asarray(y_score),
        )

    result = ExperimentResult(
        dataset=ds, settings=s, train_idx=train_idx, test_idx=test_idx, stratified_split=strat,
        stratified_cv=strat_cv, preprocessor=pre, feature_names=feature_names, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, models=models, results=results, warnings=warns,
    )
    if s.get("save_results"):                                                            # 12. save
        out_dir = Path(s["results_dir"])
        if not out_dir.is_absolute():
            out_dir = config.PROJECT_ROOT / out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{dataset_key}_results.json"
            target.write_text(json.dumps(result.summary_dict(), indent=2, default=_json_default), encoding="utf-8")
            result.results_path = target
        except OSError as exc:
            result.warnings.append(f"Could not save results to {out_dir}: {exc}")
    return result


def predict_samples(result: ExperimentResult, X_raw: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Predict RAW (un-preprocessed) samples with the models trained by the pipeline."""
    Xt = transform_features(result.preprocessor, X_raw[result.dataset.X.columns])
    out = {}
    for name, model in result.models.items():
        out[name] = {"pred": model.predict(Xt), "scores": model.class_scores(Xt), "proba": model.predict_proba(Xt)}
    return out


def best_model(result: ExperimentResult) -> Tuple[Optional[str], str]:
    """Best observed model on the test set: balanced accuracy, then F1, then accuracy.
    Returns (name or None for a tie, short explanation)."""
    def key(r: ModelResult):
        m = r.metrics
        return tuple(-1.0 if m[k] is None else round(m[k], 12) for k in ("balanced_accuracy", "f1", "accuracy"))
    a, b = (result.results[n] for n in MODEL_NAMES)
    if key(a) == key(b):
        return None, "RBF and PNN tied on balanced accuracy, F1 and accuracy."
    winner = a if key(a) > key(b) else b
    return winner.name, f"{winner.name} had the higher balanced accuracy / F1 / accuracy (checked in that order)."
