"""Loading, validation and target construction for the two datasets.

Nothing here modifies the original files: they are only read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import config

PathLike = Union[str, Path]


class DatasetError(Exception):
    """Base class for dataset problems."""


class DatasetNotFoundError(DatasetError):
    """The configured dataset file does not exist."""


class DatasetFormatError(DatasetError):
    """The file exists but does not have the expected structure."""


@dataclass
class LoadedDataset:
    """A validated dataset ready for the experiment pipeline."""

    key: str
    label: str
    X: pd.DataFrame                      # model input features only
    y: pd.Series                         # binary target (0/1)
    numeric_features: List[str]
    categorical_features: List[str]
    raw: pd.DataFrame                    # all columns (for the explorer), incl. original target/grades
    class_names: Dict[int, str]
    report: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    source_path: Optional[Path] = None


# ------------------------------------------------------------------ helpers
def _check_file(key: str, path: Path) -> None:
    if not path.is_file():
        info = config.DATASET_INFO[key]
        raise DatasetNotFoundError(
            f"{info['label']} data file not found at:\n    {path}\n"
            f"Fix: open config.py and update {config.CONFIG_PATH_NAMES[key]} so it points to your "
            f"local copy (or set the environment variable {config.ENV_VARS[key]}). "
            f"config.py is located in: {config.PROJECT_ROOT}"
        )


def binarize_heart_target(original: pd.Series) -> pd.Series:
    """0 -> 0 (no disease); 1, 2, 3, 4 -> 1 (disease)."""
    if original.isna().any():
        raise DatasetFormatError("The Cleveland target column contains missing/invalid values.")
    return (original > 0).astype(int).rename("target_binary")


def binarize_student_target(g3: pd.Series) -> pd.Series:
    """G3 < 10 -> 0 (fail); G3 >= 10 -> 1 (pass)."""
    if g3.isna().any():
        raise DatasetFormatError("The G3 column contains missing/invalid values.")
    return (g3 >= config.STUDENT_PASS_THRESHOLD).astype(int).rename("pass_binary")


def _build_report(X: pd.DataFrame, y: pd.Series, raw: pd.DataFrame, n_num: int, n_cat: int) -> Dict[str, Any]:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    missing = X.isna().sum()
    return {
        "n_records": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric_features": n_num,
        "n_categorical_features": n_cat,
        "n_raw_columns": int(raw.shape[1]),
        "target_counts": {int(k): int(v) for k, v in counts.items()},
        "missing_per_feature": {c: int(v) for c, v in missing.items() if v > 0},
        "total_missing_values": int(missing.sum()),
    }


# -------------------------------------------------------------- Cleveland
def load_heart(path: Optional[PathLike] = None) -> LoadedDataset:
    """Load the Cleveland Heart Disease file (headerless CSV, '?' = missing)."""
    path = Path(path) if path is not None else config.get_dataset_path("heart")
    _check_file("heart", path)
    try:
        text = pd.read_csv(path, header=None, dtype=str, skipinitialspace=True, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetFormatError(f"Could not parse {path} as a headerless CSV: {exc}") from exc
    if text.shape[1] != len(config.HEART_COLUMNS):
        raise DatasetFormatError(
            f"Expected {len(config.HEART_COLUMNS)} columns in the Cleveland file, found {text.shape[1]}. "
            "Is this the 'processed.cleveland.data' file?"
        )
    text.columns = list(config.HEART_COLUMNS)
    text = text.apply(lambda s: s.str.strip())

    warns: List[str] = []
    text = text.replace({"?": np.nan, "": np.nan})   # BEFORE numeric conversion
    numeric = text.apply(pd.to_numeric, errors="coerce")
    extra_nan = int(numeric.isna().sum().sum() - text.isna().sum().sum())
    if extra_nan > 0:
        warns.append(f"{extra_nan} non-numeric entries (other than '?') were converted to missing values.")

    original = numeric["target"]
    binary = binarize_heart_target(original)
    unexpected = sorted(set(original.unique()) - {0, 1, 2, 3, 4})
    if unexpected:
        warns.append(f"Unexpected original target values found: {unexpected} (expected 0-4).")

    feature_cols = [c for c in config.HEART_COLUMNS if c != "target"]
    X = numeric[feature_cols].copy()
    raw = numeric.copy()
    raw["target_binary"] = binary

    if len(X) != config.HEART_EXPECTED_ROWS:
        warns.append(f"Expected {config.HEART_EXPECTED_ROWS} records, found {len(X)}.")
    for col, expected in config.HEART_EXPECTED_MISSING.items():
        found = int(X[col].isna().sum())
        if found != expected:
            warns.append(f"Expected {expected} missing values in '{col}', found {found}.")

    num, cat = list(config.HEART_NUMERIC_FEATURES), list(config.HEART_CATEGORICAL_FEATURES)
    return LoadedDataset(
        key="heart", label=config.DATASET_INFO["heart"]["label"], X=X, y=binary,
        numeric_features=num, categorical_features=cat, raw=raw,
        class_names=dict(config.DATASET_INFO["heart"]["class_names"]),
        report=_build_report(X, binary, raw, len(num), len(cat)), warnings=warns, source_path=path,
    )


# ---------------------------------------------------------------- Student
def load_student(path: Optional[PathLike] = None) -> LoadedDataset:
    """Load student-por.csv (semicolon separated, with header). G1/G2/G3 are never inputs."""
    path = Path(path) if path is not None else config.get_dataset_path("student")
    _check_file("student", path)
    try:
        df = pd.read_csv(path, sep=";")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetFormatError(f"Could not parse {path} as a ';'-separated CSV: {exc}") from exc
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    if df.shape[1] == 1:
        raise DatasetFormatError("Only one column was found - the file does not look ';'-separated.")
    missing_cols = [c for c in config.STUDENT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetFormatError(f"Missing expected columns in the student file: {missing_cols}")

    warns: List[str] = []
    extra = [c for c in df.columns if c not in config.STUDENT_COLUMNS]
    if extra:
        warns.append(f"Ignoring unexpected extra columns: {extra}")
    df = df[config.STUDENT_COLUMNS].copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].isna(), df[c].astype(str).str.strip())
    for c in config.STUDENT_NUMERIC_FEATURES + config.STUDENT_LEAKAGE_COLUMNS:
        before = int(df[c].isna().sum())
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if int(df[c].isna().sum()) > before:
            warns.append(f"Non-numeric values in '{c}' were converted to missing values.")

    binary = binarize_student_target(df[config.STUDENT_TARGET_SOURCE])
    X = df[config.STUDENT_FEATURE_COLUMNS].copy()
    leaked = [c for c in config.STUDENT_LEAKAGE_COLUMNS if c in X.columns]
    if leaked:  # defensive: should be impossible
        raise DatasetFormatError(f"Leakage columns present in the feature matrix: {leaked}")
    raw = df.copy()
    raw["pass_binary"] = binary

    if len(X) != config.STUDENT_EXPECTED_ROWS:
        warns.append(f"Expected {config.STUDENT_EXPECTED_ROWS} records, found {len(X)}.")
    num, cat = list(config.STUDENT_NUMERIC_FEATURES), list(config.STUDENT_CATEGORICAL_FEATURES)
    return LoadedDataset(
        key="student", label=config.DATASET_INFO["student"]["label"], X=X, y=binary,
        numeric_features=num, categorical_features=cat, raw=raw,
        class_names=dict(config.DATASET_INFO["student"]["class_names"]),
        report=_build_report(X, binary, raw, len(num), len(cat)), warnings=warns, source_path=path,
    )


def load_dataset(key: str, path: Optional[PathLike] = None) -> LoadedDataset:
    """Load a dataset by key ('heart' or 'student')."""
    if key == "heart":
        return load_heart(path)
    if key == "student":
        return load_student(path)
    raise KeyError(f"Unknown dataset '{key}'. Valid keys: {list(config.DATASET_KEYS)}")
