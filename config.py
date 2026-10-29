"""Central configuration for NeuralNetClassifier.

Edit the two dataset paths below if your files live somewhere else.
Everything tunable (seed, split, hyper-parameter grids, CV, output folders)
is defined here so that no settings are scattered through the code base.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 1. DATASET PATHS  (change these if your files move)
#    Raw strings (r"...") keep Windows back-slashes, spaces and '+' intact.
#    Optional override without editing this file: set the environment
#    variables NNC_HEART_PATH / NNC_STUDENT_PATH.
# --------------------------------------------------------------------------- #
# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Your local Windows paths
local_heart = r"D:\Last year 2026-2027\1st term\Neural network\NN Project\heart+disease\processed.cleveland.data"
local_student = r"D:\Last year 2026-2027\1st term\Neural network\NN Project\student+performance\student\student-por.csv"

# Repository fallback paths (pointing to your project's 'data' folder)
repo_heart = BASE_DIR / "data" / "processed.cleveland.data"
repo_student = BASE_DIR / "data" / "student-por.csv"

# Dynamic selection: use local D:\ path if it exists, otherwise use the repo's data folder
default_heart = local_heart if Path(local_heart).exists() else repo_heart
default_student = local_student if Path(local_student).exists() else repo_student

HEART_DATA_PATH = Path(os.environ.get("NNC_HEART_PATH", default_heart))
STUDENT_DATA_PATH = Path(os.environ.get("NNC_STUDENT_PATH", default_student))

DATASET_PATHS: Dict[str, Path] = {"heart": HEART_DATA_PATH, "student": STUDENT_DATA_PATH}
CONFIG_PATH_NAMES = {"heart": "HEART_DATA_PATH", "student": "STUDENT_DATA_PATH"}
ENV_VARS = {"heart": "NNC_HEART_PATH", "student": "NNC_STUDENT_PATH"}
DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "heart": {
        "label": "Cleveland Heart Disease",
        "task": "Predict presence of heart disease (original target 1-4) vs. absence (0).",
        "class_names": {0: "No heart disease", 1: "Heart disease"},
    },
    "student": {
        "label": "Student Performance (Portuguese)",
        "task": "Predict final-grade outcome: fail (G3 < 10) vs. pass (G3 >= 10).",
        "class_names": {0: "Fail (G3 < 10)", 1: "Pass (G3 >= 10)"},
    },
}
DATASET_KEYS = tuple(DATASET_INFO)


def get_dataset_path(key: str) -> Path:
    """Return the configured path for a dataset key ('heart' or 'student')."""
    if key not in DATASET_PATHS:
        raise KeyError(f"Unknown dataset '{key}'. Valid keys: {list(DATASET_PATHS)}")
    return Path(DATASET_PATHS[key])


# --------------------------------------------------------------------------- #
# 2. DATASET SPECIFICATIONS
# --------------------------------------------------------------------------- #
HEART_COLUMNS: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
HEART_NUMERIC_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HEART_CATEGORICAL_FEATURES: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
HEART_EXPECTED_ROWS = 303
HEART_EXPECTED_MISSING = {"ca": 4, "thal": 2}

STUDENT_COLUMNS: List[str] = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2", "G3",
]
STUDENT_TARGET_SOURCE = "G3"
STUDENT_PASS_THRESHOLD = 10
STUDENT_LEAKAGE_COLUMNS: List[str] = ["G1", "G2", "G3"]  # never used as inputs
STUDENT_EXPECTED_ROWS = 649
# Text-valued (nominal / binary) columns -> imputed + one-hot encoded.
STUDENT_CATEGORICAL_FEATURES: List[str] = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic",
]
# Integer-coded scales where the ORDER is meaningful (1-5 ratings, counts, age...)
# -> kept numeric, imputed (median) and standardised.
STUDENT_NUMERIC_FEATURES: List[str] = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
    "goout", "Dalc", "Walc", "health", "absences",
]
STUDENT_FEATURE_COLUMNS: List[str] = [c for c in STUDENT_COLUMNS if c not in STUDENT_LEAKAGE_COLUMNS]

# --------------------------------------------------------------------------- #
# 3. EXPERIMENT SETTINGS
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.20                      # 80 % train / 20 % held-out test
CV_FOLDS = 5                          # stratified CV on the TRAINING split only
CV_SCORING = "balanced_accuracy"      # falls back to accuracy if undefined

RBF_N_CENTERS_CANDIDATES: List[int] = [5, 10, 20, 40, 80]
RBF_WIDTH_METHODS: List[str] = ["global", "local"]
RBF_WIDTH_SCALE_CANDIDATES: List[float] = [0.5, 1.0, 2.0]
RBF_RIDGE_CANDIDATES: List[float] = [1e-3, 1e-1, 1.0]
RBF_N_NEIGHBORS = 2                   # used by the "local" width rule

PNN_BANDWIDTH_CANDIDATES: List[float] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
PNN_PRIORS = "empirical"              # "empirical" (class frequencies) or "uniform"

TIMING_REPEATS = 5                    # repeats for inference-time measurement

RESULTS_DIR = PROJECT_ROOT / "results"
SAVE_RESULTS = True


def default_settings() -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of all experiment settings."""
    return {
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
        "rbf_n_centers": list(RBF_N_CENTERS_CANDIDATES),
        "rbf_width_methods": list(RBF_WIDTH_METHODS),
        "rbf_width_scales": list(RBF_WIDTH_SCALE_CANDIDATES),
        "rbf_ridges": list(RBF_RIDGE_CANDIDATES),
        "rbf_n_neighbors": RBF_N_NEIGHBORS,
        "pnn_bandwidths": list(PNN_BANDWIDTH_CANDIDATES),
        "pnn_priors": PNN_PRIORS,
        "timing_repeats": TIMING_REPEATS,
        "save_results": SAVE_RESULTS,
        "results_dir": str(RESULTS_DIR),
    }
