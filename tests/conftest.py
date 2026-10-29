"""Synthetic fixtures used ONLY for software tests. They are not real data and are never shown as results."""
import csv

import numpy as np
import pandas as pd
import pytest

import config


@pytest.fixture
def blobs():
    """Two well separated Gaussian blobs (train, test)."""
    rng = np.random.default_rng(0)
    def make(n):
        X0 = rng.normal(-2.0, 0.7, size=(n, 3)); X1 = rng.normal(2.0, 0.7, size=(n, 3))
        return np.vstack([X0, X1]), np.array([0] * n + [1] * n)
    return make(60), make(25)


@pytest.fixture
def heart_csv(tmp_path):
    """Headerless Cleveland-style CSV with '?' markers (4 in 'ca', 2 in 'thal')."""
    rng = np.random.default_rng(1)
    n = 160
    z = rng.normal(size=n)
    rows = []
    for i in range(n):
        target = 0 if z[i] + rng.normal(0, 0.4) < 0 else int(rng.integers(1, 5))
        ca = "?" if i in (3, 17, 40, 77) else f"{float(np.clip(round(1 + z[i] + rng.normal(0, .5)), 0, 3)):.1f}"
        thal = "?" if i in (5, 60) else f"{float(rng.choice([3, 6, 7], p=[.5, .1, .4]) if z[i] > 0 else 3):.1f}"
        rows.append([int(55 + 8 * z[i]), int(rng.integers(0, 2)), int(rng.integers(1, 5)), int(130 + 10 * z[i]),
                     int(240 + 20 * rng.normal()), int(rng.integers(0, 2)), int(rng.integers(0, 3)), int(150 - 12 * z[i]),
                     int(z[i] > 0.3), round(float(max(0, z[i] + rng.normal(0, .5))), 1), int(rng.integers(1, 4)), ca, thal, target])
    path = tmp_path / "processed.cleveland.data"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


@pytest.fixture
def student_csv(tmp_path):
    """Semicolon-separated, header, quoted strings; G3 < 10 for roughly half the rows."""
    rng = np.random.default_rng(2)
    n = 150
    z = rng.normal(size=n)
    data = {
        "school": rng.choice(["GP", "MS"], n), "sex": rng.choice(["F", "M"], n), "age": rng.integers(15, 20, n),
        "address": rng.choice(["U", "R"], n), "famsize": rng.choice(["GT3", "LE3"], n), "Pstatus": rng.choice(["T", "A"], n),
        "Medu": rng.integers(0, 5, n), "Fedu": rng.integers(0, 5, n),
        "Mjob": rng.choice(["at_home", "teacher", "health", "services", "other"], n),
        "Fjob": rng.choice(["at_home", "teacher", "health", "services", "other"], n),
        "reason": rng.choice(["course", "home", "reputation", "other"], n), "guardian": rng.choice(["mother", "father", "other"], n),
        "traveltime": rng.integers(1, 5, n), "studytime": np.clip(np.round(2 + z), 1, 4).astype(int),
        "failures": np.clip(np.round(-z), 0, 3).astype(int), "schoolsup": rng.choice(["yes", "no"], n),
        "famsup": rng.choice(["yes", "no"], n), "paid": rng.choice(["yes", "no"], n), "activities": rng.choice(["yes", "no"], n),
        "nursery": rng.choice(["yes", "no"], n), "higher": np.where(z > -1, "yes", "no"), "internet": rng.choice(["yes", "no"], n),
        "romantic": rng.choice(["yes", "no"], n), "famrel": rng.integers(1, 6, n), "freetime": rng.integers(1, 6, n),
        "goout": rng.integers(1, 6, n), "Dalc": rng.integers(1, 6, n), "Walc": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n), "absences": rng.integers(0, 20, n),
    }
    g2 = np.clip(np.round(10 + 3 * z + rng.normal(0, 1, n)), 0, 20).astype(int)
    data["G1"] = np.clip(g2 + rng.integers(-1, 2, n), 0, 20)
    data["G2"] = g2
    data["G3"] = np.clip(g2 + rng.integers(-1, 2, n), 0, 20)
    df = pd.DataFrame(data)[config.STUDENT_COLUMNS]
    path = tmp_path / "student-por.csv"
    df.to_csv(path, sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)
    return path


SMALL_SETTINGS = {"cv_folds": 3, "rbf_n_centers": [3, 6], "rbf_width_methods": ["global", "local"], "rbf_width_scales": [1.0],
                  "rbf_ridges": [0.1], "pnn_bandwidths": [0.5, 1.0], "timing_repeats": 1, "save_results": False}


@pytest.fixture
def small_settings():
    return dict(SMALL_SETTINGS)
