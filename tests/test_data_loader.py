import numpy as np
import pandas as pd
import pytest

import config
from src.data_loader import (DatasetFormatError, DatasetNotFoundError, binarize_heart_target,
                             binarize_student_target, load_dataset, load_heart, load_student)


def test_heart_loads_with_exact_columns_and_missing_markers(heart_csv):
    ds = load_heart(heart_csv)
    assert list(ds.raw.columns[:14]) == config.HEART_COLUMNS
    assert ds.X.shape == (160, 13)
    assert ds.X["ca"].isna().sum() == 4          # '?' became NaN
    assert ds.X["thal"].isna().sum() == 2
    assert all(pd.api.types.is_numeric_dtype(ds.X[c]) for c in ds.X.columns)
    assert ds.report["total_missing_values"] == 6
    assert len(ds.X) == len(ds.y)                # no records deleted


def test_heart_target_conversion_and_no_target_in_features(heart_csv):
    ds = load_heart(heart_csv)
    original = ds.raw["target"]
    assert set(ds.y.unique()) <= {0, 1}
    assert ((original == 0) == (ds.y == 0)).all()
    assert "target" not in ds.X.columns and "target_binary" not in ds.X.columns


def test_binarize_functions():
    assert binarize_heart_target(pd.Series([0, 1, 2, 3, 4])).tolist() == [0, 1, 1, 1, 1]
    assert binarize_student_target(pd.Series([0, 9, 10, 11, 20])).tolist() == [0, 0, 1, 1, 1]
    with pytest.raises(DatasetFormatError):
        binarize_heart_target(pd.Series([0, np.nan]))


def test_student_loader_prevents_grade_leakage(student_csv):
    ds = load_student(student_csv)
    assert ds.X.shape[1] == 30
    for col in ("G1", "G2", "G3"):
        assert col not in ds.X.columns
    assert set(ds.numeric_features) | set(ds.categorical_features) == set(ds.X.columns)
    assert ((ds.raw["G3"] >= 10).astype(int) == ds.y).all()
    assert set(ds.y.unique()) == {0, 1}


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_heart(tmp_path / "nope.data")
    assert "config.py" in str(exc.value) and "HEART_DATA_PATH" in str(exc.value)
    with pytest.raises(DatasetNotFoundError) as exc2:
        load_student(tmp_path / "nope.csv")
    assert "STUDENT_DATA_PATH" in str(exc2.value)


def test_wrong_format_detected(tmp_path):
    bad = tmp_path / "bad.data"
    bad.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(DatasetFormatError):
        load_heart(bad)
    bad2 = tmp_path / "bad.csv"
    bad2.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(DatasetFormatError):
        load_student(bad2)


def test_student_missing_column_detected(student_csv, tmp_path):
    df = pd.read_csv(student_csv, sep=";").drop(columns=["G3"])
    p = tmp_path / "short.csv"
    df.to_csv(p, sep=";", index=False)
    with pytest.raises(DatasetFormatError) as exc:
        load_student(p)
    assert "G3" in str(exc.value)


def test_load_dataset_dispatch_and_unknown_key(heart_csv):
    assert load_dataset("heart", heart_csv).key == "heart"
    with pytest.raises(KeyError):
        load_dataset("nope")


def test_unexpected_record_count_warns(heart_csv):
    assert any("Expected 303 records" in w for w in load_heart(heart_csv).warnings)
