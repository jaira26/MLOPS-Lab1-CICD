import numpy as np
import pytest

from src.data import TARGET_COL, validate_data


def test_real_data_passes_validation(df):
    assert validate_data(df) is True


def test_expected_shape(df):
    assert df.shape == (150, 5)


def test_rejects_nulls(df):
    broken = df.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="null"):
        validate_data(broken)


def test_rejects_missing_target(df):
    with pytest.raises(ValueError, match="missing target"):
        validate_data(df.drop(columns=[TARGET_COL]))


def test_rejects_unknown_label(df):
    broken = df.copy()
    broken.loc[0, TARGET_COL] = 7
    with pytest.raises(ValueError, match="unexpected target labels"):
        validate_data(broken)


def test_rejects_negative_measurement(df):
    broken = df.copy()
    broken.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        validate_data(broken)


@pytest.mark.parametrize("n_drop", [1, 2])
def test_rejects_wrong_feature_count(df, n_drop):
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    with pytest.raises(ValueError, match="features"):
        validate_data(df.drop(columns=feature_cols[:n_drop]))