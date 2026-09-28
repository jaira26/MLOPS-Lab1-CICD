import pandas as pd
from sklearn.datasets import load_iris

TARGET_COL = "target"
EXPECTED_N_FEATURES = 4
EXPECTED_CLASSES = {0, 1, 2}


def load_data() -> pd.DataFrame:
    """Load the Iris dataset as one DataFrame: 4 measurement columns plus 'target'."""
    return load_iris(as_frame=True).frame


def validate_data(df: pd.DataFrame) -> bool:
    """Raise ValueError listing every problem if the data breaks the expected schema."""
    errors = []

    if df.empty:
        errors.append("dataframe is empty")
    if TARGET_COL not in df.columns:
        errors.append(f"missing target column '{TARGET_COL}'")
    else:
        unexpected = set(df[TARGET_COL].unique()) - EXPECTED_CLASSES
        if unexpected:
            errors.append(f"unexpected target labels: {sorted(unexpected)}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    if len(feature_cols) != EXPECTED_N_FEATURES:
        errors.append(f"expected {EXPECTED_N_FEATURES} features, got {len(feature_cols)}")

    if df.isnull().values.any():
        errors.append("dataframe contains null values")
    elif feature_cols and (df[feature_cols] <= 0).values.any():
        errors.append("flower measurements must be positive")

    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))
    return True