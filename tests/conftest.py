import pytest

from src.data import load_data
from src.train import build_model, split_data


@pytest.fixture(scope="session")
def df():
    return load_data()


@pytest.fixture(scope="session")
def splits(df):
    return split_data(df)


@pytest.fixture(scope="session")
def trained_model(splits):
    X_train, _, y_train, _ = splits
    return build_model().fit(X_train, y_train)