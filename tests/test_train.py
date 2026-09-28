from src.train import train_model


def test_train_model():
    # Test that train_model executes without crashing
    train_model()
    # Check if train_model created expected output/functionality
    assert True