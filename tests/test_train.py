import json

from src import config
from src.train import (
    build_model,
    cross_validate_model,
    main,
    passes_quality_gate,
    split_features_target,
)


def test_model_meets_quality_gate(df):
    X, y = split_features_target(df)
    metrics = cross_validate_model(X, y)
    assert metrics["cv_accuracy"] >= config.MIN_ACCURACY
    assert metrics["cv_f1_macro"] >= config.MIN_F1_MACRO


def test_predictions_are_valid_labels(trained_model, splits):
    _, X_test, _, _ = splits
    preds = trained_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds) <= {0, 1, 2}


def test_training_is_deterministic(splits):
    X_train, X_test, y_train, _ = splits
    a = build_model().fit(X_train, y_train).predict(X_test)
    b = build_model().fit(X_train, y_train).predict(X_test)
    assert (a == b).all()


def test_quality_gate_rejects_bad_metrics():
    assert passes_quality_gate({"cv_accuracy": 0.5, "cv_f1_macro": 0.5}) is False


def test_main_saves_model_and_metrics(tmp_path):
    assert main(model_dir=str(tmp_path)) == 0
    metrics = json.loads((tmp_path / config.METRICS_FILENAME).read_text())
    assert metrics["passed_quality_gate"] is True
    assert (tmp_path / config.MODEL_FILENAME).exists()


def test_main_blocks_weak_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "N_ESTIMATORS", 1)
    monkeypatch.setattr(config, "MAX_DEPTH", 1)
    assert main(model_dir=str(tmp_path)) == 1
    assert not (tmp_path / config.MODEL_FILENAME).exists()