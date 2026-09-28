import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src import config
from src.data import TARGET_COL, load_data, validate_data


def split_features_target(df):
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def split_data(df):
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
    )


def cross_validate_model(X, y) -> dict:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(build_model(), X, y, cv=cv, scoring=["accuracy", "f1_macro"])
    return {
        "cv_accuracy": round(float(scores["test_accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "cv_f1_macro": round(float(scores["test_f1_macro"].mean()), 4),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "holdout_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "holdout_f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "n_test_samples": len(y_test),
    }


def passes_quality_gate(metrics: dict) -> bool:
    return (
        metrics["cv_accuracy"] >= config.MIN_ACCURACY
        and metrics["cv_f1_macro"] >= config.MIN_F1_MACRO
    )


def main(model_dir: str | None = None) -> int:
    df = load_data()
    validate_data(df)

    X, y = split_features_target(df)
    metrics = cross_validate_model(X, y)

    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model().fit(X_train, y_train)
    metrics.update(evaluate(model, X_test, y_test))

    metrics["min_accuracy"] = config.MIN_ACCURACY
    metrics["min_f1_macro"] = config.MIN_F1_MACRO
    metrics["passed_quality_gate"] = passes_quality_gate(metrics)

    out_dir = Path(model_dir or config.MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / config.METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if not metrics["passed_quality_gate"]:
        print("Quality gate FAILED: model is below the required thresholds. Model not saved.")
        return 1

    joblib.dump(model, out_dir / config.MODEL_FILENAME)
    print(f"Quality gate passed. Model saved to {out_dir / config.MODEL_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())