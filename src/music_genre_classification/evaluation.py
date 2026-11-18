"""Consistent classification metrics and confusion-matrix data."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)


def evaluate_classifier(model, X_test, y_test, *, labels=None) -> dict:
    """Evaluate a fitted estimator on held-out recordings."""
    predictions = model.predict(X_test)
    class_labels = labels if labels is not None else sorted(set(y_test))
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(
            y_test, predictions, labels=class_labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=class_labels).tolist(),
        "labels": list(class_labels),
    }


def feature_matrix(features: pd.DataFrame):
    """Separate metadata from numeric predictors without leaking track IDs or labels."""
    required = {"track_id", "genre"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features is missing required columns: {sorted(missing)}")
    return features.drop(columns=["track_id", "genre"]), features["genre"].copy()
