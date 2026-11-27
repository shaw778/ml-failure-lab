"""Refitting a scaler on test data destroys the learned feature scale and origin."""

import math

import numpy as np
from sklearn.preprocessing import StandardScaler


def training_fixture() -> tuple[np.ndarray, StandardScaler]:
    # Single feature with values [0.0, 2.0]
    X_train = np.array([[0.0], [2.0]], dtype=float)
    scaler = StandardScaler().fit(X_train)
    return X_train, scaler


def report(approach: str, test_data: np.ndarray | list[float] | None = None) -> dict:
    if approach not in {"bad", "fixed"}:
        raise ValueError("Choose 'bad' or 'fixed'.")

    _, training_scaler = training_fixture()
    if test_data is None:
        X_test = np.array([[10.0], [12.0]], dtype=float)
    else:
        raw = np.asarray(test_data, dtype=float)
        X_test = raw.reshape(-1, 1) if raw.ndim == 1 else raw

    if X_test.ndim != 2 or X_test.shape[1] != 1 or len(X_test) == 0:
        raise ValueError("Test data must be a non-empty two-dimensional array with one column.")

    train_mean = float(training_scaler.mean_[0])
    train_scale = float(training_scaler.scale_[0])
    expected_transformed = (X_test - train_mean) / train_scale

    if approach == "bad":
        # Misleading operation: refit a new scaler independently on test records
        test_scaler = StandardScaler().fit(X_test)
        transformed = test_scaler.transform(X_test)
        applied_mean = float(test_scaler.mean_[0])
        applied_scale = float(test_scaler.scale_[0])
    else:
        # Correction: transform test records using parameters learned on training data
        transformed = training_scaler.transform(X_test)
        applied_mean = train_mean
        applied_scale = train_scale

    predictions = (transformed[:, 0] > 5.0).astype(int).tolist()
    expected_predictions = (expected_transformed[:, 0] > 5.0).astype(int).tolist()

    reused = math.isclose(applied_mean, train_mean, rel_tol=1e-5, abs_tol=1e-5) and math.isclose(
        applied_scale, train_scale, rel_tol=1e-5, abs_tol=1e-5
    )

    return {
        "approach": approach,
        "raw_test_values": X_test.flatten().tolist(),
        "transformed_values": transformed.flatten().tolist(),
        "expected_transformed_values": expected_transformed.flatten().tolist(),
        "training_mean": train_mean,
        "training_scale": train_scale,
        "applied_mean": applied_mean,
        "applied_scale": applied_scale,
        "predictions": predictions,
        "expected_predictions": expected_predictions,
        "reused_training_parameters": reused,
    }


def require_training_scaler_parameters(result: dict) -> None:
    if not (
        math.isclose(result["applied_mean"], result["training_mean"], rel_tol=1e-5, abs_tol=1e-5)
        and math.isclose(
            result["applied_scale"], result["training_scale"], rel_tol=1e-5, abs_tol=1e-5
        )
    ):
        raise ValueError(
            f"Test data was scaled using mean={result['applied_mean']} and "
            f"scale={result['applied_scale']}, but training parameters were "
            f"mean={result['training_mean']} and scale={result['training_scale']}."
        )
    for actual, expected in zip(
        result["transformed_values"],
        result["expected_transformed_values"],
        strict=True,
    ):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Transformed test values {result['transformed_values']} do not match "
                f"expected values {result['expected_transformed_values']} under training scale."
            )


def run() -> dict:
    return {
        "case": "test-scaler-refit",
        "question": "Did test preprocessing preserve the training scale and origin?",
        "measurements": [report("bad"), report("fixed")],
        "requirement": "Test inputs must be transformed using parameters fitted on training data.",
        "interpretation": (
            "Refitting a StandardScaler on test inputs maps [10, 12] to [-1, 1], "
            "shifting the coordinate frame so large values appear near zero. Reusing the "
            "training scaler (mean=1, scale=1) yields [9, 11], preserving the feature's meaning."
        ),
        "limitation": (
            "Reusing the training scaler guarantees consistent coordinates, but it does "
            "not remove underlying distribution shift or guarantee model generalization."
        ),
    }


def verify(approach: str) -> None:
    result = report(approach)
    require_training_scaler_parameters(result)
