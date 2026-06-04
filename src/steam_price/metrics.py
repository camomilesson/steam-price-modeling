from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def to_price(log_values: np.ndarray) -> np.ndarray:
    return np.expm1(log_values).clip(min=0.49)


def evaluate_predictions(model_name: str, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float | str]:
    actual = to_price(y_true_log)
    predicted = to_price(y_pred_log)
    ratio = np.maximum(actual, predicted) / np.minimum(actual, predicted)

    return {
        "model": model_name,
        "mae_dollars": float(mean_absolute_error(actual, predicted)),
        "median_ae_dollars": float(median_absolute_error(actual, predicted)),
        "rmse_dollars": float(mean_squared_error(actual, predicted) ** 0.5),
        "mean_actual_price": float(actual.mean()),
        "mean_predicted_price": float(predicted.mean()),
        "pct_within_25pct": float((ratio <= 1.25).mean() * 100),
        "pct_within_50pct": float((ratio <= 1.50).mean() * 100),
    }
