from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from steam_price.metrics import to_price


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_log_price(artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    feature_builder = artifact["feature_builder"]
    model = artifact["model"]
    matrix = feature_builder.transform(frame)
    return model.predict(matrix)


def predict_price(artifact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    return to_price(predict_log_price(artifact, frame))
