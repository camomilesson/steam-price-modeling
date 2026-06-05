from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import GameFeatures, PricePrediction
from steam_price.metrics import to_price


class ModelLoader:
    def __init__(self) -> None:
        self.artifact: dict[str, Any] | None = None
        self.model_info: dict[str, Any] = {}

    def load(self) -> None:
        models_dir = Path(os.getenv("MODELS_DIR", "models"))
        model_path = models_dir / "final_model.joblib"

        print(f"[model] Loading model from {model_path}")
        self.artifact = joblib.load(model_path)
        self.model_info = dict(self.artifact.get("metadata", {}))
        print(f"[model] Loaded {self.model_info.get('model_family', 'model')}")

    def predict(self, features: GameFeatures) -> PricePrediction:
        if self.artifact is None:
            raise RuntimeError("Model is not loaded")

        frame = self._features_to_dataframe(features)
        matrix = self.artifact["feature_builder"].transform(frame)
        predicted_log = self.artifact["model"].predict(matrix)
        predicted_price = float(to_price(predicted_log)[0])
        proposed_price = float(features.proposed_price)
        price_ratio = proposed_price / predicted_price if predicted_price > 0 else float("inf")

        return PricePrediction(
            name=features.name,
            proposed_price=round(proposed_price, 2),
            predicted_market_price=round(predicted_price, 2),
            price_ratio=round(price_ratio, 3),
            price_alignment=self._alignment_label(price_ratio),
            selected_scope=self.model_info.get("selected_scope", ""),
            model_family=self.model_info.get("model_family", ""),
        )

    def _features_to_dataframe(self, features: GameFeatures) -> pd.DataFrame:
        platform_count = int(features.windows) + int(features.mac) + int(features.linux)
        release_age_years = max(date.today().year - features.release_year, 0)
        dlc_count_clipped = min(features.dlc_count, 50)
        achievements_clipped = min(features.achievements, 500)

        row = {
            "Name": features.name,
            "current_price": features.proposed_price,
            "required_age_clipped": min(features.required_age, 18),
            "platform_count": platform_count,
            "Release year": features.release_year,
            "release_age_years": release_age_years,
            "dlc_count_clipped": dlc_count_clipped,
            "achievements_clipped": achievements_clipped,
            "log1p_dlc_count": float(np.log1p(features.dlc_count)),
            "log1p_achievements": float(np.log1p(features.achievements)),
            "genre_count": len(features.genres),
            "tag_count": len(features.tags),
            "category_count": len(features.categories),
            "Windows": features.windows,
            "Mac": features.mac,
            "Linux": features.linux,
            "genre_list": features.genres,
            "tag_list": features.tags,
            "category_list": features.categories,
        }
        return pd.DataFrame([row])

    @staticmethod
    def _alignment_label(price_ratio: float) -> str:
        if price_ratio < 0.6:
            return "below-market"
        if price_ratio <= 1.5:
            return "market-aligned"
        return "above-market"

    @property
    def is_loaded(self) -> bool:
        return self.artifact is not None
