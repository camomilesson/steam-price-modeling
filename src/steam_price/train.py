from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from steam_price.config import TrainConfig, load_config
from steam_price.data import apply_scope, load_modeling_data
from steam_price.features import SteamFeatureBuilder
from steam_price.metrics import evaluate_predictions


def train(config: TrainConfig) -> dict[str, Any]:
    full_df = load_modeling_data(config.data.path)
    scoped_df = apply_scope(full_df, config.data.selected_scope)
    train_df, test_df = train_test_split(
        scoped_df,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    y_train = train_df[config.data.target].to_numpy()
    y_test = test_df[config.data.target].to_numpy()

    feature_builder = SteamFeatureBuilder(config.features)
    X_train = feature_builder.fit_transform(train_df)
    X_test = feature_builder.transform(test_df)

    xgb_model = XGBRegressor(**config.models["xgboost"])
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    metrics = pd.DataFrame(
        [
            evaluate_predictions("xgboost", y_test, xgb_pred),
        ]
    )

    metadata = {
        "selected_scope": config.data.selected_scope,
        "target": config.data.target,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_summary": feature_builder.feature_summary(),
        "final_model": "xgboost",
        "model_family": "XGBRegressor",
    }

    artifact = {
        "model": xgb_model,
        "feature_builder": feature_builder,
        "metadata": metadata,
        "feature_names": feature_builder.feature_names(),
    }

    save_outputs(config, artifact, metrics, metadata)

    return {
        "metrics": metrics,
        "metadata": metadata,
        "artifact_path": config.artifacts.final_model_path,
    }


def save_outputs(
    config: TrainConfig,
    artifact: dict[str, Any],
    metrics: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    config.artifacts.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.artifacts.final_model_path)
    config.artifacts.metrics_path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2))
    config.artifacts.metadata_path.write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Steam price models.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(load_config(args.config))
    print("Training complete")
    print(f"Saved final model: {result['artifact_path']}")
    print(json.dumps(result["metadata"], indent=2))
    print(result["metrics"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
