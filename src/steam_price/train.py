from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from steam_price.config import TrainConfig, load_config
from steam_price.data import apply_scope, load_modeling_data
from steam_price.features import SteamFeatureBuilder
from steam_price.metrics import evaluate_predictions


def train(config: TrainConfig) -> dict[str, Any]:
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    full_df = load_modeling_data(config.data.path)
    scoped_df = apply_scope(full_df, config.data.selected_scope)
    trainval_df, test_df = train_test_split(
        scoped_df,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )
    relative_val_size = config.data.val_size / (1.0 - config.data.test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=relative_val_size,
        random_state=config.data.random_state,
    )

    y_train = train_df[config.data.target].to_numpy()
    y_val = val_df[config.data.target].to_numpy()
    y_test = test_df[config.data.target].to_numpy()

    feature_builder = SteamFeatureBuilder(config.features)
    X_train = feature_builder.fit_transform(train_df)
    X_val = feature_builder.transform(val_df)

    best_params = tune_xgboost(config, X_train, y_train, X_val, y_val)

    final_feature_builder = SteamFeatureBuilder(config.features)
    X_trainval = final_feature_builder.fit_transform(trainval_df)
    X_test = final_feature_builder.transform(test_df)
    y_trainval = trainval_df[config.data.target].to_numpy()

    xgb_model = XGBRegressor(**best_params)
    xgb_model.fit(X_trainval, y_trainval)
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
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_summary": final_feature_builder.feature_summary(),
        "final_model": "xgboost",
        "model_family": "XGBRegressor",
        "tuning_enabled": config.tuning.enabled,
        "tuning_trials": config.tuning.n_trials if config.tuning.enabled else 0,
        "best_params": best_params,
    }

    artifact = {
        "model": xgb_model,
        "feature_builder": final_feature_builder,
        "metadata": metadata,
        "feature_names": final_feature_builder.feature_names(),
    }

    save_outputs(config, artifact, metrics, metadata)
    log_final_run(config, metrics, metadata)

    return {
        "metrics": metrics,
        "metadata": metadata,
        "artifact_path": config.artifacts.final_model_path,
    }


def tune_xgboost(
    config: TrainConfig,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
) -> dict[str, Any]:
    base_params = dict(config.models["xgboost"])
    if not config.tuning.enabled:
        return base_params

    def objective(trial: optuna.Trial) -> float:
        params = sample_xgboost_params(config, trial)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        metrics = evaluate_predictions("xgboost", y_val, val_pred)

        with mlflow.start_run(run_name=f"trial-{trial.number:03d}"):
            mlflow.set_tag("run_type", "optuna_trial")
            mlflow.set_tag("selected_scope", config.data.selected_scope)
            mlflow.log_params(params)
            mlflow.log_metric("val_mae_dollars", metrics["mae_dollars"])
            mlflow.log_metric("val_median_ae_dollars", metrics["median_ae_dollars"])
            mlflow.log_metric("val_rmse_dollars", metrics["rmse_dollars"])
            mlflow.log_metric("val_pct_within_25pct", metrics["pct_within_25pct"])
            mlflow.log_metric("val_pct_within_50pct", metrics["pct_within_50pct"])

        return float(metrics[config.tuning.metric])

    sampler = optuna.samplers.TPESampler(seed=config.data.random_state)
    study = optuna.create_study(direction=config.tuning.direction, sampler=sampler)
    study.optimize(objective, n_trials=config.tuning.n_trials)
    return {**base_params, **study.best_params}


def sample_xgboost_params(config: TrainConfig, trial: optuna.Trial) -> dict[str, Any]:
    base_params = dict(config.models["xgboost"])
    space = config.tuning.search_space
    return {
        **base_params,
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(space["n_estimators"][0]),
            int(space["n_estimators"][1]),
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(space["learning_rate"][0]),
            float(space["learning_rate"][1]),
        ),
        "max_depth": trial.suggest_int("max_depth", int(space["max_depth"][0]), int(space["max_depth"][1])),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            int(space["min_child_weight"][0]),
            int(space["min_child_weight"][1]),
        ),
        "subsample": trial.suggest_float("subsample", float(space["subsample"][0]), float(space["subsample"][1])),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            float(space["colsample_bytree"][0]),
            float(space["colsample_bytree"][1]),
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", float(space["reg_alpha"][0]), float(space["reg_alpha"][1])),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            float(space["reg_lambda"][0]),
            float(space["reg_lambda"][1]),
        ),
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


def log_final_run(config: TrainConfig, metrics: pd.DataFrame, metadata: dict[str, Any]) -> None:
    row = metrics.iloc[0].to_dict()
    with mlflow.start_run(run_name=config.mlflow.final_run_name):
        mlflow.set_tag("run_type", "final_model")
        mlflow.set_tag("selected_scope", config.data.selected_scope)
        mlflow.log_params(metadata["best_params"])
        mlflow.log_metrics({
            "test_mae_dollars": row["mae_dollars"],
            "test_median_ae_dollars": row["median_ae_dollars"],
            "test_rmse_dollars": row["rmse_dollars"],
            "test_pct_within_25pct": row["pct_within_25pct"],
            "test_pct_within_50pct": row["pct_within_50pct"],
        })
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_artifact(str(config.artifacts.final_model_path))
        mlflow.log_artifact(str(config.artifacts.metrics_path))
        mlflow.log_artifact(str(config.artifacts.metadata_path))


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
