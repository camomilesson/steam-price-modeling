from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class DataConfig:
    path: Path
    target: str
    selected_scope: str
    val_size: float
    test_size: float
    random_state: int


@dataclass(frozen=True)
class FeatureConfig:
    tag_min_count: int
    category_min_count: int
    numeric: list[str]
    boolean: list[str]
    multilabel: dict[str, str]


@dataclass(frozen=True)
class ArtifactConfig:
    model_dir: Path
    final_model_path: Path
    metrics_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool
    n_trials: int
    direction: str
    metric: str
    search_space: dict[str, list[float | int]]


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    final_run_name: str


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    features: FeatureConfig
    models: dict[str, dict[str, Any]]
    tuning: TuningConfig
    mlflow: MLflowConfig
    artifacts: ArtifactConfig


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text())

    return TrainConfig(
        data=DataConfig(
            path=Path(raw["data"]["path"]),
            target=raw["data"]["target"],
            selected_scope=raw["data"]["selected_scope"],
            val_size=float(raw["data"]["val_size"]),
            test_size=float(raw["data"]["test_size"]),
            random_state=int(raw["data"]["random_state"]),
        ),
        features=FeatureConfig(
            tag_min_count=int(raw["features"]["tag_min_count"]),
            category_min_count=int(raw["features"]["category_min_count"]),
            numeric=list(raw["features"]["numeric"]),
            boolean=list(raw["features"]["boolean"]),
            multilabel=dict(raw["features"]["multilabel"]),
        ),
        models=dict(raw["models"]),
        tuning=TuningConfig(
            enabled=bool(raw["tuning"]["enabled"]),
            n_trials=int(raw["tuning"]["n_trials"]),
            direction=raw["tuning"]["direction"],
            metric=raw["tuning"]["metric"],
            search_space=dict(raw["tuning"]["search_space"]),
        ),
        mlflow=MLflowConfig(
            tracking_uri=raw["mlflow"]["tracking_uri"],
            experiment_name=raw["mlflow"]["experiment_name"],
            final_run_name=raw["mlflow"]["final_run_name"],
        ),
        artifacts=ArtifactConfig(
            model_dir=Path(raw["artifacts"]["model_dir"]),
            final_model_path=Path(raw["artifacts"]["final_model_path"]),
            metrics_path=Path(raw["artifacts"]["metrics_path"]),
            metadata_path=Path(raw["artifacts"]["metadata_path"]),
        ),
    )
