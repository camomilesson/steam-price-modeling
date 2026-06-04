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
class TrainConfig:
    data: DataConfig
    features: FeatureConfig
    models: dict[str, dict[str, Any]]
    artifacts: ArtifactConfig


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text())

    return TrainConfig(
        data=DataConfig(
            path=Path(raw["data"]["path"]),
            target=raw["data"]["target"],
            selected_scope=raw["data"]["selected_scope"],
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
        artifacts=ArtifactConfig(
            model_dir=Path(raw["artifacts"]["model_dir"]),
            final_model_path=Path(raw["artifacts"]["final_model_path"]),
            metrics_path=Path(raw["artifacts"]["metrics_path"]),
            metadata_path=Path(raw["artifacts"]["metadata_path"]),
        ),
    )
