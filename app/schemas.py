from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GameFeatures(BaseModel):
    """Friendly input schema for a single Steam game price prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Example Quest",
                "proposed_price": 19.99,
                "release_year": 2024,
                "required_age": 0,
                "dlc_count": 1,
                "achievements": 24,
                "windows": True,
                "mac": False,
                "linux": False,
                "genres": ["Indie", "Adventure"],
                "tags": ["Story Rich", "Singleplayer", "Exploration"],
                "categories": ["Single-player", "Family Sharing", "Steam Cloud"],
            }
        }
    )

    name: str = Field(..., min_length=1, examples=["Example Quest"])
    proposed_price: float = Field(..., ge=0, examples=[19.99])
    release_year: int = Field(..., ge=1970, le=2100, examples=[2024])
    required_age: int = Field(0, ge=0, le=21, examples=[0])
    dlc_count: int = Field(0, ge=0, examples=[1])
    achievements: int = Field(0, ge=0, examples=[24])
    windows: bool = Field(True, examples=[True])
    mac: bool = Field(False, examples=[False])
    linux: bool = Field(False, examples=[False])
    genres: list[str] = Field(default_factory=list, examples=[["Indie", "Adventure"]])
    tags: list[str] = Field(default_factory=list, examples=[["Story Rich", "Singleplayer"]])
    categories: list[str] = Field(default_factory=list, examples=[["Single-player", "Family Sharing"]])


class PricePrediction(BaseModel):
    name: str
    proposed_price: float
    predicted_market_price: float
    price_ratio: float
    price_alignment: str
    selected_scope: str
    model_family: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    final_model: str
    model_family: str
    selected_scope: str
    target: str
    train_rows: int
    val_rows: int | None = None
    test_rows: int
    feature_summary: dict[str, int]
    tuning_enabled: bool | None = None
    tuning_trials: int | None = None
    best_params: dict[str, float | int | str] | None = None
