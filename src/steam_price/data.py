from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


LIST_COLUMNS = ["genre_list", "tag_list", "category_list"]


def parse_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    return ast.literal_eval(str(value))


def load_modeling_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in LIST_COLUMNS:
        frame[column] = frame[column].apply(parse_list)
    return frame


def apply_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "full_valid":
        return frame.copy()
    if scope == "review_count >= 10":
        return frame.loc[frame["review_count"] >= 10].copy()
    if scope == "review_count >= 50":
        return frame.loc[frame["review_count"] >= 50].copy()
    if scope == "recommendations > 0":
        return frame.loc[frame["log1p_recommendations"] > 0].copy()
    raise ValueError(f"Unsupported scope: {scope}")
