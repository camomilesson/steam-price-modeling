from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from steam_price.config import FeatureConfig


def label_counts(series: pd.Series) -> pd.Series:
    return series.explode().loc[lambda s: s.notna() & s.ne("")].value_counts()


def keep_vocab(items: list[str], vocab: list[str]) -> list[str]:
    vocab_set = set(vocab)
    return [item for item in items if item in vocab_set]


@dataclass
class SteamFeatureBuilder:
    config: FeatureConfig
    genre_vocab: list[str] | None = None
    tag_vocab: list[str] | None = None
    category_vocab: list[str] | None = None

    def __post_init__(self) -> None:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        self.numeric_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.config.numeric),
                ("boolean", SimpleImputer(strategy="most_frequent"), self.config.boolean),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        self.genre_binarizer: MultiLabelBinarizer | None = None
        self.tag_binarizer: MultiLabelBinarizer | None = None
        self.category_binarizer: MultiLabelBinarizer | None = None

    def fit(self, frame: pd.DataFrame) -> SteamFeatureBuilder:
        self.genre_vocab = sorted(label_counts(frame[self.config.multilabel["genre"]]).index.tolist())
        self.tag_vocab = sorted(
            label_counts(frame[self.config.multilabel["tag"]])
            .loc[lambda s: s >= self.config.tag_min_count]
            .index.tolist()
        )
        self.category_vocab = sorted(
            label_counts(frame[self.config.multilabel["category"]])
            .loc[lambda s: s >= self.config.category_min_count]
            .index.tolist()
        )
        self.genre_binarizer = MultiLabelBinarizer(classes=self.genre_vocab, sparse_output=True)
        self.tag_binarizer = MultiLabelBinarizer(classes=self.tag_vocab, sparse_output=True)
        self.category_binarizer = MultiLabelBinarizer(classes=self.category_vocab, sparse_output=True)
        self._transform(frame, fit=True)
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return self._transform(frame, fit=False)

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        self.fit(frame)
        return self.transform(frame)

    def feature_names(self) -> list[str]:
        self._require_fitted()
        return (
            self.config.numeric
            + self.config.boolean
            + [f"genre::{name}" for name in self.genre_vocab or []]
            + [f"tag::{name}" for name in self.tag_vocab or []]
            + [f"category::{name}" for name in self.category_vocab or []]
        )

    def feature_summary(self) -> dict[str, int]:
        self._require_fitted()
        return {
            "numeric_features": len(self.config.numeric),
            "boolean_features": len(self.config.boolean),
            "genre_features": len(self.genre_vocab or []),
            "tag_features": len(self.tag_vocab or []),
            "category_features": len(self.category_vocab or []),
            "total_features": len(self.feature_names()),
        }

    def _transform(self, frame: pd.DataFrame, fit: bool) -> sparse.csr_matrix:
        self._require_vocab()
        frame_for_transform = frame.copy()
        frame_for_transform[self.config.boolean] = frame_for_transform[self.config.boolean].astype(float)
        frame_for_transform[self.config.multilabel["genre"]] = frame_for_transform[
            self.config.multilabel["genre"]
        ].apply(lambda items: keep_vocab(items, self.genre_vocab or []))
        frame_for_transform[self.config.multilabel["tag"]] = frame_for_transform[
            self.config.multilabel["tag"]
        ].apply(lambda items: keep_vocab(items, self.tag_vocab or []))
        frame_for_transform[self.config.multilabel["category"]] = frame_for_transform[
            self.config.multilabel["category"]
        ].apply(lambda items: keep_vocab(items, self.category_vocab or []))

        if fit:
            numeric_matrix = self.numeric_transformer.fit_transform(frame_for_transform)
            genre_matrix = self.genre_binarizer.fit_transform(frame_for_transform[self.config.multilabel["genre"]])
            tag_matrix = self.tag_binarizer.fit_transform(frame_for_transform[self.config.multilabel["tag"]])
            category_matrix = self.category_binarizer.fit_transform(
                frame_for_transform[self.config.multilabel["category"]]
            )
        else:
            self._require_fitted()
            numeric_matrix = self.numeric_transformer.transform(frame_for_transform)
            genre_matrix = self.genre_binarizer.transform(frame_for_transform[self.config.multilabel["genre"]])
            tag_matrix = self.tag_binarizer.transform(frame_for_transform[self.config.multilabel["tag"]])
            category_matrix = self.category_binarizer.transform(frame_for_transform[self.config.multilabel["category"]])

        return sparse.hstack(
            [
                sparse.csr_matrix(numeric_matrix),
                genre_matrix,
                tag_matrix,
                category_matrix,
            ],
            format="csr",
        )

    def _require_vocab(self) -> None:
        if self.genre_vocab is None or self.tag_vocab is None or self.category_vocab is None:
            raise RuntimeError("Feature vocabularies are not fitted")

    def _require_fitted(self) -> None:
        self._require_vocab()
        if self.genre_binarizer is None or self.tag_binarizer is None or self.category_binarizer is None:
            raise RuntimeError("Feature builder is not fitted")
