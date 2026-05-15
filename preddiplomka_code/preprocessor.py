"""Модуль предобработки данных.

Реализует pipeline DataPreprocessor + FeatureBuilder из проектной части ВКР:
- one-hot кодирование категориальных признаков (drop_first для устранения
  мультиколлинеарности у линейной модели);
- стандартизация числовых признаков;
- лог-преобразование тяжелохвостых целевых переменных (CTR, CR, CPC);
- стандартизация целевых для устойчивого обучения многовыходной MLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_COLS, TARGET_COLS


@dataclass
class Preprocessor:
    """Обёртка для X-преобразования и Y-преобразования.

    Атрибуты заполняются после вызова ``fit``.
    """

    column_transformer: ColumnTransformer | None = None
    target_scaler: StandardScaler | None = None
    feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Preprocessor":
        numerical_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"),
                    CATEGORICAL_COLS,
                ),
            ],
            remainder="drop",
        )
        self.column_transformer.fit(X)

        # Имена признаков после преобразования
        self.feature_names_ = (
            numerical_cols
            + list(
                self.column_transformer.named_transformers_["cat"].get_feature_names_out(
                    CATEGORICAL_COLS
                )
            )
        )

        # Целевые: log1p + standardize
        y_log = np.log1p(y)
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(y_log)

        return self

    def transform_x(self, X: pd.DataFrame) -> np.ndarray:
        assert self.column_transformer is not None, "Call fit first"
        return self.column_transformer.transform(X)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        assert self.target_scaler is not None, "Call fit first"
        return self.target_scaler.transform(np.log1p(y))

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Обратное преобразование: scaled -> log -> original."""
        assert self.target_scaler is not None, "Call fit first"
        y_log = self.target_scaler.inverse_transform(y_scaled)
        # Защита: log1p ожидает >= -1; после inverse_transform значения близки к log-шкале
        y_orig = np.expm1(y_log)
        # Гарантируем неотрицательность (целевые > 0 по природе)
        return np.clip(y_orig, a_min=0.0, a_max=None)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_) if self.feature_names_ else 0
