"""Базовая модель — многовыходная линейная регрессия.

Используется как baseline для проверки гипотезы исследования: насколько
нейросетевая модель снижает RMSE по сравнению с классическим линейным
подходом.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearBaseline:
    """Обёртка над sklearn.LinearRegression для многовыходной задачи.

    Многовыходность реализуется штатно: при передаче y формы (N, K)
    sklearn обучает K независимых линейных моделей.
    """

    def __init__(self) -> None:
        self.model = LinearRegression(n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
