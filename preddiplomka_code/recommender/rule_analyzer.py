"""Модуль анализа отклонений метрик от пороговых значений.

Реализует RuleAnalyzer из проектной части ВКР: для каждой целевой
метрики (CTR, CR, CPC) проверяет, отклоняется ли её прогнозное значение
от заданного порога или среднеотраслевого уровня, и какова величина
отклонения. Эта величина используется для вычисления приоритета
рекомендации по формуле:

    priority = weight_metric * |relative_deviation|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Веса метрик отражают относительную бизнес-важность
METRIC_WEIGHTS: dict[str, float] = {
    "ctr": 1.0,
    "cr": 1.5,
    "cpc": 1.2,
}

# Пороговые «нормальные» значения метрик (типичные ориентиры для
# рекламных кампаний; легко заменяются на бенчмарк по нише)
METRIC_THRESHOLDS: dict[str, dict[str, float]] = {
    "ctr": {"low": 0.010, "high": 0.080},
    "cr":  {"low": 0.015, "high": 0.150},
    "cpc": {"low": 10.0,  "high": 300.0},
}


@dataclass
class MetricDeviation:
    metric: str
    value: float
    threshold: float
    direction: Literal["below", "above"]
    relative_deviation: float
    priority: float


class RuleAnalyzer:
    """Сравнивает прогнозные значения метрик с пороговыми и формирует
    список отклонений с рассчитанным приоритетом.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.weights = weights or METRIC_WEIGHTS
        self.thresholds = thresholds or METRIC_THRESHOLDS

    def analyze(self, predicted: dict[str, float]) -> list[MetricDeviation]:
        deviations: list[MetricDeviation] = []

        for metric, value in predicted.items():
            if metric not in self.thresholds:
                continue
            bounds = self.thresholds[metric]
            weight = self.weights.get(metric, 1.0)

            if metric == "cpc":
                # Для CPC «плохо» — когда выше нормы
                if value > bounds["high"]:
                    rel_dev = (value - bounds["high"]) / bounds["high"]
                    deviations.append(MetricDeviation(
                        metric=metric, value=value, threshold=bounds["high"],
                        direction="above", relative_deviation=rel_dev,
                        priority=weight * abs(rel_dev),
                    ))
            else:
                # Для CTR и CR «плохо» — когда ниже нормы
                if value < bounds["low"]:
                    rel_dev = (bounds["low"] - value) / bounds["low"]
                    deviations.append(MetricDeviation(
                        metric=metric, value=value, threshold=bounds["low"],
                        direction="below", relative_deviation=rel_dev,
                        priority=weight * abs(rel_dev),
                    ))

        # Сортировка по убыванию приоритета
        deviations.sort(key=lambda d: d.priority, reverse=True)
        return deviations
