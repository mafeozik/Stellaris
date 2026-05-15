"""Модуль формирования ранжированных рекомендаций.

Реализует RecommendationGenerator из проектной части ВКР: на основе
списка отклонений метрик от пороговых значений (от RuleAnalyzer)
формирует текстовые рекомендации с пояснением причины и ранжирует их
по убыванию приоритета.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from recommender.rule_analyzer import MetricDeviation


@dataclass
class Recommendation:
    """Структура одной рекомендации."""

    title: str
    explanation: str
    target_metric: str
    priority: float
    expected_impact: str


# Шаблоны рекомендаций по метрикам
TEMPLATES: dict[tuple[str, str], dict[str, str]] = {
    ("ctr", "below"): {
        "title": "Повысить CTR через пересмотр креативов и таргетинга",
        "explanation": (
            "Прогнозное значение CTR ({value:.4f}) ниже целевого порога ({threshold:.4f}) "
            "на {pct:.1f}%. Низкий CTR указывает на слабую релевантность объявления "
            "целевой аудитории."
        ),
        "expected_impact": (
            "Перепроверка заголовков, обновление креативов и сужение таргетинга "
            "по поведенческим признакам способны поднять CTR на 15–40%."
        ),
    },
    ("cr", "below"): {
        "title": "Оптимизировать посадочную страницу и предложение",
        "explanation": (
            "Прогнозное значение CR ({value:.4f}) ниже целевого порога ({threshold:.4f}) "
            "на {pct:.1f}%. Низкая конверсия означает разрыв между обещанием в "
            "объявлении и опытом на посадочной странице."
        ),
        "expected_impact": (
            "A/B-тестирование форм, упрощение пути конверсии и уточнение УТП "
            "обычно повышают CR на 10–30%."
        ),
    },
    ("cpc", "above"): {
        "title": "Снизить CPC: корректировка ставок и качества",
        "explanation": (
            "Прогнозное значение CPC ({value:.2f} руб.) выше нормы ({threshold:.2f} руб.) "
            "на {pct:.1f}%. Высокая стоимость клика указывает на низкий Quality "
            "Score или агрессивные ставки в высококонкурентной нише."
        ),
        "expected_impact": (
            "Повышение Quality Score (через релевантность ключей и креатива), "
            "снижение ставок в часы низкой эффективности и уход от перегретых "
            "плейсментов снижают CPC на 10–25%."
        ),
    },
}


class RecommendationGenerator:
    """Формирует упорядоченный по приоритету список рекомендаций."""

    def generate(self, deviations: Sequence[MetricDeviation]) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        for d in deviations:
            tpl = TEMPLATES.get((d.metric, d.direction))
            if tpl is None:
                continue
            pct = d.relative_deviation * 100
            recommendations.append(Recommendation(
                title=tpl["title"],
                explanation=tpl["explanation"].format(
                    value=d.value, threshold=d.threshold, pct=pct,
                ),
                target_metric=d.metric,
                priority=d.priority,
                expected_impact=tpl["expected_impact"],
            ))
        return recommendations

    def format_for_display(self, recs: Sequence[Recommendation]) -> str:
        if not recs:
            return "Все метрики в пределах нормы. Рекомендаций нет."
        lines: list[str] = []
        for i, r in enumerate(recs, start=1):
            lines.append(f"#{i} [приоритет {r.priority:.2f}] {r.title}")
            lines.append(f"   Причина: {r.explanation}")
            lines.append(f"   Ожидаемый эффект: {r.expected_impact}")
            lines.append("")
        return "\n".join(lines)
