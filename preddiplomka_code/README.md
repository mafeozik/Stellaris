# Прототип рекомендательной системы управления рекламными кампаниями

Преддипломная практика, ИКМО-06-24, Курицын Д.С.

## Назначение

Экспериментальная проверка гипотезы: нейросетевая модель (MLP) снижает RMSE
прогнозирования метрик рекламных кампаний (CTR, CR, CPC) не менее чем на 15%
по сравнению с базовой моделью линейной регрессии.

## Структура

```
preddiplomka/
├── config.py                       # Конфигурация (seed, гиперпараметры, пути)
├── data_generator.py               # Модуль генерации синтетических данных
├── preprocessor.py                 # Модуль предобработки (DataPreprocessor + FeatureBuilder)
├── models/
│   ├── linear_baseline.py          # Базовая модель: линейная регрессия
│   └── mlp.py                      # Нейросетевая модель: MLP 256-128-64
├── recommender/
│   ├── rule_analyzer.py            # Анализатор отклонений метрик
│   └── recommendation_engine.py    # Формирование ранжированных рекомендаций
├── train_and_evaluate.py           # Главный скрипт: обучение + оценка
├── requirements.txt
├── data/                           # Генерируемые датасеты
└── results/                        # Метрики + графики
```

## Запуск

```bash
# 1. Окружение
python -m venv .venv
source .venv/bin/activate  # на Linux/macOS
pip install -r requirements.txt

# 2. Полный прогон (данные -> обучение -> оценка)
python train_and_evaluate.py
```

Время выполнения: 1–3 минуты на CPU, 30–60 секунд на GPU.

## Результаты

После прогона в `results/` появятся:
- `metrics.json` — все цифры (RMSE/MAE по каждой целевой, итог по гипотезе)
- `loss_curves.png` — кривые обучения MLP
- `rmse_comparison.png` — столбчатая диаграмма RMSE: LR vs MLP
- `predictions_scatter.png` — диаграмма рассеяния «предсказание vs факт»
- `feature_correlations.png` — корреляции признаков с целевыми

Также в `data/synthetic_campaigns.csv` сохраняется сгенерированный датасет.
