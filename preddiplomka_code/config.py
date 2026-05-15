"""Конфигурация проекта.

Все гиперпараметры, пути и константы — здесь, чтобы при необходимости менять
их в одном месте и иметь возможность ссылаться на конкретные значения в отчёте.
"""

from pathlib import Path

# Воспроизводимость
SEED = 42

# Пути
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Генерация данных
N_SAMPLES = 80_000

# Разбиение
TEST_RATIO = 0.20
VAL_RATIO_OF_TRAIN = 0.10  # 10% от train (т.е. 8% от total)

# Целевые переменные и категориальные признаки
TARGET_COLS = ["ctr", "cr", "cpc"]
CATEGORICAL_COLS = [
    "platform",
    "ad_type",
    "device",
    "region",
    "audience_segment",
]

# MLP-архитектура (соответствует проектной части ВКР)
MLP_HIDDEN = (256, 128, 64)
MLP_DROPOUT = 0.2

# Обучение MLP
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_BATCH_SIZE = 256
MLP_EPOCHS = 120
MLP_PATIENCE = 15  # early stopping

# Порог гипотезы
HYPOTHESIS_RMSE_REDUCTION_PCT = 15.0
