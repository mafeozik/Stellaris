"""Главный скрипт прогона: данные → обучение → оценка → артефакты.

Полный цикл воспроизведения экспериментов для проверки гипотезы:
1. Генерация синтетического датасета.
2. Препроцессинг (стандартизация, one-hot, лог-преобразование целевых).
3. Разбиение train/val/test (72/8/20).
4. Обучение базовой модели (линейная регрессия).
5. Обучение нейросетевой модели (MLP).
6. Оценка обеих моделей на тестовой выборке.
7. Сохранение метрик и графиков.
8. Демонстрация работы модуля рекомендаций на одном примере.

Запуск:
    python train_and_evaluate.py
"""

from __future__ import annotations

import json
import random

import matplotlib
matplotlib.use("Agg")  # без интерактивного бэкенда (для серверной среды)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import torch

from config import (
    DATA_DIR,
    HYPOTHESIS_RMSE_REDUCTION_PCT,
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN,
    MLP_LR,
    MLP_WEIGHT_DECAY,
    N_SAMPLES,
    RESULTS_DIR,
    SEED,
    TARGET_COLS,
    TEST_RATIO,
    VAL_RATIO_OF_TRAIN,
)
from data_generator import generate_synthetic_data
from models.linear_baseline import LinearBaseline
from models.mlp import get_device, predict, train_mlp
from preprocessor import Preprocessor
from recommender.recommendation_engine import RecommendationGenerator
from recommender.rule_analyzer import RuleAnalyzer


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def evaluate_per_target(
    y_true: np.ndarray,
    y_pred_lr: np.ndarray,
    y_pred_mlp: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Считает RMSE/MAE для каждой целевой переменной, на оригинальной и
    лог-шкале, плюс относительное улучшение MLP над LR.
    """
    results: dict[str, dict[str, float]] = {}
    for i, name in enumerate(TARGET_COLS):
        rmse_lr = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred_lr[:, i])))
        rmse_mlp = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred_mlp[:, i])))
        mae_lr = float(mean_absolute_error(y_true[:, i], y_pred_lr[:, i]))
        mae_mlp = float(mean_absolute_error(y_true[:, i], y_pred_mlp[:, i]))

        # Лог-шкала: устойчивее для тяжелохвостых метрик (особенно CPC)
        eps = 1e-9
        y_true_log = np.log1p(np.clip(y_true[:, i], 0, None))
        y_pred_lr_log = np.log1p(np.clip(y_pred_lr[:, i], 0, None))
        y_pred_mlp_log = np.log1p(np.clip(y_pred_mlp[:, i], 0, None))
        rmse_lr_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_lr_log)))
        rmse_mlp_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_mlp_log)))

        reduction_pct = (rmse_lr - rmse_mlp) / (rmse_lr + eps) * 100.0
        reduction_pct_log = (rmse_lr_log - rmse_mlp_log) / (rmse_lr_log + eps) * 100.0

        results[name] = {
            "rmse_lr": rmse_lr,
            "rmse_mlp": rmse_mlp,
            "mae_lr": mae_lr,
            "mae_mlp": mae_mlp,
            "rmse_reduction_pct": reduction_pct,
            "rmse_lr_logscale": rmse_lr_log,
            "rmse_mlp_logscale": rmse_mlp_log,
            "rmse_reduction_pct_logscale": reduction_pct_log,
        }
    return results


def plot_loss_curves(history) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.train_losses, label="Train", linewidth=2)
    ax.plot(history.val_losses, label="Validation", linewidth=2)
    ax.axvline(history.best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best epoch ({history.best_epoch})")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE loss (scaled targets)")
    ax.set_title("Кривые обучения MLP")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=130)
    plt.close()


def plot_rmse_comparison(per_target: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    targets = list(per_target.keys())
    rmse_lr = [per_target[t]["rmse_lr_logscale"] for t in targets]
    rmse_mlp = [per_target[t]["rmse_mlp_logscale"] for t in targets]
    x = np.arange(len(targets))
    w = 0.36
    bars1 = ax.bar(x - w / 2, rmse_lr, w, label="Линейная регрессия", color="#E07A5F")
    bars2 = ax.bar(x + w / 2, rmse_mlp, w, label="MLP", color="#3D5A80")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets])
    ax.set_ylabel("RMSE (log-шкала)")
    ax.set_title("Сравнение RMSE: линейная регрессия vs MLP")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Подписи столбцов и % улучшения
    for i, (l, m) in enumerate(zip(rmse_lr, rmse_mlp)):
        ax.text(i - w / 2, l, f"{l:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        reduction = (l - m) / l * 100
        ax.text(i, max(l, m) * 1.10, f"−{reduction:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="green" if reduction >= 15 else "red")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=130)
    plt.close()


def plot_predictions_scatter(
    y_true: np.ndarray, y_lr: np.ndarray, y_mlp: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, name in enumerate(TARGET_COLS):
        ax = axes[i]
        # сэмплируем для читаемости
        n_show = min(3000, len(y_true))
        idx = np.random.default_rng(SEED).choice(len(y_true), n_show, replace=False)
        ax.scatter(y_true[idx, i], y_lr[idx, i], alpha=0.25, s=8,
                   label="LR", color="#E07A5F")
        ax.scatter(y_true[idx, i], y_mlp[idx, i], alpha=0.25, s=8,
                   label="MLP", color="#3D5A80")
        lo, hi = float(np.min(y_true[:, i])), float(np.max(y_true[:, i]))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5)
        if name == "cpc":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"Фактическое {name.upper()}")
        ax.set_ylabel(f"Прогноз {name.upper()}")
        ax.set_title(f"{name.upper()}: прогноз vs факт")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predictions_scatter.png", dpi=130)
    plt.close()


def plot_feature_correlations(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    targets_corr = corr.loc[TARGET_COLS, [c for c in numeric_cols if c not in TARGET_COLS]]
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(targets_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(targets_corr.index)))
    ax.set_yticklabels([t.upper() for t in targets_corr.index])
    ax.set_xticks(range(len(targets_corr.columns)))
    ax.set_xticklabels(targets_corr.columns, rotation=45, ha="right")
    for i in range(targets_corr.shape[0]):
        for j in range(targets_corr.shape[1]):
            ax.text(j, i, f"{targets_corr.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(targets_corr.values[i, j]) > 0.4 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="Корреляция Пирсона")
    ax.set_title("Корреляции числовых признаков с целевыми переменными")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_correlations.png", dpi=130)
    plt.close()


def demonstrate_recommender(predicted_metrics: dict[str, float]) -> str:
    analyzer = RuleAnalyzer()
    generator = RecommendationGenerator()
    deviations = analyzer.analyze(predicted_metrics)
    recs = generator.generate(deviations)
    return generator.format_for_display(recs)


def main() -> None:
    print("=" * 70)
    print("Преддипломная практика — экспериментальная проверка гипотезы")
    print("=" * 70)
    set_seeds()
    device = get_device()
    print(f"Устройство: {device}")
    print(f"Случайное зерно: {SEED}")
    print()

    # === 1. Данные ===
    print("[1/6] Генерация синтетического датасета...")
    df = generate_synthetic_data(N_SAMPLES)
    df.to_csv(DATA_DIR / "synthetic_campaigns.csv", index=False)
    print(f"  Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
    print(f"  Целевые статистики:")
    print(df[TARGET_COLS].describe().round(4).to_string())
    print()

    # === 2. Препроцессинг + разбиение ===
    print("[2/6] Препроцессинг и разбиение train/val/test...")
    X_raw = df.drop(columns=TARGET_COLS)
    y_raw = df[TARGET_COLS].values

    # train+val / test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_raw, y_raw, test_size=TEST_RATIO, random_state=SEED,
    )
    # train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_RATIO_OF_TRAIN, random_state=SEED,
    )

    preprocessor = Preprocessor()
    preprocessor.fit(X_train, y_train)
    X_train_p = preprocessor.transform_x(X_train)
    X_val_p = preprocessor.transform_x(X_val)
    X_test_p = preprocessor.transform_x(X_test)

    y_train_s = preprocessor.transform_y(y_train)
    y_val_s = preprocessor.transform_y(y_val)
    # y_test храним в оригинальной шкале для интерпретируемых метрик

    print(f"  Train: X={X_train_p.shape}, y={y_train_s.shape}")
    print(f"  Val:   X={X_val_p.shape}, y={y_val_s.shape}")
    print(f"  Test:  X={X_test_p.shape}, y={y_test.shape}")
    print(f"  Число признаков после OHE: {preprocessor.n_features}")
    print()

    # === 3. Линейная регрессия (baseline) ===
    print("[3/6] Обучение линейной регрессии (baseline)...")
    lr_model = LinearBaseline()
    lr_model.fit(X_train_p, y_train_s)
    y_lr_scaled = lr_model.predict(X_test_p)
    y_lr_pred = preprocessor.inverse_y(y_lr_scaled)
    print("  Готово.")
    print()

    # === 4. MLP ===
    print("[4/6] Обучение MLP...")
    print(f"  Архитектура: вход={X_train_p.shape[1]} → "
          f"{' → '.join(map(str, MLP_HIDDEN))} → {len(TARGET_COLS)}")
    print(f"  Dropout={MLP_DROPOUT}, LR={MLP_LR}, batch={MLP_BATCH_SIZE}, "
          f"epochs<={MLP_EPOCHS}")
    mlp_model, history = train_mlp(
        X_train_p, y_train_s,
        X_val_p, y_val_s,
        verbose=True,
    )
    y_mlp_scaled = predict(mlp_model, X_test_p)
    y_mlp_pred = preprocessor.inverse_y(y_mlp_scaled)
    print()

    # === 5. Оценка ===
    print("[5/6] Оценка на тестовой выборке...")
    per_target = evaluate_per_target(y_test, y_lr_pred, y_mlp_pred)

    # Печать таблицы
    print()
    print(f"{'Метрика':<8} {'RMSE LR':>12} {'RMSE MLP':>12} {'MAE LR':>12} "
          f"{'MAE MLP':>12} {'Δ RMSE, %':>12} {'Δ RMSE log, %':>15}")
    print("-" * 90)
    for name, metrics in per_target.items():
        print(f"{name.upper():<8} "
              f"{metrics['rmse_lr']:>12.5f} "
              f"{metrics['rmse_mlp']:>12.5f} "
              f"{metrics['mae_lr']:>12.5f} "
              f"{metrics['mae_mlp']:>12.5f} "
              f"{metrics['rmse_reduction_pct']:>12.2f} "
              f"{metrics['rmse_reduction_pct_logscale']:>15.2f}")
    print()

    # Сводный показатель: среднее снижение RMSE по лог-шкале (стабильнее)
    avg_reduction_log = float(np.mean([
        m["rmse_reduction_pct_logscale"] for m in per_target.values()
    ]))
    confirmed = avg_reduction_log >= HYPOTHESIS_RMSE_REDUCTION_PCT

    print(f"Среднее снижение RMSE по лог-шкале: {avg_reduction_log:.2f}%")
    print(f"Порог гипотезы: {HYPOTHESIS_RMSE_REDUCTION_PCT:.1f}%")
    print(f"Гипотеза: {'ПОДТВЕРЖДЕНА' if confirmed else 'НЕ ПОДТВЕРЖДЕНА'}")
    print()

    # === 6. Артефакты ===
    print("[6/6] Сохранение артефактов...")
    plot_loss_curves(history)
    plot_rmse_comparison(per_target)
    plot_predictions_scatter(y_test, y_lr_pred, y_mlp_pred)
    plot_feature_correlations(df)

    # Демонстрация модуля рекомендаций
    sample_predicted = {
        "ctr": float(y_mlp_pred[0, 0]),
        "cr": float(y_mlp_pred[0, 1]),
        "cpc": float(y_mlp_pred[0, 2]),
    }
    rec_text = demonstrate_recommender(sample_predicted)

    # Сохраняем JSON с полными результатами
    output = {
        "config": {
            "seed": SEED,
            "n_samples": N_SAMPLES,
            "device": str(device),
            "mlp_hidden": list(MLP_HIDDEN),
            "mlp_dropout": MLP_DROPOUT,
            "mlp_lr": MLP_LR,
            "mlp_batch_size": MLP_BATCH_SIZE,
            "mlp_weight_decay": MLP_WEIGHT_DECAY,
            "test_ratio": TEST_RATIO,
            "val_ratio_of_train": VAL_RATIO_OF_TRAIN,
            "n_features_after_ohe": preprocessor.n_features,
        },
        "train_sizes": {
            "train": int(X_train_p.shape[0]),
            "val": int(X_val_p.shape[0]),
            "test": int(X_test_p.shape[0]),
        },
        "per_target": per_target,
        "hypothesis": {
            "threshold_pct": HYPOTHESIS_RMSE_REDUCTION_PCT,
            "avg_rmse_reduction_pct_logscale": avg_reduction_log,
            "confirmed": confirmed,
        },
        "training_history": {
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "total_epochs": len(history.train_losses),
        },
        "recommender_demo": {
            "sample_predicted_metrics": sample_predicted,
            "recommendations_text": rec_text,
        },
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Сохранено: {RESULTS_DIR / 'metrics.json'}")
    print(f"  Сохранено: {RESULTS_DIR / 'loss_curves.png'}")
    print(f"  Сохранено: {RESULTS_DIR / 'rmse_comparison.png'}")
    print(f"  Сохранено: {RESULTS_DIR / 'predictions_scatter.png'}")
    print(f"  Сохранено: {RESULTS_DIR / 'feature_correlations.png'}")
    print()

    # Демо рекомендаций
    print("=" * 70)
    print("Демонстрация модуля рекомендаций для первого тестового примера")
    print("=" * 70)
    print(f"Прогнозные метрики: CTR={sample_predicted['ctr']:.4f}, "
          f"CR={sample_predicted['cr']:.4f}, "
          f"CPC={sample_predicted['cpc']:.2f} руб.")
    print()
    print(rec_text)


if __name__ == "__main__":
    main()
