"""Модуль генерации синтетических данных рекламных кампаний.

Генерирует датасет с правдоподобными нелинейными зависимостями между
параметрами кампании и целевыми метриками (CTR, CR, CPC).

Структура признакового пространства основана на:
- наблюдаемых характеристиках открытого датасета Avazu CTR Prediction;
- описаниях признаков кампаний в документации Google Ads API и
  Яндекс.Директ.

Параметры распределений и формы зависимостей подобраны так, чтобы
воспроизвести нелинейные закономерности, типичные для рекламных систем:
- сатурация CTR от quality_score (сигмоид);
- мультипликативные взаимодействия platform x ad_type;
- зависимость CTR от часа суток с разными пиками для разных аудиторий;
- убывающая отдача (логарифм) от бюджета;
- сверхлинейная зависимость CPC от уровня конкуренции.

Эти нелинейности позволяют MLP превосходить линейную регрессию.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEED


def generate_synthetic_data(n_samples: int, seed: int = SEED) -> pd.DataFrame:
    """Генерирует синтетический датасет рекламных кампаний.

    Args:
        n_samples: Количество строк (наблюдений) в датасете.
        seed: Зерно генератора псевдослучайных чисел для воспроизводимости.

    Returns:
        pandas.DataFrame с признаками и тремя целевыми переменными:
        ctr, cr, cpc.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    # =========================================================================
    # 1. Категориальные признаки
    # =========================================================================

    platforms = rng.choice(
        ["Google", "Yandex", "VK", "Telegram"],
        size=n,
        p=[0.35, 0.35, 0.20, 0.10],
    )
    ad_types = rng.choice(
        ["search", "banner", "video", "native"],
        size=n,
        p=[0.40, 0.30, 0.15, 0.15],
    )
    devices = rng.choice(
        ["desktop", "mobile", "tablet"],
        size=n,
        p=[0.35, 0.55, 0.10],
    )
    regions = rng.choice(
        ["Moscow", "SPb", "regions", "abroad"],
        size=n,
        p=[0.30, 0.15, 0.50, 0.05],
    )
    audience = rng.choice(
        ["B2B", "B2C_young", "B2C_adult", "B2C_senior"],
        size=n,
        p=[0.20, 0.30, 0.35, 0.15],
    )

    # =========================================================================
    # 2. Временные и числовые признаки
    # =========================================================================

    hour = rng.integers(0, 24, n)
    day_of_week = rng.integers(0, 7, n)
    duration_days = rng.integers(1, 91, n)

    # Бюджет — log-нормальное распределение (медиана ~ 22 тыс. руб.)
    budget = np.clip(rng.lognormal(mean=10.0, sigma=1.0, size=n), 1_000, 1_000_000)

    num_keywords = rng.integers(1, 101, n)

    # Оценка качества рекламы (Google Quality Score, 0–10)
    quality_score = np.clip(rng.normal(6.0, 1.5, n), 0, 10)

    # Оценка качества креатива (0–1)
    creative_score = np.clip(rng.normal(0.6, 0.2, n), 0, 1)

    # Исторические метрики (лаги)
    hist_ctr_7d = np.clip(rng.lognormal(mean=-4.0, sigma=0.5, size=n), 0.001, 0.20)
    hist_ctr_30d = np.clip(
        hist_ctr_7d * rng.normal(1.0, 0.10, n),
        0.001, 0.20,
    )
    hist_cr_7d = np.clip(rng.lognormal(mean=-3.5, sigma=0.4, size=n), 0.005, 0.25)
    hist_cpc_7d = np.clip(rng.lognormal(mean=4.0, sigma=0.6, size=n), 5.0, 1000.0)

    competitor_count = rng.integers(0, 51, n)
    bid_amount = np.clip(rng.lognormal(mean=4.0, sigma=0.5, size=n), 5.0, 500.0)

    df = pd.DataFrame({
        "platform": platforms,
        "ad_type": ad_types,
        "device": devices,
        "region": regions,
        "audience_segment": audience,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "duration_days": duration_days,
        "budget_rub": budget,
        "num_keywords": num_keywords,
        "quality_score": quality_score,
        "creative_score": creative_score,
        "historical_ctr_7d": hist_ctr_7d,
        "historical_ctr_30d": hist_ctr_30d,
        "historical_cr_7d": hist_cr_7d,
        "historical_cpc_7d": hist_cpc_7d,
        "competitor_count": competitor_count,
        "bid_amount_rub": bid_amount,
    })

    # =========================================================================
    # 3. Генерация целевых переменных с заложенными нелинейностями
    # =========================================================================

    # --- CTR ---
    # Сатурация CTR от quality_score (сигмоид)
    ctr_base = 0.08 / (1.0 + np.exp(-(df["quality_score"].values - 5.0) / 1.5))

    # Мультипликаторы по платформе и типу объявления
    platform_ctr_mult = df["platform"].map({
        "Google": 1.30, "Yandex": 1.10, "VK": 0.70, "Telegram": 0.90,
    }).values
    adtype_ctr_mult = df["ad_type"].map({
        "search": 1.60, "banner": 0.50, "video": 1.00, "native": 1.20,
    }).values

    # Взаимодействие platform x ad_type (поисковая реклама в Google усиливается)
    interact_search_google = ((df["platform"] == "Google") & (df["ad_type"] == "search")).values
    interact_boost = np.where(interact_search_google, 1.30, 1.0)

    # Взаимодействие час x аудитория
    business_hours = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 18)).values
    evening = ((df["hour_of_day"] >= 18) & (df["hour_of_day"] <= 23)).values
    aud = df["audience_segment"].values

    hour_aud_mult = np.ones(n)
    hour_aud_mult = np.where((aud == "B2B") & business_hours, 1.40, hour_aud_mult)
    hour_aud_mult = np.where((aud == "B2B") & ~business_hours, 0.65, hour_aud_mult)
    hour_aud_mult = np.where((aud == "B2C_young") & evening, 1.35, hour_aud_mult)
    hour_aud_mult = np.where((aud == "B2C_senior") & business_hours, 1.15, hour_aud_mult)

    # Устройство
    device_mult = df["device"].map({
        "desktop": 1.00, "mobile": 1.10, "tablet": 0.80,
    }).values

    # Креатив (линейный эффект 0.5–1.5)
    creative_impact = 0.5 + df["creative_score"].values

    # Исторический CTR (lag-фича — сильный предиктор)
    hist_contrib_ctr = df["historical_ctr_7d"].values / 0.02
    hist_factor_ctr = 0.5 + 0.5 * hist_contrib_ctr

    # Мультипликативный шум
    noise_ctr = rng.lognormal(mean=0.0, sigma=0.15, size=n)

    ctr = (
        ctr_base
        * platform_ctr_mult
        * adtype_ctr_mult
        * interact_boost
        * hour_aud_mult
        * device_mult
        * creative_impact
        * hist_factor_ctr
        * noise_ctr
    )
    ctr = np.clip(ctr, 0.0001, 0.50)

    # --- CR (Conversion Rate) ---
    cr_base = 0.10 / (1.0 + np.exp(-(df["quality_score"].values - 6.0) / 1.2))

    # Эффект бюджета — убывающая отдача (логарифм)
    budget_log = np.log1p(df["budget_rub"].values) / 12.0
    budget_factor = 0.5 + 0.5 * budget_log

    # Аудитория (B2B конвертится лучше)
    audience_cr_mult = df["audience_segment"].map({
        "B2B": 1.40, "B2C_young": 0.70, "B2C_adult": 1.20, "B2C_senior": 1.00,
    }).values

    # Число ключевых слов — оптимум около 40
    keyword_factor = 0.5 + np.exp(-((df["num_keywords"].values - 40.0) / 30.0) ** 2)

    # Исторический CR
    hist_contrib_cr = df["historical_cr_7d"].values / 0.05
    hist_factor_cr = 0.4 + 0.6 * hist_contrib_cr

    noise_cr = rng.lognormal(mean=0.0, sigma=0.20, size=n)

    cr = (
        cr_base
        * budget_factor
        * audience_cr_mult
        * keyword_factor
        * hist_factor_cr
        * noise_cr
    )
    cr = np.clip(cr, 0.0005, 0.50)

    # --- CPC (Cost Per Click) ---
    cpc_platform_mult = df["platform"].map({
        "Google": 1.60, "Yandex": 1.30, "VK": 0.70, "Telegram": 0.80,
    }).values
    cpc_adtype_mult = df["ad_type"].map({
        "search": 1.50, "banner": 0.80, "video": 1.20, "native": 1.00,
    }).values

    # Конкуренция — сверхлинейная зависимость (степень 1.3)
    comp_factor = 1.0 + (df["competitor_count"].values / 10.0) ** 1.3

    # Скидка от quality_score
    bid_contribution = df["bid_amount_rub"].values * (1.20 - df["quality_score"].values / 20.0)

    # Исторический CPC
    hist_contrib_cpc = df["historical_cpc_7d"].values / 50.0
    hist_factor_cpc = 0.5 + 0.5 * hist_contrib_cpc

    noise_cpc = rng.lognormal(mean=0.0, sigma=0.25, size=n)

    cpc = (
        bid_contribution
        * cpc_platform_mult
        * cpc_adtype_mult
        * comp_factor
        * hist_factor_cpc
        * noise_cpc
        * 0.40
    )
    cpc = np.clip(cpc, 1.0, 5000.0)

    df["ctr"] = ctr
    df["cr"] = cr
    df["cpc"] = cpc

    return df


if __name__ == "__main__":
    from config import DATA_DIR, N_SAMPLES

    print(f"Generating {N_SAMPLES} samples...")
    df = generate_synthetic_data(N_SAMPLES)

    out_path = DATA_DIR / "synthetic_campaigns.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Shape: {df.shape}")
    print("\nTargets summary:")
    print(df[["ctr", "cr", "cpc"]].describe())
