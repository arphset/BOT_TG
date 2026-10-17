"""Цветокоррекция: баланс белого, автоуровни, вибрация.

Все коррекции намеренно осторожные: фото с задуманной тёплой или «киношной»
гаммой не должны превращаться в нейтрально-серые. Поэтому баланс белого
исправляет сдвиг лишь частично и с ограничением коэффициентов, а уровни
растягиваются только если гистограмма действительно сжата.
"""

from __future__ import annotations

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


def _valid_mask(image: np.ndarray) -> np.ndarray:
    """Пиксели для оценки цвета: без пересветов и глубоких теней."""
    lo = image.min(axis=2)
    hi = image.max(axis=2)
    return (hi < 245) & (lo > 10)


def white_balance(
    image: np.ndarray,
    strength: float = 0.6,
    max_gain: float = 1.25,
    min_cast: float = 0.03,
) -> np.ndarray:
    """Баланс белого по модели «серого мира» с защитой от перекоррекции.

    * оцениваются только средние тона (без пересветов и теней);
    * коэффициенты ограничены диапазоном [1/max_gain, max_gain];
    * если сдвиг меньше ``min_cast`` (3 %), изображение не меняется;
    * исправляется лишь доля ``strength`` найденного сдвига.
    """
    ensure_bgr_u8(image)
    mask = _valid_mask(image)
    if mask.sum() < 0.05 * mask.size:
        return image.copy()
    linear = image.astype(np.float32) / 255.0
    means = np.array([linear[..., c][mask].mean() for c in range(3)], dtype=np.float32)
    gray = float(means.mean())
    if gray <= 1e-6:
        return image.copy()
    gains = gray / np.maximum(means, 1e-6)
    if np.abs(gains - 1.0).max() < min_cast:
        return image.copy()
    gains = np.clip(1.0 + strength * (gains - 1.0), 1.0 / max_gain, max_gain)
    out = linear * gains.reshape(1, 1, 3)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def auto_levels(
    image: np.ndarray,
    clip_percent: float = 0.4,
    max_stretch: float = 1.6,
    min_range: float = 0.92,
) -> np.ndarray:
    """Растяжение яркости (L в Lab) между перцентилями, цвет не трогается.

    Если диапазон яркости уже занимает ≥ ``min_range`` от полного, ничего не
    делается. Растяжение ограничено ``max_stretch``, чтобы тёмные или
    туманные по замыслу кадры не превращались в контрастные.
    """
    ensure_bgr_u8(image)
    lab = to_lab(image)
    l_ch = lab[:, :, 0]
    lo, hi = np.percentile(l_ch, [clip_percent, 100.0 - clip_percent])
    span = float(hi - lo)
    if span <= 1e-3 or span >= min_range * 100.0:
        return image.copy()
    scale = min(100.0 / span, max_stretch)
    # Растягиваем вокруг середины текущего диапазона (общая яркость почти не меняется),
    # затем сдвигаем, чтобы новый диапазон поместился в 0..100.
    center = float(lo + hi) / 2.0
    half = span * scale / 2.0
    target = min(max(center, half), 100.0 - half)
    lab[:, :, 0] = np.clip((l_ch - center) * scale + target, 0.0, 100.0)
    return from_lab(lab)


def _skin_weight(lab: np.ndarray) -> np.ndarray:
    """Мягкая маска тонов кожи в Lab: оттенок 30–70° полностью, спад до 20° и 80°."""
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    hue = np.degrees(np.arctan2(b, a))
    chroma = np.sqrt(a * a + b * b)
    in_hue = np.clip((30.0 - np.abs(hue - 50.0)) / 10.0, 0.0, 1.0)
    in_chroma = np.clip((chroma - 5.0) / 10.0, 0.0, 1.0) * np.clip((60.0 - chroma) / 20.0, 0.0, 1.0)
    return in_hue * in_chroma


def vibrance(image: np.ndarray, amount: float = 0.25, protect_skin: bool = True) -> np.ndarray:
    """«Вибрация»: усиливает насыщенность бледных цветов сильнее, чем ярких.

    Насыщенные цвета почти не меняются, кожа защищена (иначе лица краснеют).
    """
    ensure_bgr_u8(image)
    lab = to_lab(image)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a * a + b * b)
    # бледным (chroma≈0..20) — полный эффект, насыщенным (≥60) — почти ноль
    weight = np.clip(1.0 - chroma / 60.0, 0.0, 1.0)
    if protect_skin:
        weight *= 1.0 - 0.8 * _skin_weight(lab)
    boost = 1.0 + amount * weight
    lab[:, :, 1] = a * boost
    lab[:, :, 2] = b * boost
    return from_lab(lab)


def color_correct(
    image: np.ndarray,
    wb_strength: float = 0.6,
    vibrance_amount: float = 0.2,
) -> np.ndarray:
    """Баланс белого → автоуровни → вибрация (шаг «цветокоррекция» в «Авто»)."""
    out = white_balance(image, strength=wb_strength)
    out = auto_levels(out)
    if vibrance_amount > 0:
        out = vibrance(out, amount=vibrance_amount)
    return out


def chroma_cast(image: np.ndarray) -> float:
    """Величина цветового сдвига средних тонов (0 — нейтрально), для отчёта и тестов."""
    ensure_bgr_u8(image)
    mask = _valid_mask(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask.sum() == 0:
        return 0.0
    a = lab[..., 1][mask].mean() - 128.0
    b = lab[..., 2][mask].mean() - 128.0
    return float(np.hypot(a, b))
