"""Тени и света: локальная тонокоррекция.

Яркость раскладывается на базовый слой (освещение, крупные формы) и детали.
Тени поднимаются и выбитые света приглушаются только в базовом слое, детали
возвращаются как есть — поэтому лица в тени светлеют без потери фактуры,
а небо и окна получают обратно рисунок. Базовый слой строится управляемым
фильтром, который держит края: вокруг тёмных фигур на светлом фоне нет ореолов.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bot.algorithms._guided import guided_filter
from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


@dataclass(frozen=True)
class ToneEstimate:
    shadows: float  # 0..1 — насколько нужно поднять тени
    highlights: float  # 0..1 — насколько нужно приглушить света

    @property
    def needed(self) -> bool:
        return self.shadows > 0.05 or self.highlights > 0.05


def _base_layer(l_norm: np.ndarray) -> np.ndarray:
    h, w = l_norm.shape
    radius = max(4, int(min(h, w) * 0.02))
    return np.clip(guided_filter(l_norm, l_norm, radius, 0.01), 0.0, 1.0)


def estimate_tone(image: np.ndarray, max_side: int = 800) -> ToneEstimate:
    """Сколько теней и светов нужно вытянуть.

    Тени: доля кадра, где освещение (базовый слой) темнее 25 %. Света: доля почти выбитых
    областей (базовый слой светлее 88 %).
    """
    ensure_bgr_u8(image)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    base = _base_layer(to_lab(small)[:, :, 0] / 100.0)
    dark_fraction = float((base < 0.25).mean())
    bright_fraction = float((base > 0.88).mean())
    # Подъём по доле тёмного, а не по средней яркости: при контровом свете яркий фон
    # поднимает среднее, но тёмный объект всё равно нужно вытянуть.
    shadows = float(np.clip((dark_fraction - 0.08) * 2.5, 0.0, 1.0))
    highlights = float(np.clip((bright_fraction - 0.05) * 3.0, 0.0, 1.0))
    return ToneEstimate(shadows=shadows, highlights=highlights)


def shadows_highlights(
    image: np.ndarray,
    shadows: float | None = None,
    highlights: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Поднимает тени и приглушает света по базовому слою яркости.

    ``shadows``/``highlights`` (0..1) по умолчанию берутся из оценки кадра.
    Максимальный подъём теней ≈ +16 % яркости около 25 % уровня, приглушение
    светов ≈ −12 % около 75–90 %. Цвет (a/b) слегка подстраивается, чтобы
    поднятые тени не выглядели серыми.
    """
    ensure_bgr_u8(image)
    if shadows is None or highlights is None:
        est = estimate_tone(image)
        shadows = est.shadows if shadows is None else shadows
        highlights = est.highlights if highlights is None else highlights
    shadows = float(np.clip(shadows * scale, 0.0, 1.0))
    highlights = float(np.clip(highlights * scale, 0.0, 1.0))
    if shadows <= 0.01 and highlights <= 0.01:
        return image.copy()

    lab = to_lab(image)
    l_norm = lab[:, :, 0] / 100.0
    base = _base_layer(l_norm)
    detail = l_norm - base
    lift = shadows * 1.5 * base * (1.0 - base) ** 3  # пик ≈ +0,16 при base≈0,25
    compress = highlights * 2.0 * base**3 * (1.0 - base)  # пик ≈ −0,21·0,6 при base≈0,75
    new_base = np.clip(base + lift - 0.6 * compress, 0.0, 1.0)
    new_l = np.clip(new_base + detail, 0.0, 1.0)
    # Подстройка насыщенности: при подъёме яркости цвет сохраняет относительную насыщенность.
    ratio = np.clip((new_base + 0.05) / (base + 0.05), 0.8, 1.6)
    chroma_gain = 1.0 + 0.5 * (ratio - 1.0)
    lab[:, :, 0] = new_l * 100.0
    lab[:, :, 1] *= chroma_gain
    lab[:, :, 2] *= chroma_gain
    return from_lab(lab)
