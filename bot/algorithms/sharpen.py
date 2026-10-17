"""Нерезкое маскирование (unsharp masking) с подавлением ореолов."""

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


def clamp_to_local_range(
    values: np.ndarray,
    reference: np.ndarray,
    radius: int = 1,
    overshoot: float = 0.1,
    min_overshoot: float = 0.3,
) -> np.ndarray:
    """Ограничивает ``values`` локальным диапазоном ``reference`` (окно (2r+1)²).

    Допускается выход за локальные min/max на долю ``overshoot`` локального
    размаха (но не меньше ``min_overshoot``). Так края остаются чёткими,
    а светлые и тёмные каймы-ореолы, которые даёт резкость или
    деконволюция, срезаются.
    """
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    local_min = cv2.erode(reference, kernel)
    local_max = cv2.dilate(reference, kernel)
    allowance = np.maximum(overshoot * (local_max - local_min), min_overshoot)
    return np.clip(values, local_min - allowance, local_max + allowance)


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.5,
    amount: float = 0.8,
    threshold: float = 0.8,
    halo_control: float | None = 0.1,
) -> np.ndarray:
    """Классическое нерезкое маскирование по яркости.

    Результат = L + amount * (L - blur(L)) в пространстве Lab, поэтому
    цветных ореолов нет. Пиксели, где |L - blur(L)| < ``threshold``
    (в единицах L, 0..100), не усиливаются, чтобы не вытягивать шум
    на ровных участках. ``halo_control`` — допустимый выход за локальный
    диапазон исходника (доля размаха); ``None`` отключает подавление ореолов.
    """
    ensure_bgr_u8(image)
    lab = to_lab(image)
    l_ch = lab[:, :, 0]
    detail = l_ch - cv2.GaussianBlur(l_ch, (0, 0), sigma)
    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0.0
    sharpened = l_ch + amount * detail
    if halo_control is not None:
        sharpened = clamp_to_local_range(sharpened, l_ch, overshoot=halo_control)
    lab[:, :, 0] = np.clip(sharpened, 0.0, 100.0)
    return from_lab(lab)
