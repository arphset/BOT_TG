"""Нерезкое маскирование (unsharp masking)."""

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.5,
    amount: float = 0.8,
    threshold: float = 0.8,
) -> np.ndarray:
    """Классическое нерезкое маскирование по яркости.

    Результат = L + amount * (L - blur(L)) в пространстве Lab, поэтому
    цветных ореолов нет. Пиксели, где |L - blur(L)| < ``threshold``
    (в единицах L, 0..100), не усиливаются, чтобы не вытягивать шум
    на ровных участках.
    """
    ensure_bgr_u8(image)
    lab = to_lab(image)
    l_ch = lab[:, :, 0]
    detail = l_ch - cv2.GaussianBlur(l_ch, (0, 0), sigma)
    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0.0
    lab[:, :, 0] = np.clip(l_ch + amount * detail, 0.0, 100.0)
    return from_lab(lab)
