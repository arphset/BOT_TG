"""Локальный контраст: CLAHE в канале L пространства Lab."""

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


def clahe_lab(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Применяет CLAHE только к яркости (L), цветовые каналы не трогает.

    CLAHE в OpenCV работает с 8-битным L, поэтому L квантуется, выравнивается,
    а дробная часть исходного L добавляется обратно — без лишних потерь точности.
    """
    ensure_bgr_u8(image)
    lab = to_lab(image)
    l_scaled = lab[:, :, 0] * 2.55
    l_u8 = np.clip(l_scaled + 0.5, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l_u8).astype(np.float32) + (l_scaled - l_u8)
    lab[:, :, 0] = np.clip(l_eq / 2.55, 0.0, 100.0)
    return from_lab(lab)
