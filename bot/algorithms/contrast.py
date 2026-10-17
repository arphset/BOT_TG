"""Локальный контраст: CLAHE в канале L пространства Lab, с адаптацией к кадру."""

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


def _clahe_l(l_ch: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """CLAHE для L (0..100, float). OpenCV работает с 8 битами, поэтому L квантуется,
    выравнивается, а дробная часть добавляется обратно — без лишних потерь точности."""
    l_scaled = l_ch * 2.55
    l_u8 = np.clip(l_scaled + 0.5, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l_u8).astype(np.float32) + (l_scaled - l_u8)
    return np.clip(l_eq / 2.55, 0.0, 100.0)


def adaptive_clip_limit(l_ch: np.ndarray, base: float) -> float:
    """Сила CLAHE по контрасту кадра: плоский кадр — сильнее, контрастный — слабее.

    Ориентир — стандартное отклонение L: ~18 у «обычного» кадра. Коэффициент
    ограничен 0,6–1,4 от ``base``.
    """
    std = float(l_ch.std())
    factor = float(np.clip(18.0 / max(std, 1e-3), 0.6, 1.4))
    return base * factor


def flatness_weight(l_ch: np.ndarray, noise_l: float, radius: float = 4.0) -> np.ndarray:
    """Вес применения CLAHE: 0 на ровных участках (локальный разброс на уровне шума),
    1 там, где есть структура. Плавный переход, маска размыта, чтобы не было границ."""
    mean = cv2.GaussianBlur(l_ch, (0, 0), radius)
    var = cv2.GaussianBlur(l_ch * l_ch, (0, 0), radius) - mean * mean
    local_std = np.sqrt(np.maximum(var, 0.0))
    lo = max(noise_l * 1.5, 0.4)
    hi = lo * 3.0
    weight = np.clip((local_std - lo) / (hi - lo), 0.0, 1.0)
    weight = 0.15 + 0.85 * weight  # ровные участки получают малую долю эффекта, а не ноль
    return cv2.GaussianBlur(weight, (0, 0), radius * 2)


def clahe_lab(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
    adaptive: bool = False,
    noise_sigma: float | None = None,
) -> np.ndarray:
    """Применяет CLAHE только к яркости (L), цветовые каналы не трогает.

    ``adaptive=True``: сила подбирается по контрасту кадра, а на ровных
    участках (небо, стены) эффект ослабляется, чтобы не вытягивать шум
    и ступеньки. ``noise_sigma`` — оценка шума в единицах серого (0..255);
    если не задана, считается по кадру.
    """
    ensure_bgr_u8(image)
    lab = to_lab(image)
    l_ch = lab[:, :, 0]
    if not adaptive:
        lab[:, :, 0] = _clahe_l(l_ch, clip_limit, tile_grid)
        return from_lab(lab)

    if noise_sigma is None:
        from bot.algorithms.denoise import estimate_noise

        noise_sigma = estimate_noise(image)
    noise_l = noise_sigma * 100.0 / 255.0
    l_eq = _clahe_l(l_ch, adaptive_clip_limit(l_ch, clip_limit), tile_grid)
    weight = flatness_weight(l_ch, noise_l)
    lab[:, :, 0] = l_ch + weight * (l_eq - l_ch)
    return from_lab(lab)
