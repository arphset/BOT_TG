"""Выравнивание завала горизонта.

Ищем длинные прямые (преобразование Хафа по контурам). Почти горизонтальные
линии (горизонт, полки, карнизы, столы) — главный источник: если они
согласованно наклонены на 0,5–6°, кадр поворачивается и обрезается до
наибольшего прямоугольника тех же пропорций. Вертикали используются только
когда горизонтальных линий мало: из-за перспективы они сходятся и могут
обмануть. Если линии противоречат друг другу, кадр не трогается.
"""

from __future__ import annotations

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8

MIN_TILT = 0.5
MAX_TILT = 6.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, wts = values[order], weights[order]
    cum = np.cumsum(wts)
    return float(v[np.searchsorted(cum, cum[-1] / 2.0)])


def _consensus(deviations: np.ndarray, lengths: np.ndarray, min_support: float) -> float | None:
    """Согласованный наклон: взвешенная медиана, если линий с ним рядом достаточно."""
    if deviations.size == 0 or lengths.sum() < min_support:
        return None
    med = _weighted_median(deviations, lengths)
    near = np.abs(deviations - med) < 1.0
    if lengths[near].sum() < min_support or lengths[near].sum() < 0.6 * lengths.sum():
        return None
    return float(np.average(deviations[near], weights=lengths[near]))


def detect_tilt(image: np.ndarray, max_side: int = 1000) -> float | None:
    """Наклон кадра в градусах (поворот на это значение выравнивает) или None."""
    ensure_bgr_u8(image)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    short = min(gray.shape)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=80, minLineLength=int(short * 0.2), maxLineGap=int(short * 0.02) + 2
    )
    if lines is None:
        return None
    seg = lines[:, 0, :].astype(np.float32)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))  # (-180, 180], y вниз
    angle = (angle + 90.0) % 180.0 - 90.0  # (-90, 90]
    horiz = np.abs(angle) < MAX_TILT + 2
    vert = np.abs(np.abs(angle) - 90.0) < MAX_TILT + 2
    min_support = short * 0.6
    tilt = _consensus(angle[horiz], length[horiz], min_support)
    if tilt is None:
        v_dev = np.where(angle[vert] > 0, angle[vert] - 90.0, angle[vert] + 90.0)
        tilt = _consensus(v_dev, length[vert], min_support * 1.5)
    if tilt is None or not (MIN_TILT <= abs(tilt) <= MAX_TILT):
        return None
    return tilt


def straighten(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Поворачивает кадр на ``angle`` (по умолчанию — найденный наклон) и обрезает
    до наибольшего прямоугольника тех же пропорций без пустых углов."""
    ensure_bgr_u8(image)
    if angle is None:
        angle = detect_tilt(image)
    if angle is None or abs(angle) < 1e-3:
        return image.copy()
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    t = np.radians(abs(angle))
    k = 1.0 / (np.cos(t) + np.sin(t) * max(w / h, h / w))
    cw, ch = int(w * k), int(h * k)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return rotated[y0 : y0 + ch, x0 : x0 + cw].copy()
