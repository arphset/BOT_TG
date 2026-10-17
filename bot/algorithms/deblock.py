"""Удаление блочных артефактов JPEG (деблокинг).

JPEG кодирует картинку блоками 8×8; при сильном сжатии на границах блоков
появляются ступеньки («квадратики»). Детектор ищет такую сетку и её сдвиг
(кадр мог быть обрезан после сжатия); если кадр масштабировали, сетки нет,
и деблокинг не нужен. Фильтр сглаживает только небольшие ступеньки на
границах блоков в ровных местах — настоящие края остаются нетронутыми.
"""

from __future__ import annotations

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab

BLOCK = 8


def _phase_ratio(diff_profile: np.ndarray) -> tuple[float, int]:
    """Во сколько раз перепады на «самой блочной» фазе сильнее остальных и какая это фаза."""
    n = len(diff_profile) - len(diff_profile) % BLOCK
    if n < BLOCK * 4:
        return 1.0, 0
    phases = diff_profile[:n].reshape(-1, BLOCK).mean(axis=0)
    best = int(np.argmax(phases))
    others = np.delete(phases, best).mean()
    return float(phases[best] / max(others, 1e-6)), best


def blockiness(image: np.ndarray) -> tuple[float, int, int]:
    """Оценка блочности: (коэффициент, сдвиг сетки по x, по y).

    Коэффициент ≈ 1 у несжатых и масштабированных кадров, растёт с силой
    сжатия. Перепады считаются по медиане вдоль строки/столбца, чтобы
    редкие сильные края не влияли.
    """
    ensure_bgr_u8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    # Ровные места информативнее: ограничиваем вклад сильных краёв.
    dx = np.minimum(dx, 20.0)
    dy = np.minimum(dy, 20.0)
    rx, phase_x = _phase_ratio(dx.mean(axis=0))
    ry, phase_y = _phase_ratio(dy.mean(axis=1))
    # Перепад между столбцами x и x+1 ↔ граница блока перед столбцом x+1.
    return (rx + ry) / 2.0, (phase_x + 1) % BLOCK, (phase_y + 1) % BLOCK


def _filter_boundaries(ch: np.ndarray, offset: int, threshold: float) -> np.ndarray:
    """Сглаживает вертикальные границы блоков (между столбцами x-1 и x) в канале ``ch``."""
    out = ch.copy()
    w = ch.shape[1]
    xs = np.arange(offset if offset >= 3 else offset + BLOCK, w - 3, BLOCK)
    if xs.size == 0:
        return out
    p2, p1, p0 = ch[:, xs - 3], ch[:, xs - 2], ch[:, xs - 1]
    q0, q1, q2 = ch[:, xs], ch[:, xs + 1], ch[:, xs + 2]
    step = q0 - p0
    activity = np.abs(p0 - p1) + np.abs(q1 - q0)
    # Полное сглаживание для малых ступенек в ровных местах, плавный спад к порогу.
    w_step = np.clip((threshold - np.abs(step)) / (0.5 * threshold), 0.0, 1.0)
    w_flat = np.clip((threshold - activity) / (0.5 * threshold), 0.0, 1.0)
    d = step * w_step * w_flat
    # Ступенька превращается в линейный переход на 6 пикселей (шаг d/6 на пиксель).
    out[:, xs - 3] = p2 + d / 12.0
    out[:, xs - 2] = p1 + 3.0 * d / 12.0
    out[:, xs - 1] = p0 + 5.0 * d / 12.0
    out[:, xs] = q0 - 5.0 * d / 12.0
    out[:, xs + 1] = q1 - 3.0 * d / 12.0
    out[:, xs + 2] = q2 - d / 12.0
    return out


def deblock(image: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Деблокинг по найденной 8×8 сетке (яркость и цвет, Lab).

    ``threshold`` — максимальная ступенька (в единицах L, 0..100), которая
    считается артефактом; по умолчанию берётся из типичной ступеньки на
    границах блоков этого кадра.
    """
    ensure_bgr_u8(image)
    _, ox, oy = blockiness(image)
    lab = to_lab(image)
    for c in range(3):
        ch = lab[:, :, c]
        if threshold is None:
            # Типичная крупная ступенька на границах блоков (90-й перцентиль) задаёт порог:
            # всё, что заметно больше, — настоящие края. Берутся именно столбцы границ.
            xs = np.arange(ox if ox >= 1 else BLOCK, ch.shape[1], BLOCK)
            typical = float(np.percentile(np.abs(ch[:, xs] - ch[:, xs - 1]), 90)) if xs.size else 1.0
            thr = float(np.clip(typical * 1.5, 1.5, 10.0))
        else:
            thr = threshold
        ch = _filter_boundaries(ch, ox, thr)
        ch = _filter_boundaries(ch.T.copy(), oy, thr).T
        lab[:, :, c] = ch
    return from_lab(lab)


def is_blocky(image: np.ndarray, min_ratio: float = 1.3) -> bool:
    return blockiness(image)[0] >= min_ratio
