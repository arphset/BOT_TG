"""Управляемый фильтр (guided filter, He et al.) на box-фильтрах OpenCV.

Нужен для дымки (уточнение карты пропускания) и тонокоррекции (базовый слой
без ореолов). В opencv-python-headless нет модуля ximgproc, поэтому своя
реализация: O(N), не зависит от радиуса.
"""

import cv2
import numpy as np


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Сглаживает ``src``, сохраняя края ``guide`` (оба — float32, один канал)."""
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)
