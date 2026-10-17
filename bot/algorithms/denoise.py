"""Шумоподавление non-local means."""

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8


def estimate_noise(image: np.ndarray) -> float:
    """Оценка σ гауссова шума по методу Immerkær (по яркости).

    Быстрая оценка, используется режимом «Авто», чтобы решить,
    нужно ли шумоподавление и с какой силой.
    """
    ensure_bgr_u8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    inner = np.abs(response[1:-1, 1:-1])
    return float(np.sqrt(np.pi / 2.0) * inner.sum() / (6.0 * (w - 2) * (h - 2)))


def nl_means(
    image: np.ndarray,
    strength: float = 7.0,
    color_strength: float | None = None,
    template_window: int = 7,
    search_window: int = 15,
) -> np.ndarray:
    """Цветное шумоподавление non-local means (``fastNlMeansDenoisingColored``).

    ``strength`` — параметр h для яркости; ``color_strength`` — для цвета
    (по умолчанию равен ``strength``). Окно поиска 15 вместо стандартного 21
    заметно быстрее при почти том же качестве.
    """
    ensure_bgr_u8(image)
    h_color = strength if color_strength is None else color_strength
    return cv2.fastNlMeansDenoisingColored(
        image, None, float(strength), float(h_color), template_window, search_window
    )
