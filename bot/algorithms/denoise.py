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


BAND_EDGES = np.array([0, 40, 80, 130, 190, 256], dtype=np.float32)


def estimate_noise_by_luma(image: np.ndarray, min_fraction: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Оценка шума по зонам яркости: (центры зон 0..255, σ в каждой зоне).

    Реальный шум камеры сильнее в тенях, чем в светах. Оценка робастная —
    по медиане отклика фильтра Immerkær (дисперсия отклика 36σ²), поэтому
    края и текстура её почти не завышают. Зоны, где пикселей меньше
    ``min_fraction``, заполняются соседними значениями.
    """
    ensure_bgr_u8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = np.abs(cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT))
    # Яркость зоны берём по сглаженной картинке, чтобы шум не перекидывал пиксели между зонами.
    level = cv2.GaussianBlur(gray, (0, 0), 2.0)
    centers = (BAND_EDGES[:-1] + BAND_EDGES[1:]) / 2.0
    sigmas = np.full(len(centers), np.nan, dtype=np.float32)
    total = gray.size
    for i in range(len(centers)):
        mask = (level >= BAND_EDGES[i]) & (level < BAND_EDGES[i + 1])
        if mask.sum() >= min_fraction * total:
            sigmas[i] = 1.4826 * float(np.median(response[mask])) / 6.0
    if np.all(np.isnan(sigmas)):
        sigmas[:] = estimate_noise(image)
    else:
        valid = ~np.isnan(sigmas)
        sigmas = np.interp(centers, centers[valid], sigmas[valid]).astype(np.float32)
    return centers, sigmas


def denoise_by_luma(
    image: np.ndarray,
    scale: float = 1.05,
    min_strength: float = 2.5,
    max_strength: float = 10.0,
    min_weight: float = 0.3,
) -> np.ndarray:
    """Шумоподавление с силой по зонам яркости.

    NLM запускается один раз с силой под самую шумную зону, а результат
    подмешивается к исходнику с весом σ_зоны / σ_max: шумные тени чистятся
    полностью, чистые света — частично (не ниже ``min_weight``), чтобы там
    не терялись детали. По времени — как обычный NLM.
    """
    ensure_bgr_u8(image)
    centers, sigmas = estimate_noise_by_luma(image)
    s_max = float(sigmas.max())
    strength = float(np.clip(s_max * scale, min_strength, max_strength))
    denoised = nl_means(image, strength=strength)
    if s_max <= 1e-6:
        return denoised
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    level = cv2.GaussianBlur(gray, (0, 0), 3.0)
    weight = np.interp(level, centers, np.clip(sigmas / s_max, min_weight, 1.0)).astype(np.float32)
    out = image.astype(np.float32) + weight[..., None] * (denoised.astype(np.float32) - image.astype(np.float32))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


CHROMA_RATIO = 1.8


def nl_means(
    image: np.ndarray,
    strength: float = 7.0,
    color_strength: float | None = None,
    template_window: int = 7,
    search_window: int = 15,
) -> np.ndarray:
    """Цветное шумоподавление non-local means (``fastNlMeansDenoisingColored``).

    OpenCV чистит отдельно яркость (L, сила ``strength``) и цвет (a/b,
    сила ``color_strength``). Цветной шум — разноцветные пятна в тенях —
    заметнее и неприятнее яркостного «зерна», поэтому по умолчанию цвет
    чистится в ``CHROMA_RATIO`` раз сильнее: пятна уходят, а текстура
    не превращается в «пластик». Окно поиска 15 вместо стандартного 21
    заметно быстрее при почти том же качестве.
    """
    ensure_bgr_u8(image)
    h_color = strength * CHROMA_RATIO if color_strength is None else color_strength
    return cv2.fastNlMeansDenoisingColored(
        image, None, float(strength), float(h_color), template_window, search_window
    )
