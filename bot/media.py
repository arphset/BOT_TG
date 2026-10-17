"""Декодирование, кодирование и подготовка превью."""

from __future__ import annotations

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    """Файл не удалось прочитать как изображение."""


def decode_image(data: bytes) -> np.ndarray:
    """Байты файла → BGR uint8. Ориентация из EXIF учитывается OpenCV."""
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if image is None:
        raise ImageDecodeError("файл не похож на изображение в поддерживаемом формате")
    return image


def limit_megapixels(image: np.ndarray, max_megapixels: float) -> tuple[np.ndarray, bool]:
    """Уменьшает изображение, если оно больше ``max_megapixels``. Возвращает (кадр, уменьшен ли)."""
    h, w = image.shape[:2]
    if max_megapixels <= 0 or h * w <= max_megapixels * 1_000_000:
        return image, False
    scale = (max_megapixels * 1_000_000 / (h * w)) ** 0.5
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), True


def encode_image(image: np.ndarray, fmt: str = "jpg", quality: int = 95) -> bytes:
    """Кодирует результат. ``fmt`` — ``jpg`` или ``png``."""
    if fmt == "png":
        ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("не удалось закодировать изображение")
    return buf.tobytes()


def fit(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, text: str) -> None:
    scale = max(0.6, image.shape[1] / 900)
    thickness = max(1, int(scale * 2))
    org = (int(12 * scale), int(36 * scale))
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def make_comparison(before: np.ndarray, after: np.ndarray, max_side: int = 2560) -> np.ndarray:
    """Склейка «до / после» в одно изображение (рядом или друг под другом)."""
    if before.shape != after.shape:
        before = cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = after.shape[:2]
    horizontal = w <= h  # портретные кадры ставим рядом, альбомные — друг под другом
    half = max_side // 2
    left, right = fit(before, half).copy(), fit(after, half).copy()
    _label(left, "before")
    _label(right, "after")
    gap_shape = (left.shape[0], 6, 3) if horizontal else (6, left.shape[1], 3)
    gap = np.full(gap_shape, 255, dtype=np.uint8)
    return np.hstack([left, gap, right]) if horizontal else np.vstack([left, gap, right])
