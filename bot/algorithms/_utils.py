"""Общие проверки и преобразования для алгоритмов."""

import cv2
import numpy as np


def ensure_bgr_u8(image: np.ndarray) -> None:
    """Проверяет, что изображение — трёхканальный BGR uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError("ожидается np.ndarray")
    if image.dtype != np.uint8:
        raise ValueError(f"ожидается uint8, получено {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"ожидается изображение HxWx3, получено {image.shape}")


def to_lab(image: np.ndarray) -> np.ndarray:
    """BGR uint8 → Lab float32 (L в 0..100).

    Работа во float, а не в 8-битном Lab: 8-битный круг BGR→Lab→BGR
    сам по себе портит пиксели на несколько уровней.
    """
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def from_lab(lab: np.ndarray) -> np.ndarray:
    """Lab float32 → BGR uint8."""
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)
