"""Слияние серии: Mertens exposure fusion или усреднение кадров."""

from collections.abc import Sequence

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8


def _check(frames: Sequence[np.ndarray]) -> None:
    if not frames:
        raise ValueError("пустая серия кадров")
    for frame in frames:
        ensure_bgr_u8(frame)
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("все кадры серии должны быть одного размера")


def merge_mertens(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Слияние экспозиций Mertens (не требует знать выдержки)."""
    _check(frames)
    if len(frames) == 1:
        return frames[0].copy()
    fused = cv2.createMergeMertens().process(list(frames))
    return np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8)


def average_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Усреднение кадров одной выдержки: шум падает примерно в √N раз."""
    _check(frames)
    acc = np.zeros(frames[0].shape, dtype=np.float32)
    for frame in frames:
        acc += frame
    return np.clip(acc / len(frames) + 0.5, 0, 255).astype(np.uint8)


def exposure_spread(frames: Sequence[np.ndarray]) -> float:
    """Разброс средней яркости кадров (0..255); помогает выбрать Mertens или усреднение."""
    _check(frames)
    means = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) for f in frames]
    return max(means) - min(means)
