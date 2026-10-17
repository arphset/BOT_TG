"""Выравнивание кадров серии методом ECC (евклидова модель)."""

import logging
from collections.abc import Sequence

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8

logger = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Кадр не удалось выровнять с опорным."""


def _estimate_warp(ref_gray: np.ndarray, gray: np.ndarray, iterations: int, eps: float) -> np.ndarray:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    _, warp = cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    return warp


def align_frames(
    frames: Sequence[np.ndarray],
    reference_index: int = 0,
    max_side: int = 1024,
    iterations: int = 100,
    eps: float = 1e-5,
) -> list[np.ndarray]:
    """Выравнивает все кадры по опорному.

    Преобразование ищется на уменьшенной копии (до ``max_side`` пикселей
    по длинной стороне), затем сдвиг масштабируется и применяется к
    полноразмерному кадру. Если ECC не сошёлся, бросается ``AlignmentError``.
    """
    if not frames:
        raise ValueError("пустая серия кадров")
    for frame in frames:
        ensure_bgr_u8(frame)
    ref = frames[reference_index]
    h, w = ref.shape[:2]
    if any(f.shape != ref.shape for f in frames):
        raise ValueError("все кадры серии должны быть одного размера")

    scale = min(1.0, max_side / max(h, w))
    small_size = (max(1, int(w * scale)), max(1, int(h * scale)))

    def small_gray(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0

    ref_gray = small_gray(ref)
    aligned: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        if idx == reference_index:
            aligned.append(frame.copy())
            continue
        try:
            warp = _estimate_warp(ref_gray, small_gray(frame), iterations, eps)
        except cv2.error as exc:
            raise AlignmentError(f"кадр {idx + 1} не выровнялся (ECC не сошёлся)") from exc
        warp[:, 2] /= scale
        aligned.append(
            cv2.warpAffine(
                frame,
                warp,
                (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
        logger.debug("кадр %d выровнен, сдвиг %.1f/%.1f px", idx, warp[0, 2], warp[1, 2])
    return aligned
