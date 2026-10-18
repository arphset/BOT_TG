"""Удаление дымки методом тёмного канала (dark channel prior, He et al.).

В чистых участках природных фото почти всегда есть пиксель с одним очень
тёмным каналом. Дымка (туман, смог, грязное стекло, засветка) поднимает
«тёмный канал» — по нему оценивается толщина дымки и восстанавливается
картинка. Метод известен тем, что «перетемняет» большие белые объекты
(стены, одежда, небо), поэтому здесь он осторожный: пропускание не ниже
``t_min``, сила частичная, а детектор требует дымку по всему кадру, а не
только на белых участках.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bot.algorithms._guided import guided_filter
from bot.algorithms._utils import ensure_bgr_u8


def _dark_channel(img: np.ndarray, patch: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(img.min(axis=2), kernel)


def _patch_size(shape: tuple[int, ...]) -> int:
    return int(np.clip(min(shape[:2]) // 60, 7, 31)) | 1


@dataclass(frozen=True)
class HazeEstimate:
    score: float  # медиана тёмного канала, нормированная на атмосферный свет (0..1)
    airlight: tuple[float, float, float]  # B, G, R (0..1)
    contrast: float  # стандартное отклонение яркости (0..1)
    black_point: float  # 1-й перцентиль минимального канала, доля атмосферного света

    @property
    def is_hazy(self) -> bool:
        # Главный признак дымки — в кадре нет настоящего чёрного: даже самые тёмные места
        # «приподняты». Плюс тёмный канал высок по всему кадру и картинка плоская.
        # Белые стены и одежда дают высокий тёмный канал, но чёрная точка у них остаётся низкой.
        return self.black_point > 0.22 and self.score > 0.35 and self.contrast < 0.22

    @property
    def strength(self) -> float:
        """Сила коррекции: толщина дымки в самых тёмных местах (≈ 1 − t)."""
        return float(np.clip(self.black_point * 1.1, 0.0, 0.85))


def estimate_haze(image: np.ndarray, max_side: int = 800) -> HazeEstimate:
    ensure_bgr_u8(image)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    img = small.astype(np.float32) / 255.0
    dark = _dark_channel(img, _patch_size(img.shape))
    # Атмосферный свет — среди 0,1 % самых «дымных» пикселей берём самые яркие.
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    candidates = img.reshape(-1, 3)[idx]
    airlight = candidates[np.argsort(candidates.sum(axis=1))[-max(1, n // 10):]].mean(axis=0)
    airlight = np.maximum(airlight, 0.3)
    norm_dark = _dark_channel(img / airlight.reshape(1, 1, 3), _patch_size(img.shape))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    black_point = float(np.percentile(img.min(axis=2), 1)) / float(airlight.mean())
    return HazeEstimate(
        score=float(np.median(norm_dark)),
        airlight=tuple(float(v) for v in airlight),  # type: ignore[arg-type]
        contrast=float(gray.std()),
        black_point=black_point,
    )


def dehaze(
    image: np.ndarray,
    amount: float | None = None,
    t_min: float = 0.35,
    haze: HazeEstimate | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Убирает дымку. ``amount`` — доля убираемой дымки (ω в статье); по умолчанию
    берётся из оценки (толщина дымки в тёмных местах) и умножается на ``scale``.
    ``t_min`` — нижняя граница пропускания: ограничивает усиление контраста
    (не больше 1/t_min) и защищает белые объекты от перетемнения."""
    ensure_bgr_u8(image)
    haze = haze or estimate_haze(image)
    if amount is None:
        amount = haze.strength
    amount = float(np.clip(amount * scale, 0.0, 0.9))
    if amount <= 0.01:
        return image.copy()
    img = image.astype(np.float32) / 255.0
    airlight = np.array(haze.airlight, dtype=np.float32).reshape(1, 1, 3)
    patch = _patch_size(img.shape)
    transmission = 1.0 - amount * _dark_channel(img / airlight, patch)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    radius = max(8, patch * 3)
    transmission = guided_filter(gray, transmission.astype(np.float32), radius, 1e-3)
    transmission = np.clip(transmission, t_min, 1.0)[..., None]
    restored = (img - airlight) / transmission + airlight
    return np.clip(restored * 255.0 + 0.5, 0, 255).astype(np.uint8)
