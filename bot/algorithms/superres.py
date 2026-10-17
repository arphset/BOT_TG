"""Нейросетевое улучшение Real-ESRGAN (realesr-general-x4v3, SRVGG-компакт) через onnxruntime.

Модель увеличивает кадр в 4 раза. Перед инференсом вход уменьшается до
``max_side`` по длинной стороне, затем обрабатывается тайлами с перекрытием:
так пик памяти не зависит от размера кадра.

Варианты (по официальному inference_realesrgan.py, параметр ``-dn``):
* ``general`` — realesr-general-x4v3, сильное шумоподавление (``-dn 1``); подходит для шумных кадров;
* ``wdn``     — realesr-general-wdn-x4v3, слабое шумоподавление (``-dn 0``): меньше «пластика»,
  сохраняет мелкую текстуру, но оставляет шум.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8

logger = logging.getLogger(__name__)

SCALE = 4
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
VARIANTS: dict[str, str] = {
    "general": "realesr-general-x4v3.onnx",
    "wdn": "realesr-general-wdn-x4v3.onnx",
}
Variant = Literal["general", "wdn"]


class SuperResUnavailable(RuntimeError):
    """Модель или onnxruntime недоступны."""


class SuperResTimeout(RuntimeError):
    """Инференс не укладывается в отведённое время."""


@lru_cache(maxsize=2)
def _session(path: str, threads: int) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SuperResUnavailable("onnxruntime не установлен") from exc
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def model_path(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"неизвестный вариант модели: {variant}")
    return MODELS_DIR / VARIANTS[variant]


def downscale_to(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image
    s = max_side / max(h, w)
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _tiles(length: int, tile: int) -> list[tuple[int, int]]:
    return [(start, min(start + tile, length)) for start in range(0, length, tile)]


def realesrgan(
    image: np.ndarray,
    variant: Variant = "general",
    max_side: int = 1024,
    tile: int = 256,
    overlap: int = 16,
    time_budget: float | None = None,
    threads: int = 1,
) -> np.ndarray:
    """Real-ESRGAN ×4 с тайловым инференсом.

    ``time_budget`` — лимит времени в секундах. После каждого тайла
    оценивается полное время; если прогноз больше лимита, бросается
    ``SuperResTimeout``, не дожидаясь, пока лимит истечёт.
    """
    ensure_bgr_u8(image)
    path = model_path(variant)
    if not path.exists():
        raise SuperResUnavailable(f"нет файла модели {path.name}")
    session = _session(str(path), threads)
    input_name = session.get_inputs()[0].name

    small = downscale_to(image, max_side)
    h, w = small.shape[:2]
    x = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = np.empty((h * SCALE, w * SCALE, 3), dtype=np.uint8)

    tile = max(16, tile)
    grid = [(ys, xs) for ys in _tiles(h, tile) for xs in _tiles(w, tile)]
    started = time.perf_counter()
    for done, ((y0, y1), (x0, x1)) in enumerate(grid, start=1):
        py0, py1 = max(0, y0 - overlap), min(h, y1 + overlap)
        px0, px1 = max(0, x0 - overlap), min(w, x1 + overlap)
        result = session.run(None, {input_name: np.ascontiguousarray(x[:, :, py0:py1, px0:px1])})[0][0]
        oy, ox = (y0 - py0) * SCALE, (x0 - px0) * SCALE
        crop = result[:, oy : oy + (y1 - y0) * SCALE, ox : ox + (x1 - x0) * SCALE]
        out[y0 * SCALE : y1 * SCALE, x0 * SCALE : x1 * SCALE] = np.clip(
            crop.transpose(1, 2, 0) * 255.0 + 0.5, 0, 255
        ).astype(np.uint8)

        if time_budget is not None:
            elapsed = time.perf_counter() - started
            projected = elapsed / done * len(grid)
            if projected > time_budget:
                raise SuperResTimeout(
                    f"модель не укладывается в {time_budget:.0f} с (прогноз {projected:.0f} с)"
                )
    logger.info(
        "Real-ESRGAN %s: %dx%d -> %dx%d, тайлов %d, %.1f с",
        variant, w, h, w * SCALE, h * SCALE, len(grid), time.perf_counter() - started,
    )
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
