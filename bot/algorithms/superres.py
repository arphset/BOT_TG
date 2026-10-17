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

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab
from bot.algorithms.denoise import estimate_noise

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


def fuse_detail(
    original: np.ndarray,
    restored: np.ndarray,
    noise_sigma: float,
    detail_sigma: float = 1.2,
    threshold_k: float = 2.5,
    amount: float = 0.8,
) -> np.ndarray:
    """Переносит реальные мелкие детали оригинала в восстановленное нейросетью изображение.

    Оба кадра одного размера. Работа идёт по яркости (L в Lab); цвет берётся
    у нейросети (он чище). Высокие частоты оригинала пропускаются через
    мягкий порог по уровню шума: всё, что слабее ~``threshold_k``·σ, считается
    шумом и отбрасывается, сильное (края, текстура) сохраняется. В каждой
    точке берётся та деталь, что сильнее: оригинала или нейросети.
    """
    ensure_bgr_u8(original)
    ensure_bgr_u8(restored)
    if original.shape != restored.shape:
        raise ValueError("кадры для переноса деталей должны быть одного размера")
    lab_o = to_lab(original)
    lab_r = to_lab(restored)
    l_o, l_r = lab_o[:, :, 0], lab_r[:, :, 0]
    hp_o = l_o - cv2.GaussianBlur(l_o, (0, 0), detail_sigma)
    base_r = cv2.GaussianBlur(l_r, (0, 0), detail_sigma)
    hp_r = l_r - base_r

    t = max(threshold_k * noise_sigma * 100.0 / 255.0, 0.5)  # σ в единицах серого → единицы L
    weight = np.clip((np.abs(hp_o) - t) / t, 0.0, 1.0)  # 0 ниже порога, 1 выше 2t
    real = hp_o * weight * amount
    detail = np.where(np.abs(real) > np.abs(hp_r), real, hp_r)
    lab_r[:, :, 0] = np.clip(base_r + detail, 0.0, 100.0)
    return from_lab(lab_r)


OutputMode = Literal["auto", "x4", "original"]


def realesrgan_fit(
    image: np.ndarray,
    variant: Variant = "general",
    output: OutputMode = "auto",
    max_side: int = 1024,
    tile: int = 384,
    overlap: int = 16,
    time_budget: float | None = None,
    threads: int = 1,
) -> np.ndarray:
    """Real-ESRGAN с выбором размера результата.

    * ``x4``       — увеличение в 4 раза (от уменьшенного до ``max_side`` кадра);
    * ``original`` — результат в размере исходника, с переносом реальных деталей;
    * ``auto``     — ×4 для маленьких кадров (длинная сторона ≤ ``max_side``),
      исходный размер для больших: так большое фото не «дорисовывается».
    """
    up = realesrgan(
        image, variant=variant, max_side=max_side, tile=tile, overlap=overlap,
        time_budget=time_budget, threads=threads,
    )
    keep = output == "original" or (output == "auto" and max(image.shape[:2]) > max_side)
    if not keep:
        return up
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if up.shape[0] >= h else cv2.INTER_CUBIC
    restored = cv2.resize(up, (w, h), interpolation=interp)
    return fuse_detail(image, restored, estimate_noise(image))


def output_is_x4(shape: tuple[int, ...], output: str, max_side: int) -> bool:
    """Будет ли результат увеличен ×4 (для подписи шага)."""
    return output == "x4" or (output == "auto" and max(shape[:2]) <= max_side)
