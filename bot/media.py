"""Декодирование, кодирование и подготовка превью.

Пиксели читает OpenCV, а метаданные (ICC-профиль и EXIF) — Pillow: OpenCV их
теряет. Без ICC-профиля фото, снятые в Display P3 (iPhone, многие Android),
после обработки выглядят тусклее и с другим оттенком, хотя пиксели не испорчены.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

EXIF_ORIENTATION = 0x0112


class ImageDecodeError(ValueError):
    """Файл не удалось прочитать как изображение."""


@dataclass(frozen=True)
class ImageMeta:
    """Метаданные, которые нужно перенести в результат."""

    icc_profile: bytes | None = None
    exif: bytes | None = None


def read_meta(data: bytes) -> ImageMeta:
    """ICC-профиль и EXIF исходника. Ориентация в EXIF сбрасывается в 1,
    потому что OpenCV при чтении уже повернул пиксели. Ошибки не критичны."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            icc = im.info.get("icc_profile") or None
            exif_obj = im.getexif()
            exif: bytes | None = None
            if exif_obj:
                exif_obj[EXIF_ORIENTATION] = 1
                exif = exif_obj.tobytes()
    except Exception:  # noqa: BLE001 — без метаданных обработка всё равно возможна
        logger.info("метаданные исходника не прочитаны")
        return ImageMeta()
    return ImageMeta(icc_profile=icc, exif=exif)


def to_srgb(image: np.ndarray, meta: ImageMeta) -> np.ndarray:
    """Переводит пиксели из профиля исходника в sRGB.

    Нужно для превью: Telegram пережимает фото и выбрасывает ICC-профиль,
    поэтому превью должно быть уже в sRGB. Файл-документ отдаётся
    с исходным профилем без преобразования.
    """
    if not meta.icc_profile:
        return image
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(meta.icc_profile))
        dst = ImageCms.createProfile("sRGB")
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        converted = ImageCms.profileToProfile(pil, src, dst, outputMode="RGB")
        return cv2.cvtColor(np.asarray(converted), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        logger.info("не удалось перевести превью в sRGB, отдаю как есть")
        return image


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


def encode_image(image: np.ndarray, fmt: str = "jpg", quality: int = 95, meta: ImageMeta | None = None) -> bytes:
    """Кодирует результат. ``fmt`` — ``jpg`` или ``png``.

    JPEG сохраняется без цветового субсэмплинга (4:4:4): тонкие цветные края
    не размываются. ICC-профиль и EXIF из ``meta`` переносятся в файл.
    """
    meta = meta or ImageMeta()
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    extra: dict[str, bytes] = {}
    if meta.icc_profile:
        extra["icc_profile"] = meta.icc_profile
    if meta.exif:
        extra["exif"] = meta.exif
    buf = io.BytesIO()
    try:
        if fmt == "png":
            pil.save(buf, "PNG", compress_level=6, **extra)
        else:
            pil.save(buf, "JPEG", quality=quality, subsampling=0, optimize=True, **extra)
    except Exception as exc:  # noqa: BLE001
        if not extra:
            raise RuntimeError("не удалось закодировать изображение") from exc
        logger.warning("метаданные не записались (%s), сохраняю без них", exc)
        return encode_image(image, fmt, quality, None)
    return buf.getvalue()


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
