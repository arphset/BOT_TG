"""Поиск лиц классическими каскадами Хаара (без нейросетей) и маска лица.

Лица — то, на что смотрят первым делом. Маска нужна, чтобы обработать лицо
отдельно: оценить и убрать именно его смаз (лицо часто смазано сильнее фона —
движение, промах фокуса) и сохранить больше его реальных деталей после
нейросети, чтобы кожа не становилась «пластиковой».
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8

logger = logging.getLogger(__name__)

Rect = tuple[int, int, int, int]  # x, y, w, h


@lru_cache(maxsize=4)
def _cascade(name: str) -> cv2.CascadeClassifier | None:
    """Каскад из пакета OpenCV. Читается через память: на Windows OpenCV не открывает
    пути с не-ASCII символами, а так работает в любой папке."""
    path = Path(cv2.data.haarcascades) / name
    try:
        xml = path.read_text(encoding="utf-8")
        storage = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        cascade = cv2.CascadeClassifier()
        if not cascade.read(storage.getFirstTopLevelNode()):
            raise RuntimeError("каскад не прочитан")
        return cascade
    except Exception:  # noqa: BLE001 — без детектора лиц обработка всё равно идёт
        logger.warning("каскад %s недоступен, лица не ищутся", name)
        return None


def _skin_fraction(bgr: np.ndarray) -> float:
    """Доля пикселей с тоном кожи (классический диапазон в YCrCb)."""
    if bgr.size == 0:
        return 0.0
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return float(((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 130)).mean())


def _feature_fraction(bgr: np.ndarray) -> float:
    """Доля тёмных деталей (глаза, брови, оправы) в верхней половине рамки лица.
    У ровного пятна телесного цвета их нет — так отсекаются ложные «лица»."""
    if bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = g.shape[0]
    upper = g[h // 6 : h * 3 // 5]
    return float((upper < np.median(g) - 30).mean())


def _overlaps(a: Rect, b: Rect) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy > 0.3 * min(aw * ah, bw * bh)


def detect_faces(image: np.ndarray, max_side: int = 1024, min_skin: float = 0.35) -> list[Rect]:
    """Фронтальные лица (в координатах исходника).

    Каскад берётся чувствительный (находит и лица, слегка опущенные вниз, и в очках),
    а ложные срабатывания отсекаются двумя проверками: в рамке настоящего лица
    кожи больше ``min_skin`` (брусчатка, ткань, узоры отпадают), и в верхней
    половине есть тёмные детали — глаза, брови (ровные пятна телесного цвета отпадают).
    """
    ensure_bgr_u8(image)
    cascade = _cascade("haarcascade_frontalface_default.xml")
    if cascade is None:
        return []
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_size = max(24, int(min(gray.shape) * 0.04))
    found = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_size, min_size))
    faces: list[Rect] = []
    # Крупные кандидаты первыми: из пересекающихся остаётся самый крупный.
    for x, y, fw, fh in sorted((tuple(int(v) for v in r) for r in found), key=lambda r: -r[2]):
        crop = small[y : y + fh, x : x + fw]
        if _skin_fraction(crop) < min_skin or _feature_fraction(crop) < 0.04:
            continue
        rect = (x, y, fw, fh)
        if any(_overlaps(rect, f) for f in faces):
            continue
        faces.append(rect)
    return [tuple(int(round(v / scale)) for v in r) for r in faces]  # type: ignore[misc]


def _skin_map(bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return ((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 130)).astype(np.uint8)


def face_mask(image: np.ndarray, faces: list[Rect] | None = None) -> np.ndarray:
    """Маска кожи лиц 0..1 (float32, размер кадра). Пустая, если лиц нет.

    Каскад часто находит не всё лицо, а его часть, поэтому маска строится так:
    вокруг найденной рамки (с запасом) берётся связная область тона кожи,
    касающаяся рамки, — она повторяет реальную форму лица и шеи. Светлые волосы
    по цвету близки к коже и могут попасть в маску; для её назначения
    (восстановление резкости лица) это не вредно. Края маски растушёваны.
    """
    ensure_bgr_u8(image)
    h, w = image.shape[:2]
    faces = detect_faces(image) if faces is None else faces
    mask = np.zeros((h, w), np.float32)
    if not faces:
        return mask
    for x, y, fw, fh in faces:
        cx, cy = x + fw / 2, y + fh / 2
        half = 2.4 * max(fw, fh)  # каскад часто находит часть лица со смещением
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1 = int(max(0, cy - half * 0.8)), int(min(h, cy + half * 0.9))
        skin = _skin_map(image[y0:y1, x0:x1])
        k = max(3, fw // 15) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
        n, labels = cv2.connectedComponents(skin)
        if n <= 1:
            continue
        # Компоненты кожи, пересекающиеся с найденной рамкой.
        box = labels[y - y0 : y - y0 + fh, x - x0 : x - x0 + fw]
        ids = [i for i in np.unique(box) if i != 0]
        region = np.isin(labels, ids).astype(np.float32)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], region)
    feather = max(2.0, 0.06 * float(np.median([f[2] for f in faces])))
    return np.clip(cv2.GaussianBlur(mask, (0, 0), feather), 0.0, 1.0)
