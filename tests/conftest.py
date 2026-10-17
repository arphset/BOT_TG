from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_scene(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Синтетическая «фотография»: градиент, геометрия, текстура."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = 60 + 120 * x / w
    img[..., 1] = 80 + 100 * y / h
    img[..., 2] = 120 + 60 * np.sin(x / 17.0) * np.cos(y / 23.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (w // 6, h // 5), (w // 2, h // 2), (230, 230, 230), -1)
    cv2.circle(img, (2 * w // 3, 2 * h // 3), min(h, w) // 6, (20, 40, 200), -1)
    cv2.line(img, (0, h - 20), (w, 20), (10, 10, 10), 3)
    texture = cv2.GaussianBlur(rng.normal(0, 12, (h, w, 3)).astype(np.float32), (0, 0), 1.5)
    return np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)


def add_noise(img: np.ndarray, sigma: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


@pytest.fixture
def scene() -> np.ndarray:
    return make_scene()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.config import get_config

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "42:TEST")
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
    for name in (
        "KV_REST_API_URL",
        "KV_REST_API_TOKEN",
        "ALLOWED_USER_IDS",
        "PROCESS_TIMEOUT",
        "SR_ENABLED",
        "SR_TIMEOUT",
        "SR_TILE",
        "SR_MAX_SIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    get_config.cache_clear()
    yield
    get_config.cache_clear()
