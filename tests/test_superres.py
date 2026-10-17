"""Тесты нейросетевого шага на настоящих ONNX-моделях из models/ (на маленьких кадрах — быстро)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bot.algorithms import superres
from bot.algorithms.superres import SuperResTimeout, SuperResUnavailable, realesrgan
from tests.conftest import add_noise, make_scene


@pytest.fixture
def small() -> np.ndarray:
    return make_scene(48, 64)


@pytest.mark.parametrize("variant", ["general", "balanced", "wdn"])
def test_models_present_and_upscale_x4(variant: str, small: np.ndarray) -> None:
    assert superres.model_path(variant).exists()
    original = small.copy()
    out = realesrgan(small, variant=variant)
    assert out.shape == (48 * 4, 64 * 4, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(small, original)


def test_output_resembles_input(small: np.ndarray) -> None:
    out = realesrgan(small)
    back = cv2.resize(out, (64, 48), interpolation=cv2.INTER_AREA)
    assert np.abs(back.astype(int) - small.astype(int)).mean() < 12


def test_input_downscaled_to_max_side() -> None:
    img = make_scene(120, 200)
    out = realesrgan(img, max_side=100)
    assert out.shape == (60 * 4, 100 * 4, 3)


def test_tiles_match_whole_image() -> None:
    img = make_scene(96, 128)
    whole = realesrgan(img, tile=1024)
    tiled = realesrgan(img, tile=48, overlap=16)
    diff = np.abs(whole.astype(int) - tiled.astype(int))
    assert diff.mean() < 0.1
    assert diff.max() <= 8


def test_general_denoises_stronger_than_wdn() -> None:
    # В официальном Real-ESRGAN x4v3 — «сильное шумоподавление» (-dn 1),
    # wdn — «слабое, шум сохраняется» (-dn 0).
    noisy = add_noise(make_scene(48, 64), 15)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
    assert lap(realesrgan(noisy, variant="general")) < lap(realesrgan(noisy, variant="wdn"))


def test_budget_exceeded_raises(small: np.ndarray) -> None:
    with pytest.raises(SuperResTimeout):
        realesrgan(small, tile=16, time_budget=1e-6)


def test_missing_model_raises(small: np.ndarray, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(superres, "MODELS_DIR", tmp_path)
    with pytest.raises(SuperResUnavailable):
        realesrgan(small)


def test_unknown_variant(small: np.ndarray) -> None:
    with pytest.raises(ValueError):
        realesrgan(small, variant="nope")  # type: ignore[arg-type]


def test_balanced_is_between_general_and_wdn() -> None:
    noisy = add_noise(make_scene(48, 64), 15)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
    general, balanced, wdn = (lap(realesrgan(noisy, variant=v)) for v in ("general", "balanced", "wdn"))
    assert general < balanced < wdn
