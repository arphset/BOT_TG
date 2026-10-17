"""Тесты второго пакета качества: ореолы, адаптивный CLAHE, деблокинг, оценка смаза,
шум по зонам яркости, смешанная модель."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bot.algorithms.contrast import clahe_lab
from bot.algorithms.deblock import blockiness, deblock, is_blocky
from bot.algorithms.deblur import anisotropic_psf, estimate_blur, smart_deblur
from bot.algorithms.denoise import denoise_by_luma, estimate_noise_by_luma, nl_means
from bot.algorithms.sharpen import unsharp_mask
from bot.pipeline import Mode, build_steps
from tests.conftest import add_noise, make_scene


def gray(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)


def jpeg(im: np.ndarray, q: int) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def sharp_scene(seed: int = 1) -> np.ndarray:
    """Резкая сцена без интерполяции: прямоугольники и круги."""
    rng = np.random.default_rng(seed)
    img = np.full((240, 320, 3), 128, np.uint8)
    for _ in range(30):
        x, y = int(rng.integers(0, 320)), int(rng.integers(0, 240))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 80)), y + int(rng.integers(10, 80))), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 40)), c, -1)
    return img


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a.astype(float) - b.astype(float)) ** 2).mean()))


# ---------- 7. резкость без ореолов ----------


def test_halo_control_removes_overshoot() -> None:
    step = np.zeros((40, 80, 3), np.uint8)
    step[:, :40], step[:, 40:] = 60, 190
    soft = cv2.GaussianBlur(step, (0, 0), 1.2)
    plain = gray(unsharp_mask(soft, amount=1.0, halo_control=None))[20]
    clamped = gray(unsharp_mask(soft, amount=1.0))[20]
    assert plain.min() < 56 and plain.max() > 194  # без контроля — ореол
    assert clamped.min() >= 58 and clamped.max() <= 193
    np.testing.assert_allclose(clamped[39:42], plain[39:42], atol=2)  # склон края так же крутой


# ---------- 8. адаптивный CLAHE ----------


def test_adaptive_clahe_spares_flat_noisy_areas() -> None:
    sky = np.zeros((120, 160, 3), np.float32)
    sky[...] = (200, 170, 140)
    sky = add_noise(np.clip(sky + np.linspace(0, 20, 160)[None, :, None], 0, 255).astype(np.uint8), 4)
    tex = cv2.convertScaleAbs(make_scene(120, 160), alpha=0.5, beta=60)
    img = np.vstack([sky, tex])
    hp = lambda r: (r - cv2.GaussianBlur(r, (0, 0), 2)).std()  # noqa: E731
    plain, adaptive = clahe_lab(img, 2.0), clahe_lab(img, 2.0, adaptive=True)
    assert hp(gray(adaptive)[:120]) < 0.75 * hp(gray(plain)[:120])  # шум неба растёт меньше
    assert gray(adaptive)[120:].std() > 0.95 * gray(plain)[120:].std()  # контраст текстуры тот же


def test_adaptive_clahe_shape_dtype(scene: np.ndarray) -> None:
    out = clahe_lab(scene, adaptive=True)
    assert out.shape == scene.shape and out.dtype == np.uint8


# ---------- 9. деблокинг ----------


def test_blockiness_detects_jpeg_only() -> None:
    clean = cv2.resize(make_scene(seed=2), (640, 480), interpolation=cv2.INTER_CUBIC)
    assert not is_blocky(clean)
    assert not is_blocky(add_noise(clean, 8))
    assert not is_blocky(jpeg(clean, 95))
    assert is_blocky(jpeg(clean, 40))
    rescaled = cv2.resize(jpeg(clean, 20), (500, 375), interpolation=cv2.INTER_AREA)
    assert not is_blocky(rescaled)  # после масштабирования сетки нет


def test_blockiness_finds_grid_offset() -> None:
    clean = cv2.resize(make_scene(seed=2), (640, 480), interpolation=cv2.INTER_CUBIC)
    _, ox, oy = blockiness(jpeg(clean, 20)[3:, 5:])
    assert (ox, oy) == (3, 5)


@pytest.mark.parametrize("q", [40, 20, 10])
def test_deblock_reduces_blocks_and_error(q: int) -> None:
    clean = cv2.resize(make_scene(seed=2), (640, 480), interpolation=cv2.INTER_CUBIC)
    j = jpeg(clean, q)
    d = deblock(j)
    assert blockiness(d)[0] < 0.6 * blockiness(j)[0]
    assert rmse(d, clean) < rmse(j, clean)


def test_deblock_keeps_real_edges() -> None:
    e = np.zeros((64, 64, 3), np.uint8)
    e[:, 32:] = 220  # край ровно на границе блока
    j = jpeg(e, 30)
    np.testing.assert_array_equal(gray(deblock(j))[30, 29:36], gray(j)[30, 29:36])


# ---------- 10. оценка смаза ----------


def test_sharp_image_detected_as_sharp() -> None:
    assert estimate_blur(sharp_scene()).is_sharp


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.5])
def test_gaussian_blur_estimated(sigma: float) -> None:
    est = estimate_blur(cv2.GaussianBlur(sharp_scene(), (0, 0), sigma))
    assert abs(est.sigma_major - sigma) < 0.45
    assert not est.is_directional


@pytest.mark.parametrize("angle", [0, 90])
def test_motion_direction_estimated(angle: int) -> None:
    blurred = cv2.filter2D(sharp_scene(), -1, anisotropic_psf(3.0, 0.5, angle), borderType=cv2.BORDER_REFLECT)
    est = estimate_blur(blurred)
    assert est.is_directional
    assert abs((est.angle - angle + 90) % 180 - 90) <= 15
    assert 2.4 < est.sigma_major < 3.6


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_smart_deblur_restores(sigma: float) -> None:
    clean = sharp_scene()
    blurred = add_noise(cv2.GaussianBlur(clean, (0, 0), sigma), 2)
    assert rmse(smart_deblur(blurred), clean) < 0.9 * rmse(blurred, clean)


def test_smart_deblur_leaves_sharp_image() -> None:
    img = sharp_scene()
    np.testing.assert_array_equal(smart_deblur(img), img)


# ---------- 11. шум по зонам яркости ----------


def zone_noise_scene() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    h, w = 160, 480
    base = cv2.resize(make_scene(seed=5), (w, h)).astype(np.float32)
    y, x = np.mgrid[0:h, 0:w]
    texture = 14 * np.sin(x / 1.6) * np.sin(y / 2.1)
    ramp = np.linspace(0.15, 1.0, w)[None, :, None]
    clean = np.clip(base * ramp + texture[..., None] * ramp, 0, 255)
    sigma = 2.0 + 12.0 * (1 - clean.mean(axis=2, keepdims=True) / 255.0)
    noisy = np.clip(clean + rng.normal(0, 1, clean.shape) * sigma, 0, 255).astype(np.uint8)
    return clean.astype(np.uint8), noisy


def test_noise_estimated_higher_in_shadows() -> None:
    _, noisy = zone_noise_scene()
    _, sigmas = estimate_noise_by_luma(noisy)
    assert sigmas[0] > 1.4 * sigmas[-1]


def test_zone_denoise_keeps_highlight_texture() -> None:
    clean, noisy = zone_noise_scene()
    _, sigmas = estimate_noise_by_luma(noisy)
    uniform = nl_means(noisy, strength=float(np.clip(sigmas.max() * 1.05, 2.5, 10)))
    zoned = denoise_by_luma(noisy)
    bright, dark = (slice(None), slice(360, 480)), (slice(None), slice(0, 120))
    assert rmse(zoned[bright], clean[bright]) < rmse(uniform[bright], clean[bright])
    assert rmse(zoned[dark], clean[dark]) < 1.05 * rmse(uniform[dark], clean[dark])


# ---------- встраивание в конвейер ----------


def test_deblock_enabled_only_for_jpeg_blocks() -> None:
    clean = cv2.resize(make_scene(seed=2), (640, 480), interpolation=cv2.INTER_CUBIC)
    enabled = lambda img: {s.name: s.enabled for s in build_steps(Mode.AUTO, [img])}["deblock"]  # noqa: E731
    assert not enabled(clean)
    assert enabled(jpeg(clean, 30))


def test_deblur_title_reports_estimate() -> None:
    blurred = cv2.GaussianBlur(sharp_scene(), (0, 0), 2.0)
    title = next(s.title for s in build_steps(Mode.DEBLUR, [blurred]) if s.name == "deblur")
    assert "σ≈" in title
    sharp_title = next(s.title for s in build_steps(Mode.DEBLUR, [sharp_scene()]) if s.name == "deblur")
    assert "не найден" in sharp_title
