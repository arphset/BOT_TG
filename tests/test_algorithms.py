from __future__ import annotations

import cv2
import numpy as np
import pytest

from bot.algorithms import (
    align_frames,
    average_frames,
    clahe_lab,
    estimate_noise,
    merge_mertens,
    nl_means,
    richardson_lucy,
    unsharp_mask,
)
from bot.algorithms.align import AlignmentError
from tests.conftest import add_noise, make_scene


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def laplacian_energy(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


SINGLE = [
    nl_means,
    clahe_lab,
    unsharp_mask,
    richardson_lucy,
]


@pytest.mark.parametrize("func", SINGLE, ids=lambda f: f.__name__)
def test_shape_dtype_and_purity(func, scene: np.ndarray) -> None:
    original = scene.copy()
    out = func(scene)
    assert out.shape == scene.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(scene, original)  # вход не изменён


@pytest.mark.parametrize("func", SINGLE, ids=lambda f: f.__name__)
def test_rejects_wrong_input(func) -> None:
    with pytest.raises((ValueError, TypeError)):
        func(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises((ValueError, TypeError)):
        func(np.zeros((10, 10, 3), dtype=np.float32))


def test_estimate_noise_tracks_sigma(scene: np.ndarray) -> None:
    clean = estimate_noise(scene)
    noisy = estimate_noise(add_noise(scene, 15))
    assert noisy > clean + 8
    assert 10 < noisy < 22


def test_nl_means_reduces_noise(scene: np.ndarray) -> None:
    noisy = add_noise(scene, 15)
    denoised = nl_means(noisy, strength=10)
    assert rmse(denoised, scene) < 0.7 * rmse(noisy, scene)
    assert estimate_noise(denoised) < 0.5 * estimate_noise(noisy)


def test_clahe_raises_local_contrast() -> None:
    flat = make_scene()
    low = cv2.convertScaleAbs(flat, alpha=0.3, beta=90)  # «плоское» изображение
    out = clahe_lab(low, clip_limit=3.0)
    gray_in = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert gray_out.std() > gray_in.std() * 1.15
    assert laplacian_energy(out) > 1.5 * laplacian_energy(low)


def test_unsharp_increases_edge_energy(scene: np.ndarray) -> None:
    soft = cv2.GaussianBlur(scene, (0, 0), 1.5)
    assert laplacian_energy(unsharp_mask(soft, amount=1.0)) > 1.5 * laplacian_energy(soft)


def test_unsharp_zero_amount_is_identity(scene: np.ndarray) -> None:
    out = unsharp_mask(scene, amount=0.0)
    assert np.abs(out.astype(int) - scene.astype(int)).max() <= 1  # только округление float→uint8


def test_richardson_lucy_restores_blur(scene: np.ndarray) -> None:
    blurred = cv2.GaussianBlur(scene, (0, 0), 1.5)
    restored = richardson_lucy(blurred, psf_sigma=1.5, iterations=20)
    assert rmse(restored, scene) < rmse(blurred, scene)
    assert laplacian_energy(restored) > laplacian_energy(blurred)


def test_richardson_lucy_zero_iterations(scene: np.ndarray) -> None:
    np.testing.assert_array_equal(richardson_lucy(scene, iterations=0), scene)


def _shifted(img: np.ndarray, dx: float, dy: float, angle: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_align_compensates_shift(scene: np.ndarray) -> None:
    moved = _shifted(scene, 6, -4, angle=1.0)
    aligned = align_frames([scene, moved])
    assert len(aligned) == 2
    assert aligned[1].shape == scene.shape and aligned[1].dtype == np.uint8
    crop = (slice(20, -20), slice(20, -20))
    assert rmse(aligned[1][crop], scene[crop]) < 0.4 * rmse(moved[crop], scene[crop])


def test_align_downscaled_search(scene: np.ndarray) -> None:
    big = cv2.resize(scene, (960, 720))
    moved = _shifted(big, 12, 8)
    aligned = align_frames([big, moved], max_side=320)
    crop = (slice(40, -40), slice(40, -40))
    assert rmse(aligned[1][crop], big[crop]) < 0.4 * rmse(moved[crop], big[crop])


def test_align_fails_on_unrelated_frames() -> None:
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    with pytest.raises(AlignmentError):
        align_frames([flat, flat.copy()])


def test_align_rejects_size_mismatch(scene: np.ndarray) -> None:
    with pytest.raises(ValueError):
        align_frames([scene, scene[:100]])


def test_average_reduces_noise(scene: np.ndarray) -> None:
    frames = [add_noise(scene, 15, seed=i) for i in range(8)]
    avg = average_frames(frames)
    assert avg.shape == scene.shape and avg.dtype == np.uint8
    assert rmse(avg, scene) < 0.5 * rmse(frames[0], scene)


def test_mertens_fuses_exposures(scene: np.ndarray) -> None:
    dark = cv2.convertScaleAbs(scene, alpha=0.4)
    bright = cv2.convertScaleAbs(scene, alpha=1.0, beta=90)
    fused = merge_mertens([dark, scene, bright])
    assert fused.shape == scene.shape and fused.dtype == np.uint8
    gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    dark_mean = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY).mean()
    bright_mean = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY).mean()
    assert dark_mean < gray.mean() < bright_mean
