"""Тесты третьего пакета: дымка, тени и света, лица, горизонт, умный режим с уровнями силы."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bot.algorithms.deblur import deblur_faces
from bot.algorithms.dehaze import dehaze, estimate_haze
from bot.algorithms.faces import detect_faces, face_mask
from bot.algorithms.horizon import detect_tilt, straighten
from bot.algorithms.superres import fuse_detail
from bot.algorithms.tone import estimate_tone, shadows_highlights
from bot.pipeline import STRENGTH_LEVELS, Mode, build_steps, run_pipeline
from tests.conftest import add_noise, make_scene


def gray(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a.astype(float) - b.astype(float)) ** 2).mean()))


def shapes_scene(seed: int = 1, h: int = 480, w: int = 640, n: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 60)), y + int(rng.integers(8, 60))), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 30)), c, -1)
    return img


def hazy(img: np.ndarray, t: float, airlight=(0.85, 0.85, 0.88)) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    a = np.array(airlight, np.float32)
    return np.clip((f * t + a * (1 - t)) * 255, 0, 255).astype(np.uint8)


# ---------- 1. дымка ----------


@pytest.mark.parametrize("t", [0.6, 0.4])
def test_dehaze_detects_and_restores(t: float) -> None:
    clean = cv2.resize(make_scene(seed=3), (640, 480), interpolation=cv2.INTER_CUBIC)
    h = hazy(clean, t)
    est = estimate_haze(h)
    assert est.is_hazy
    assert rmse(dehaze(h, haze=est), clean) < 0.6 * rmse(h, clean)


def test_no_haze_on_clean_and_white_scenes() -> None:
    clean = cv2.resize(make_scene(seed=3), (640, 480), interpolation=cv2.INTER_CUBIC)
    white = clean.copy()
    cv2.rectangle(white, (120, 60), (520, 420), (235, 238, 240), -1)
    room = np.full((480, 640, 3), (200, 215, 225), np.uint8)
    cv2.rectangle(room, (250, 150), (420, 480), (40, 60, 70), -1)
    for img in (clean, white, room):
        assert not estimate_haze(img).is_hazy


def test_dehaze_zero_amount_is_identity(scene: np.ndarray) -> None:
    np.testing.assert_array_equal(dehaze(scene, amount=0.0), scene)


# ---------- 2. тени и света ----------


def backlit() -> np.ndarray:
    scene = cv2.resize(make_scene(seed=4), (640, 480))
    img = np.full((480, 640, 3), 245, np.uint8)
    img[100:480, 180:460] = cv2.convertScaleAbs(scene[100:480, 180:460], alpha=0.25)
    return img


def test_backlit_subject_lifted_without_losing_detail() -> None:
    img = backlit()
    est = estimate_tone(img)
    assert est.shadows > 0.3 and est.highlights > 0.3
    out = shadows_highlights(img)
    fig = (slice(150, 470), slice(200, 440))
    hp = lambda im: (gray(im)[fig] - cv2.GaussianBlur(gray(im), (0, 0), 3)[fig]).std()  # noqa: E731
    assert gray(out)[fig].mean() > gray(img)[fig].mean() + 10
    assert hp(out) > 0.9 * hp(img)
    assert gray(out)[:80].mean() < gray(img)[:80].mean()  # выбитый фон приглушён


def test_tone_leaves_normal_exposure(scene: np.ndarray) -> None:
    assert estimate_tone(scene).shadows == 0.0
    np.testing.assert_array_equal(shadows_highlights(scene, 0.0, 0.0), scene)


# ---------- 3. лица ----------


def face_scene() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    rng = np.random.default_rng(1)
    img = shapes_scene()
    cv2.ellipse(img, (320, 220), (70, 95), 0, 0, 360, (120, 150, 205), -1)  # тон кожи
    for _ in range(12):
        cv2.circle(img, (int(rng.integers(270, 370)), int(rng.integers(150, 300))), int(rng.integers(3, 9)), (60, 80, 130), -1)
    sharp = img.copy()
    blurred = img.copy()
    reg = (slice(100, 340), slice(220, 420))
    blurred[reg] = cv2.GaussianBlur(img, (0, 0), 2.0)[reg]
    return sharp, blurred, [(250, 130, 140, 160)]


def test_face_mask_follows_skin() -> None:
    sharp, _, faces = face_scene()
    mask = face_mask(sharp, faces)
    assert mask[220, 320] > 0.9  # центр «лица»
    assert mask[20, 20] == 0.0  # фон
    assert face_mask(sharp, []).max() == 0.0


def test_no_faces_on_synthetic_scenes() -> None:
    assert detect_faces(cv2.resize(make_scene(), (1280, 960))) == []
    assert detect_faces(shapes_scene()) == []


def test_blurred_face_restored_only_inside() -> None:
    sharp, blurred, faces = face_scene()
    out = deblur_faces(blurred, faces)
    inner = (slice(150, 290), slice(270, 370))
    assert rmse(out[inner], sharp[inner]) < 0.85 * rmse(blurred[inner], sharp[inner])
    np.testing.assert_array_equal(out[420:], blurred[420:])


def test_sharp_face_left_alone() -> None:
    sharp, _, faces = face_scene()
    np.testing.assert_array_equal(deblur_faces(sharp, faces), sharp)


def test_fusion_keeps_more_face_detail() -> None:
    sharp, _, _ = face_scene()
    noisy = add_noise(sharp, 4)
    restored = cv2.GaussianBlur(sharp, (0, 0), 1.5)
    weight = np.zeros(sharp.shape[:2], np.float32)
    weight[130:310, 250:390] = 1.0
    plain = fuse_detail(noisy, restored, 4.0)
    face_aware = fuse_detail(noisy, restored, 4.0, face_weight=weight)
    inner = (slice(150, 290), slice(270, 370))
    assert rmse(face_aware[inner], sharp[inner]) < rmse(plain[inner], sharp[inner])
    np.testing.assert_array_equal(face_aware[400:], plain[400:])


# ---------- 4. горизонт ----------


def landscape() -> np.ndarray:
    img = np.zeros((600, 900, 3), np.uint8)
    img[:300] = (230, 190, 150)
    img[300:] = (60, 120, 80)
    for y in (380, 450, 520):
        cv2.line(img, (50, y), (850, y), (30, 30, 30), 4)
    return cv2.add(img, cv2.resize(make_scene(seed=2), (900, 600)) // 6)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("angle", [2.0, -3.5, 5.0])
def test_tilt_detected_and_fixed(angle: float) -> None:
    tilted = rotate(landscape(), angle)
    found = detect_tilt(tilted)
    assert found is not None and abs(found + angle) < 0.3
    fixed = straighten(tilted, found)
    assert detect_tilt(fixed) is None
    assert fixed.shape[0] < tilted.shape[0]  # с обрезкой краёв
    assert abs(fixed.shape[1] / fixed.shape[0] - 1.5) < 0.02  # пропорции сохранены


def test_level_or_strongly_tilted_left_alone() -> None:
    assert detect_tilt(landscape()) is None
    assert detect_tilt(rotate(landscape(), 10.0)) is None  # 10° — скорее задумано


def test_no_tilt_without_lines(scene: np.ndarray) -> None:
    diag = np.full((600, 900, 3), 200, np.uint8)
    for i in range(6):
        cv2.line(diag, (0, i * 100), (900, i * 100 + 400), (20, 20, 20), 3)
    assert detect_tilt(scene) is None
    assert detect_tilt(diag) is None


# ---------- умный режим ----------


def test_smart_enables_only_needed_steps() -> None:
    clean = shapes_scene()
    enabled = {s.name for s in build_steps(Mode.SMART, [clean]) if s.enabled}
    assert {"superres", "color", "contrast", "sharpen"} <= enabled
    assert not enabled & {"deblur", "face_deblur", "dehaze", "horizon"}

    names = lambda img: {s.name for s in build_steps(Mode.SMART, [img]) if s.enabled}  # noqa: E731
    assert "deblur" in names(cv2.GaussianBlur(clean, (0, 0), 1.8))
    assert "dehaze" in names(hazy(cv2.resize(make_scene(seed=3), (640, 480)), 0.5))
    assert "horizon" in names(rotate(landscape(), 3.0))
    assert "tone" in names(backlit())


def test_smart_order_sharpen_last_horizon_before() -> None:
    steps = [s.name for s in build_steps(Mode.SMART, [rotate(landscape(), 3.0)])]
    assert steps[-1] == "sharpen" and steps[-2] == "horizon"
    assert steps.index("deblur") < steps.index("superres") < steps.index("dehaze")


def test_strength_levels_scale_effect() -> None:
    img = cv2.GaussianBlur(add_noise(make_scene(96, 128), 6), (0, 0), 1.2)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
    soft, strong = (
        run_pipeline([img], build_steps(Mode.SMART, [img], strength=k)).image
        for k in (STRENGTH_LEVELS[0], STRENGTH_LEVELS[-1])
    )
    assert lap(strong) > 1.3 * lap(soft)  # «Сильнее» — резче


def test_smart_runs_end_to_end() -> None:
    img = add_noise(make_scene(), 6)
    result = run_pipeline([img], build_steps(Mode.SMART, [img]))
    assert result.errors == []
    assert result.image.dtype == np.uint8
    assert result.image.shape == (img.shape[0] * 4, img.shape[1] * 4, 3)  # маленькое → ×4
