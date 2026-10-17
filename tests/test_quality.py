"""Тесты улучшений качества: цветокоррекция, шум цвета/яркости, метаданные, перенос деталей."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from bot.algorithms import auto_levels, color_correct, nl_means, vibrance, white_balance
from bot.algorithms.color import chroma_cast
from bot.algorithms.superres import fuse_detail, output_is_x4, realesrgan_fit
from bot.media import ImageMeta, encode_image, read_meta, to_srgb
from tests.conftest import add_noise, make_scene


def tinted(img: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(img.astype(np.float32) * np.array(gains, dtype=np.float32), 0, 255).astype(np.uint8)


# ---------- цветокоррекция ----------


@pytest.mark.parametrize("func", [white_balance, auto_levels, vibrance, color_correct], ids=lambda f: f.__name__)
def test_color_shape_dtype_purity(func, scene: np.ndarray) -> None:
    original = scene.copy()
    out = func(scene)
    assert out.shape == scene.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(scene, original)


def test_white_balance_reduces_cast(scene: np.ndarray) -> None:
    yellow = tinted(scene, (0.75, 1.0, 1.1))  # BGR: меньше синего, больше красного
    fixed = white_balance(yellow)
    assert chroma_cast(fixed) < 0.75 * chroma_cast(yellow)


def test_white_balance_is_partial_and_capped(scene: np.ndarray) -> None:
    # Сильно тёплый «закат»: исправляется частично, тёплый оттенок остаётся.
    warm = tinted(scene, (0.5, 0.85, 1.2))
    fixed = white_balance(warm)
    b, g, r = (fixed[..., c].mean() for c in range(3))
    assert r > b  # не выстужено до нейтрального
    gains = fixed.astype(np.float32).mean(axis=(0, 1)) / np.maximum(warm.astype(np.float32).mean(axis=(0, 1)), 1)
    assert gains.max() < 1.3 and gains.min() > 0.75


def test_white_balance_leaves_neutral_image(scene: np.ndarray) -> None:
    gray = np.repeat(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)[..., None], 3, axis=2)
    np.testing.assert_array_equal(white_balance(gray), gray)


def test_auto_levels_stretches_flat_image(scene: np.ndarray) -> None:
    flat = cv2.convertScaleAbs(scene, alpha=0.45, beta=70)
    out = auto_levels(flat)
    lum = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)  # noqa: E731
    assert lum(out).std() > 1.3 * lum(flat).std()
    assert abs(lum(out).mean() - lum(flat).mean()) < 12  # общая яркость почти та же


def test_auto_levels_keeps_full_range_image() -> None:
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (32, 1))
    img = np.repeat(ramp[..., None], 3, axis=2)
    assert np.abs(auto_levels(img).astype(int) - img.astype(int)).max() <= 1


def test_vibrance_boosts_pale_more_than_saturated() -> None:
    pale = np.full((16, 16, 3), (150, 140, 120), np.uint8)  # бледно-голубой (BGR)
    vivid = np.full((16, 16, 3), (230, 40, 20), np.uint8)  # насыщенно-синий
    sat = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean()  # noqa: E731
    pale_gain = sat(vibrance(pale, 0.4)) / sat(pale)
    vivid_gain = sat(vibrance(vivid, 0.4)) / max(sat(vivid), 1)
    assert pale_gain > 1.1
    assert pale_gain > vivid_gain


def test_vibrance_protects_skin() -> None:
    skin = np.full((16, 16, 3), (120, 150, 205), np.uint8)  # типичный тон кожи (BGR)
    sky = np.full((16, 16, 3), (205, 160, 130), np.uint8)
    diff = lambda im: np.abs(vibrance(im, 0.4).astype(int) - im.astype(int)).mean()  # noqa: E731
    assert diff(skin) < 0.5 * diff(sky)


# ---------- шум яркости и цвета ----------


def chroma_noise(img: np.ndarray, clean: np.ndarray) -> float:
    lab_i = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    lab_c = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(float)
    return float(np.sqrt(((lab_i[..., 1:] - lab_c[..., 1:]) ** 2).mean()))


def test_chroma_denoised_stronger_by_default(scene: np.ndarray) -> None:
    noisy = add_noise(scene, 15)
    equal = nl_means(noisy, strength=8, color_strength=8)
    default = nl_means(noisy, strength=8)
    assert chroma_noise(default, scene) < chroma_noise(equal, scene)


# ---------- метаданные ----------


def p3_profile() -> bytes:
    # Профиль, отличный от sRGB, чтобы преобразование было заметно.
    prof = ImageCms.createProfile("LAB")
    return ImageCms.ImageCmsProfile(prof).tobytes()


def test_icc_and_exif_round_trip(scene: np.ndarray) -> None:
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    exif[0x0112] = 6  # Orientation: повёрнуто
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", icc_profile=srgb, exif=exif.tobytes())

    meta = read_meta(buf.getvalue())
    assert meta.icc_profile == srgb
    out = encode_image(scene, "jpg", meta=meta)
    with Image.open(io.BytesIO(out)) as im:
        assert im.info.get("icc_profile") == srgb
        assert im.getexif()[0x010F] == "TestCam"
        assert im.getexif()[0x0112] == 1  # ориентация сброшена: пиксели уже повёрнуты


def test_jpeg_is_444(scene: np.ndarray) -> None:
    with Image.open(io.BytesIO(encode_image(scene, "jpg"))) as im:
        assert im.layer[0][1:3] == im.layer[1][1:3] == (1, 1)  # без субсэмплинга цвета


def test_png_keeps_icc(scene: np.ndarray) -> None:
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    out = encode_image(scene, "png", meta=ImageMeta(icc_profile=srgb))
    with Image.open(io.BytesIO(out)) as im:
        assert im.info.get("icc_profile") == srgb


def test_read_meta_tolerates_garbage() -> None:
    assert read_meta(b"not an image") == ImageMeta()


def test_to_srgb_without_profile_is_identity(scene: np.ndarray) -> None:
    assert to_srgb(scene, ImageMeta()) is scene


def test_to_srgb_bad_profile_falls_back(scene: np.ndarray) -> None:
    np.testing.assert_array_equal(to_srgb(scene, ImageMeta(icc_profile=b"junk")), scene)


# ---------- нейросеть в исходном размере ----------


def fine_texture(h: int = 160, w: int = 200) -> np.ndarray:
    """Сцена с мелкими деталями, которые теряются при уменьшении в 2+ раза."""
    img = make_scene(h, w).astype(np.float32)
    y, x = np.mgrid[0:h, 0:w]
    stripes = 40 * np.sign(np.sin(x / 1.3) * np.sin(y / 1.7))
    img += stripes[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_output_mode_rules() -> None:
    assert output_is_x4((500, 800, 3), "auto", 1024)
    assert not output_is_x4((1500, 2000, 3), "auto", 1024)
    assert output_is_x4((1500, 2000, 3), "x4", 1024)
    assert not output_is_x4((500, 800, 3), "original", 1024)


def test_original_size_output(scene: np.ndarray) -> None:
    img = fine_texture()
    out = realesrgan_fit(img, max_side=64)  # кадр больше max_side → исходный размер
    assert out.shape == img.shape and out.dtype == np.uint8


def test_small_image_still_x4() -> None:
    img = make_scene(40, 50)
    assert realesrgan_fit(img, max_side=64).shape == (160, 200, 3)


def test_fusion_restores_real_detail() -> None:
    clean = fine_texture()
    noisy = add_noise(clean, 6)
    # «Нейросеть», видевшая кадр уменьшенным: гладкая, без мелких деталей.
    small = cv2.resize(noisy, (50, 40), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (200, 160), interpolation=cv2.INTER_CUBIC)
    fused = fuse_detail(noisy, restored, noise_sigma=6.0)
    rmse = lambda a: float(np.sqrt(((a.astype(float) - clean.astype(float)) ** 2).mean()))  # noqa: E731
    assert rmse(fused) < 0.8 * rmse(restored)


def test_fusion_does_not_bring_back_noise(scene: np.ndarray) -> None:
    noisy = add_noise(scene, 12)
    restored = cv2.GaussianBlur(scene, (0, 0), 0.8)  # почти чистая основа
    fused = fuse_detail(noisy, restored, noise_sigma=12.0)
    lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
    assert lap(fused) < 0.5 * lap(noisy)


def test_fusion_rejects_size_mismatch(scene: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fuse_detail(scene, scene[:100], 5.0)
