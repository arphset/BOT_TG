"""Деконволюция Ричардсона–Люси: оценка смаза по самому кадру и подавление «звона»."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8
from bot.algorithms.sharpen import clamp_to_local_range


def gaussian_psf(sigma: float) -> np.ndarray:
    """Нормированное гауссово ядро размытия."""
    size = max(3, int(np.ceil(sigma * 3)) * 2 + 1)
    k1d = cv2.getGaussianKernel(size, sigma, cv2.CV_32F)
    psf = k1d @ k1d.T
    return (psf / psf.sum()).astype(np.float32)


def anisotropic_psf(sigma_major: float, sigma_minor: float, angle_deg: float) -> np.ndarray:
    """Вытянутое гауссово ядро: σ вдоль направления смаза ``angle_deg`` и поперёк него."""
    sigma_major = max(sigma_major, 0.3)
    sigma_minor = max(min(sigma_minor, sigma_major), 0.3)
    radius = max(1, int(np.ceil(sigma_major * 3)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    t = np.deg2rad(angle_deg)
    u = x * np.cos(t) + y * np.sin(t)  # вдоль смаза
    v = -x * np.sin(t) + y * np.cos(t)  # поперёк
    psf = np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))
    return (psf / psf.sum()).astype(np.float32)


SHARP_FLOOR = 0.8  # оценка σ на идеально резком крае (погрешность метода)


@dataclass(frozen=True)
class BlurEstimate:
    sigma_major: float  # σ смаза вдоль направления, px
    sigma_minor: float  # σ поперёк, px
    angle: float  # направление смаза, градусы (0 — горизонтально)

    @property
    def is_sharp(self) -> bool:
        return self.sigma_major < 0.6

    @property
    def is_directional(self) -> bool:
        return self.sigma_major > 1.4 * max(self.sigma_minor, 0.3)


def estimate_blur(image: np.ndarray, reblur_sigma: float = 1.0, max_side: int = 1600) -> BlurEstimate:
    """Оценка размытия методом повторного размытия (re-blur).

    На краю, размытом гауссом σ, отношение градиентов исходника и его копии,
    дополнительно размытой σ0, равно R = √(σ² + σ0²) / σ, откуда σ = σ0 / √(R² − 1).
    Отношение считается по сильным краям отдельно для направлений 0–165°
    (край учитывается в том направлении, куда смотрит его градиент):
    у смаза от движения σ заметно больше вдоль направления движения.
    """
    ensure_bgr_u8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:  # оценка на уменьшенной копии; σ пересчитывается обратно
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    rb = cv2.GaussianBlur(g, (0, 0), reblur_sigma)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    rx, ry = cv2.Sobel(rb, cv2.CV_32F, 1, 0), cv2.Sobel(rb, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    edges = cv2.Canny(gray, 50, 150) > 0
    edges &= (gray > 8) & (gray < 247)  # без пересветов и провалов
    if edges.sum() < 50:
        return BlurEstimate(0.0, 0.0, 0.0)
    theta = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    sigmas: dict[float, float] = {}
    for ang in np.arange(0.0, 180.0, 15.0):
        diff = np.abs((theta - ang + 90.0) % 180.0 - 90.0)
        sel = edges & (diff < 15.0)
        if sel.sum() < 20:
            continue
        t = np.deg2rad(ang)
        d_orig = np.abs(gx[sel] * np.cos(t) + gy[sel] * np.sin(t))
        d_reb = np.abs(rx[sel] * np.cos(t) + ry[sel] * np.sin(t))
        ratio = d_orig / np.maximum(d_reb, 1e-6)
        ratio = ratio[(ratio > 1.0001) & (mag[sel] > 0)]
        if ratio.size < 10:
            continue
        sigma = reblur_sigma / np.sqrt(np.maximum(ratio**2 - 1.0, 1e-6))
        sigmas[float(ang)] = float(np.median(np.clip(sigma, 0.0, 10.0)))
    if not sigmas:
        return BlurEstimate(0.0, 0.0, 0.0)
    best = max(sigmas, key=sigmas.get)  # type: ignore[arg-type]
    perp = (best + 90.0) % 180.0
    nearest_perp = min(sigmas, key=lambda a: abs((a - perp + 90.0) % 180.0 - 90.0))

    def corrected(sigma: float) -> float:
        # Дискретные производные дают на идеально резком крае σ≈0,8 — вычитаем квадратично.
        return float(np.sqrt(max(sigma * sigma - SHARP_FLOOR * SHARP_FLOOR, 0.0))) / scale

    major = corrected(sigmas[best])
    minor = min(corrected(sigmas[nearest_perp]), major)
    return BlurEstimate(sigma_major=major, sigma_minor=minor, angle=best)


def richardson_lucy(
    image: np.ndarray,
    psf_sigma: float = 1.2,
    iterations: int = 12,
    psf: np.ndarray | None = None,
    ringing_control: int | None = None,
) -> np.ndarray:
    """Деконволюция Ричардсона–Люси, отдельно по каналам B, G, R.

    Размытие происходит в RGB, поэтому и восстанавливать нужно там: деконволюция
    одной яркости (L в Lab — нелинейное пространство) не возвращает цветные края
    и на тестах ухудшала картинку, а по каналам ошибка падает примерно на 20 %.
    ``psf`` — своё ядро (иначе гауссово с ``psf_sigma``).
    ``ringing_control`` — радиус ограничения по локальному диапазону исходника:
    размытие только сглаживает экстремумы, поэтому восстановленные значения
    не должны выходить за min/max исходника в окне порядка ширины смаза.
    Всё, что выходит, — «звон», и оно срезается.
    """
    ensure_bgr_u8(image)
    if iterations < 1:
        return image.copy()
    kernel = gaussian_psf(psf_sigma) if psf is None else psf.astype(np.float32)
    kernel_mirror = kernel[::-1, ::-1].copy()
    eps = 1e-6

    out = np.empty(image.shape, dtype=np.uint8)
    for c in range(3):
        observed = image[:, :, c].astype(np.float32) / 255.0 + eps
        estimate = observed.copy()
        for _ in range(iterations):
            blurred = cv2.filter2D(estimate, -1, kernel, borderType=cv2.BORDER_REFLECT)
            estimate *= cv2.filter2D(observed / (blurred + eps), -1, kernel_mirror, borderType=cv2.BORDER_REFLECT)
            np.clip(estimate, 0.0, 1.0, out=estimate)
        if ringing_control:
            estimate = clamp_to_local_range(
                estimate, observed, radius=ringing_control, overshoot=0.05, min_overshoot=0.003
            )
        out[:, :, c] = np.clip(estimate * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def smart_deblur(image: np.ndarray, blur: BlurEstimate | None = None, max_sigma: float = 3.0) -> np.ndarray:
    """Деконволюция с ядром, оценённым по самому кадру (сила и направление смаза).

    Резкий кадр (σ < 0,6 px) не меняется. Слишком сильный смаз ограничивается
    ``max_sigma``: классическая деконволюция на нём даёт больше артефактов, чем пользы.
    """
    ensure_bgr_u8(image)
    blur = blur or estimate_blur(image)
    if blur.is_sharp:
        return image.copy()
    major = min(blur.sigma_major, max_sigma)
    minor = min(blur.sigma_minor, major)
    psf = anisotropic_psf(major, minor, blur.angle)
    iterations = int(np.clip(8 + 5 * major, 10, 25))
    radius = max(1, int(np.ceil(major * 3.0)))  # окно ≈ ширине размытого края
    return richardson_lucy(image, iterations=iterations, psf=psf, ringing_control=radius)


def describe_blur(blur: BlurEstimate) -> str:
    if blur.is_sharp:
        return "смаз не найден"
    if blur.is_directional:
        return f"смаз σ≈{blur.sigma_major:.1f} px, направление {blur.angle:.0f}°"
    return f"размытие σ≈{blur.sigma_major:.1f} px"
