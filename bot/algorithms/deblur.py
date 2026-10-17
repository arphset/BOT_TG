"""Деконволюция Ричардсона–Люси с гауссовой PSF."""

import cv2
import numpy as np

from bot.algorithms._utils import ensure_bgr_u8, from_lab, to_lab


def gaussian_psf(sigma: float) -> np.ndarray:
    """Нормированное гауссово ядро размытия."""
    size = max(3, int(np.ceil(sigma * 3)) * 2 + 1)
    k1d = cv2.getGaussianKernel(size, sigma, cv2.CV_32F)
    psf = k1d @ k1d.T
    return (psf / psf.sum()).astype(np.float32)


def richardson_lucy(
    image: np.ndarray,
    psf_sigma: float = 1.2,
    iterations: int = 12,
) -> np.ndarray:
    """Деконволюция Ричардсона–Люси по яркости (канал L в Lab).

    Работа только с яркостью втрое быстрее и не даёт цветных артефактов.
    Свёртка — через ``cv2.filter2D``, что быстрее FFT для малых ядер.
    """
    ensure_bgr_u8(image)
    if iterations < 1:
        return image.copy()
    psf = gaussian_psf(psf_sigma)
    psf_mirror = psf[::-1, ::-1].copy()
    eps = 1e-6

    lab = to_lab(image)
    observed = lab[:, :, 0] / 100.0 + eps
    estimate = observed.copy()
    for _ in range(iterations):
        blurred = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        ratio = observed / (blurred + eps)
        estimate *= cv2.filter2D(ratio, -1, psf_mirror, borderType=cv2.BORDER_REFLECT)
        np.clip(estimate, 0.0, 1.0, out=estimate)

    lab[:, :, 0] = estimate * 100.0
    return from_lab(lab)
