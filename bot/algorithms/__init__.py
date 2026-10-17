"""Классические алгоритмы обработки изображений.

Каждая функция чистая: принимает и возвращает ``np.ndarray`` (BGR, uint8),
входной массив не изменяет.
"""

from bot.algorithms.align import align_frames
from bot.algorithms.contrast import clahe_lab
from bot.algorithms.deblur import richardson_lucy
from bot.algorithms.denoise import estimate_noise, nl_means
from bot.algorithms.hdr import average_frames, merge_mertens
from bot.algorithms.sharpen import unsharp_mask
from bot.algorithms.superres import realesrgan

__all__ = [
    "align_frames",
    "average_frames",
    "clahe_lab",
    "estimate_noise",
    "merge_mertens",
    "nl_means",
    "realesrgan",
    "richardson_lucy",
    "unsharp_mask",
]
