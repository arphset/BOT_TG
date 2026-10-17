"""Классические алгоритмы обработки изображений.

Каждая функция чистая: принимает и возвращает ``np.ndarray`` (BGR, uint8),
входной массив не изменяет.
"""

from bot.algorithms.align import align_frames
from bot.algorithms.color import auto_levels, color_correct, vibrance, white_balance
from bot.algorithms.contrast import clahe_lab
from bot.algorithms.deblur import richardson_lucy
from bot.algorithms.denoise import estimate_noise, nl_means
from bot.algorithms.hdr import average_frames, merge_mertens
from bot.algorithms.sharpen import unsharp_mask
from bot.algorithms.superres import realesrgan

__all__ = [
    "align_frames",
    "auto_levels",
    "average_frames",
    "clahe_lab",
    "color_correct",
    "estimate_noise",
    "merge_mertens",
    "nl_means",
    "realesrgan",
    "richardson_lucy",
    "unsharp_mask",
    "vibrance",
    "white_balance",
]
