"""Конвейер обработки: список шагов с флагами.

Шаг бывает трёх видов:

* ``series`` — работает с серией и возвращает серию (выравнивание);
* ``merge``  — сводит серию в один кадр (HDR / усреднение);
* ``image``  — работает с одним кадром (шум, контраст, резкость, нейросеть...).

Шаги ``series`` и ``merge`` для одиночного фото пропускаются автоматически.
Флаги шагов независимы; нерезкое маскирование всегда выполняется последним.
Ошибка на шаге не останавливает конвейер: шаг помечается как неудачный,
дальше идёт результат предыдущего шага.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import Literal

import numpy as np

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
from bot.algorithms.hdr import exposure_spread
from bot.algorithms.superres import Variant, realesrgan

logger = logging.getLogger(__name__)

StepKind = Literal["series", "merge", "image"]


class Mode(str, Enum):
    AUTO = "auto"
    DENOISE = "denoise"
    SHARPEN = "sharpen"
    DEBLUR = "deblur"
    FULL = "full"
    NEURAL = "neural"
    NEURAL_WDN = "neural_wdn"
    AUTO_NEURAL = "auto_neural"


MODE_TITLES: dict[Mode, str] = {
    Mode.AUTO: "Авто",
    Mode.DENOISE: "Шум",
    Mode.SHARPEN: "Резкость",
    Mode.DEBLUR: "Размытие",
    Mode.FULL: "Полный конвейер",
    Mode.NEURAL: "Нейросеть ×4",
    Mode.NEURAL_WDN: "Нейросеть ×4, детали",
    Mode.AUTO_NEURAL: "Авто + нейросеть",
}

NEURAL_MODES = frozenset({Mode.NEURAL, Mode.NEURAL_WDN, Mode.AUTO_NEURAL})


@dataclass(frozen=True)
class SuperResSettings:
    """Настройки нейросетевого шага (Real-ESRGAN)."""

    enabled: bool = True
    max_side: int = 1024
    tile: int = 384
    overlap: int = 16
    time_budget: float = 200.0


class PipelineCancelled(RuntimeError):
    """Конвейер остановлен извне (истёк таймаут)."""


@dataclass(frozen=True)
class Step:
    name: str
    title: str
    kind: StepKind
    func: Callable[..., object]
    enabled: bool = True
    # Запасной шаг: выполняется, только если шаг с этим именем не применился
    # (выключен флагом, упал или не уложился во время).
    fallback_for: str | None = None


@dataclass(frozen=True)
class StepError:
    name: str
    title: str
    message: str


@dataclass
class PipelineResult:
    image: np.ndarray
    applied: list[str] = field(default_factory=list)
    errors: list[StepError] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    elapsed: float = 0.0


def _merge_series(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Mertens, если экспозиции заметно различаются, иначе усреднение."""
    if exposure_spread(frames) > 20.0:
        return merge_mertens(frames)
    return average_frames(frames)


def _auto_denoise_strength(sigma: float) -> float:
    return float(np.clip(sigma * 1.3, 3.0, 12.0))


def build_steps(
    mode: Mode,
    frames: Sequence[np.ndarray],
    superres: SuperResSettings = SuperResSettings(),
) -> list[Step]:
    """Собирает список шагов для режима.

    Для «Авто» и «Полного конвейера» сила шумоподавления подбирается
    по оценке шума первого кадра. Деконволюция включается только в
    режимах «Размытие» и «Полный конвейер», то есть по явному выбору.

    Нейросетевые режимы идут в фиксированном порядке:
    выравнивание → слияние → Real-ESRGAN → CLAHE → нерезкое маскирование.
    """
    sigma = estimate_noise(frames[0])
    multi = len(frames) > 1
    # После усреднения N кадров шум меньше примерно в √N раз.
    effective_sigma = sigma / np.sqrt(len(frames)) if multi else sigma
    logger.info("режим=%s, кадров=%d, оценка шума σ=%.2f", mode.value, len(frames), sigma)

    align = Step("align", "выравнивание кадров (ECC)", "series", align_frames, enabled=multi)
    merge = Step("merge", "слияние серии (Mertens / усреднение)", "merge", _merge_series, enabled=multi)

    if mode is Mode.AUTO:
        need_denoise = effective_sigma > 2.0
        return [
            align,
            merge,
            Step(
                "denoise",
                "шумоподавление (non-local means)",
                "image",
                partial(nl_means, strength=_auto_denoise_strength(effective_sigma)),
                enabled=need_denoise,
            ),
            Step("contrast", "локальный контраст (CLAHE, Lab)", "image", partial(clahe_lab, clip_limit=1.5)),
            Step("sharpen", "нерезкое маскирование", "image", partial(unsharp_mask, sigma=1.2, amount=0.6)),
        ]
    if mode is Mode.DENOISE:
        return [
            align,
            merge,
            Step(
                "denoise",
                "шумоподавление (non-local means)",
                "image",
                partial(nl_means, strength=max(6.0, _auto_denoise_strength(effective_sigma))),
            ),
        ]
    if mode is Mode.SHARPEN:
        return [
            align,
            merge,
            Step("sharpen", "нерезкое маскирование", "image", partial(unsharp_mask, sigma=1.5, amount=1.0)),
        ]
    if mode is Mode.DEBLUR:
        return [
            align,
            merge,
            Step(
                "deblur",
                "деконволюция Ричардсона–Люси",
                "image",
                partial(richardson_lucy, psf_sigma=1.5, iterations=15),
            ),
        ]
    if mode is Mode.FULL:
        return [
            align,
            merge,
            Step(
                "denoise",
                "шумоподавление (non-local means)",
                "image",
                partial(nl_means, strength=_auto_denoise_strength(effective_sigma)),
                enabled=effective_sigma > 1.0,
            ),
            Step(
                "deblur",
                "деконволюция Ричардсона–Люси",
                "image",
                partial(richardson_lucy, psf_sigma=1.0, iterations=10),
            ),
            Step("contrast", "локальный контраст (CLAHE, Lab)", "image", partial(clahe_lab, clip_limit=2.0)),
            Step(
                "sharpen",
                "нерезкое маскирование",
                "image",
                partial(unsharp_mask, sigma=1.0, amount=0.4),
            ),
        ]
    if mode in (Mode.NEURAL, Mode.NEURAL_WDN):
        variant: Variant = "wdn" if mode is Mode.NEURAL_WDN else "general"
        return [
            align,
            merge,
            _superres_step(variant, superres),
            Step("contrast", "локальный контраст (CLAHE, Lab)", "image", partial(clahe_lab, clip_limit=1.5)),
            Step("sharpen", "нерезкое маскирование", "image", partial(unsharp_mask, sigma=1.5, amount=0.5)),
        ]
    if mode is Mode.AUTO_NEURAL:
        # Модель по шуму: x4v3 сильно чистит шум, wdn сохраняет текстуру.
        # Классическое шумоподавление перед нейросетью не нужно — модель чистит сама;
        # оно включается только как запасной шаг, если нейросеть не применилась.
        noisy = effective_sigma > 2.0
        return [
            align,
            merge,
            _superres_step("general" if noisy else "wdn", superres),
            Step(
                "denoise",
                "шумоподавление (non-local means)",
                "image",
                partial(nl_means, strength=_auto_denoise_strength(effective_sigma)),
                enabled=noisy,
                fallback_for="superres",
            ),
            Step("contrast", "локальный контраст (CLAHE, Lab)", "image", partial(clahe_lab, clip_limit=1.5)),
            Step("sharpen", "нерезкое маскирование", "image", partial(unsharp_mask, sigma=1.5, amount=0.5)),
        ]
    raise ValueError(f"неизвестный режим: {mode}")


def _superres_step(variant: Variant, superres: SuperResSettings) -> Step:
    return Step(
        "superres",
        f"нейросеть Real-ESRGAN ×4 ({variant}-x4v3)",
        "image",
        partial(
            realesrgan,
            variant=variant,
            max_side=superres.max_side,
            tile=superres.tile,
            overlap=superres.overlap,
            time_budget=superres.time_budget,
        ),
        enabled=superres.enabled,
    )


def order_steps(steps: Sequence[Step]) -> list[Step]:
    """Нерезкое маскирование всегда выполняется последним, остальной порядок сохраняется."""
    return [s for s in steps if s.name != "sharpen"] + [s for s in steps if s.name == "sharpen"]


def toggle(steps: Sequence[Step], name: str, enabled: bool) -> list[Step]:
    """Возвращает новый список шагов с изменённым флагом у шага ``name``."""
    return [replace(s, enabled=enabled) if s.name == name else s for s in steps]


def run_pipeline(
    frames: Sequence[np.ndarray],
    steps: Sequence[Step],
    should_stop: Callable[[], bool] = lambda: False,
) -> PipelineResult:
    """Выполняет включённые шаги по порядку.

    ``should_stop`` проверяется перед каждым шагом: так поток обработки
    прекращает работу вскоре после истечения таймаута.
    """
    if not frames:
        raise ValueError("нет кадров для обработки")
    started = time.perf_counter()
    series: list[np.ndarray] = list(frames)
    image: np.ndarray | None = None
    result = PipelineResult(image=frames[0])
    done: set[str] = set()

    for step in order_steps(steps):
        if not step.enabled:
            continue
        if step.fallback_for is not None and step.fallback_for in done:
            continue
        if should_stop():
            raise PipelineCancelled("обработка остановлена по таймауту")
        if step.kind in ("series", "merge") and (image is not None or len(series) < 2):
            continue

        t0 = time.perf_counter()
        try:
            if step.kind == "series":
                series = list(step.func(series))  # type: ignore[arg-type]
            elif step.kind == "merge":
                image = step.func(series)  # type: ignore[assignment]
            else:
                current = image if image is not None else series[0]
                image = step.func(current)  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001 — шаг не должен ронять конвейер
            logger.exception("шаг %s завершился ошибкой", step.name)
            result.errors.append(StepError(step.name, step.title, str(exc) or type(exc).__name__))
            continue
        finally:
            result.timings[step.name] = time.perf_counter() - t0
        result.applied.append(step.title)
        done.add(step.name)

    result.image = image if image is not None else series[0]
    result.elapsed = time.perf_counter() - started
    return result
