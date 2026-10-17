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
    unsharp_mask,
)
from bot.algorithms.color import color_correct
from bot.algorithms.deblock import deblock, is_blocky
from bot.algorithms.deblur import describe_blur, estimate_blur, smart_deblur
from bot.algorithms.denoise import denoise_by_luma
from bot.algorithms.hdr import exposure_spread
from bot.algorithms.superres import VARIANT_TITLES, OutputMode, Variant, output_is_x4, realesrgan_fit

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
    output: OutputMode = "auto"


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


NOISE_STRONG = 3.5  # σ выше — сильный шум
NOISE_MILD = 1.5  # σ ниже — кадр практически чистый


def pick_variant(sigma: float) -> Variant:
    """Модель Real-ESRGAN по оценке шума: сильный → x4v3, средний → смесь 50/50, чистый → wdn."""
    if sigma > NOISE_STRONG:
        return "general"
    if sigma > NOISE_MILD:
        return "balanced"
    return "wdn"


def build_steps(
    mode: Mode,
    frames: Sequence[np.ndarray],
    superres: SuperResSettings = SuperResSettings(),
) -> list[Step]:
    """Собирает список шагов для режима.

    Классический порядок: выравнивание → слияние → деблокинг JPEG → шумоподавление
    → деконволюция → цветокоррекция → CLAHE → нерезкое маскирование.
    Деблокинг включается, только если в кадре найдена сетка JPEG-блоков;
    шумоподавление — по оценке шума (сила по зонам яркости); деконволюция —
    только в «Размытии» и «Полном конвейере», с ядром, оценённым по кадру.

    Нейросетевые режимы: выравнивание → слияние → Real-ESRGAN → CLAHE → резкость
    (в «Авто + нейросеть» ещё цветокоррекция и запасные классические шаги).
    """
    first = frames[0]
    sigma = estimate_noise(first)
    multi = len(frames) > 1
    # После усреднения N кадров шум меньше примерно в √N раз.
    effective_sigma = sigma / np.sqrt(len(frames)) if multi else sigma
    blocky = is_blocky(first)
    logger.info("режим=%s, кадров=%d, шум σ=%.2f, JPEG-блоки=%s", mode.value, len(frames), sigma, blocky)

    align = Step("align", "выравнивание кадров (ECC)", "series", align_frames, enabled=multi)
    merge = Step("merge", "слияние серии (Mertens / усреднение)", "merge", _merge_series, enabled=multi)
    color = Step("color", "цветокоррекция (баланс белого, уровни, вибрация)", "image", color_correct)

    def deblock_step(**kw: object) -> Step:
        return Step("deblock", "удаление JPEG-артефактов (деблокинг)", "image", deblock, enabled=blocky, **kw)  # type: ignore[arg-type]

    def denoise_step(enabled: bool = True, min_strength: float = 2.5, **kw: object) -> Step:
        return Step(
            "denoise",
            "шумоподавление (non-local means, по зонам яркости)",
            "image",
            partial(denoise_by_luma, min_strength=min_strength),
            enabled=enabled,
            **kw,  # type: ignore[arg-type]
        )

    def deblur_step() -> Step:
        blur = estimate_blur(first)
        return Step(
            "deblur",
            f"деконволюция Ричардсона–Люси ({describe_blur(blur)})",
            "image",
            partial(smart_deblur, blur=blur),
        )

    def contrast_step(clip: float) -> Step:
        return Step(
            "contrast",
            "локальный контраст (адаптивный CLAHE, Lab)",
            "image",
            partial(clahe_lab, clip_limit=clip, adaptive=True),
        )

    def sharpen_step(sigma_px: float, amount: float) -> Step:
        return Step("sharpen", "нерезкое маскирование", "image", partial(unsharp_mask, sigma=sigma_px, amount=amount))

    if mode is Mode.AUTO:
        return [
            align,
            merge,
            deblock_step(),
            denoise_step(enabled=effective_sigma > 2.0),
            color,
            contrast_step(1.5),
            sharpen_step(1.2, 0.48),
        ]
    if mode is Mode.DENOISE:
        return [align, merge, deblock_step(), denoise_step(min_strength=6.0)]
    if mode is Mode.SHARPEN:
        return [align, merge, deblock_step(), sharpen_step(1.5, 1.0)]
    if mode is Mode.DEBLUR:
        return [align, merge, deblock_step(), deblur_step()]
    if mode is Mode.FULL:
        return [
            align,
            merge,
            deblock_step(),
            denoise_step(enabled=effective_sigma > 1.0),
            deblur_step(),
            color,
            contrast_step(2.0),
            sharpen_step(1.0, 0.32),
        ]
    if mode in (Mode.NEURAL, Mode.NEURAL_WDN):
        variant: Variant = "wdn" if mode is Mode.NEURAL_WDN else "general"
        return [
            align,
            merge,
            _superres_step(variant, superres, first.shape),
            contrast_step(1.5),
            sharpen_step(1.5, 0.4),
        ]
    if mode is Mode.AUTO_NEURAL:
        # Модель по шуму: x4v3 сильно чистит шум, wdn сохраняет текстуру, смесь 50/50 — между ними.
        # Классические деблокинг и шумоподавление перед нейросетью не нужны — модель справляется
        # сама; они включаются только как запасные шаги, если нейросеть не применилась.
        return [
            align,
            merge,
            _superres_step(pick_variant(effective_sigma), superres, first.shape),
            deblock_step(fallback_for="superres"),
            denoise_step(enabled=effective_sigma > 2.0, fallback_for="superres"),
            color,
            contrast_step(1.5),
            sharpen_step(1.5, 0.4),
        ]
    raise ValueError(f"неизвестный режим: {mode}")


def _superres_step(variant: Variant, superres: SuperResSettings, shape: tuple[int, ...]) -> Step:
    size = "×4" if output_is_x4(shape, superres.output, superres.max_side) else "в исходном размере"
    return Step(
        "superres",
        f"нейросеть Real-ESRGAN {size} ({VARIANT_TITLES[variant]})",
        "image",
        partial(
            realesrgan_fit,
            variant=variant,
            output=superres.output,
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
