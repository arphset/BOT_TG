from __future__ import annotations

import numpy as np
import pytest

from bot.pipeline import (
    NEURAL_MODES,
    Mode,
    PipelineCancelled,
    Step,
    SuperResSettings,
    build_steps,
    order_steps,
    pick_variant,
    run_pipeline,
    toggle,
)
from tests.conftest import add_noise, make_scene


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_runs_on_single_frame(mode: Mode, scene: np.ndarray) -> None:
    frames = [add_noise(scene, 8)]
    result = run_pipeline(frames, build_steps(mode, frames))
    scale = 4 if mode in NEURAL_MODES else 1
    assert result.image.shape == (scene.shape[0] * scale, scene.shape[1] * scale, 3)
    assert result.image.dtype == np.uint8
    assert result.errors == []
    assert result.applied
    assert not any("ECC" in s or "слияние" in s for s in result.applied)


def test_deblur_only_on_explicit_choice(scene: np.ndarray) -> None:
    for mode in (Mode.AUTO, Mode.DENOISE, Mode.SHARPEN):
        assert "deblur" not in {s.name for s in build_steps(mode, [scene]) if s.enabled}
    for mode in (Mode.DEBLUR, Mode.FULL):
        assert "deblur" in {s.name for s in build_steps(mode, [scene]) if s.enabled}


def test_auto_skips_denoise_on_clean_image(scene: np.ndarray) -> None:
    steps = {s.name: s.enabled for s in build_steps(Mode.AUTO, [scene])}
    assert steps["denoise"] is False
    noisy = add_noise(scene, 15)
    steps = {s.name: s.enabled for s in build_steps(Mode.AUTO, [noisy])}
    assert steps["denoise"] is True


def test_toggle_disables_step(scene: np.ndarray) -> None:
    steps = toggle(build_steps(Mode.FULL, [scene]), "contrast", False)
    result = run_pipeline([scene], steps)
    assert not any("CLAHE" in s for s in result.applied)
    assert any("резк" in s for s in result.applied)


def test_failing_step_is_skipped(scene: np.ndarray) -> None:
    def boom(_: np.ndarray) -> np.ndarray:
        raise RuntimeError("сломалось")

    steps = [
        Step("broken", "сломанный шаг", "image", boom),
        Step("invert", "инверсия", "image", lambda img: 255 - img),
    ]
    result = run_pipeline([scene], steps)
    assert [e.name for e in result.errors] == ["broken"]
    assert result.applied == ["инверсия"]
    np.testing.assert_array_equal(result.image, 255 - scene)


def test_series_steps_run_for_multiple_frames(scene: np.ndarray) -> None:
    frames = [add_noise(scene, 10, seed=i) for i in range(4)]
    result = run_pipeline(frames, build_steps(Mode.SHARPEN, frames))
    assert result.applied[0].startswith("выравнивание")
    assert result.applied[1].startswith("слияние")
    assert result.image.shape == scene.shape


def test_alignment_failure_does_not_stop_pipeline() -> None:
    flat = np.full((80, 100, 3), 100, dtype=np.uint8)
    result = run_pipeline([flat, flat.copy()], build_steps(Mode.SHARPEN, [flat, flat]))
    assert [e.name for e in result.errors] == ["align"]
    assert any("слияние" in s for s in result.applied)


@pytest.mark.parametrize("mode", [Mode.NEURAL, Mode.NEURAL_WDN])
def test_neural_order_is_fixed(mode: Mode, scene: np.ndarray) -> None:
    names = [s.name for s in build_steps(mode, [scene])]
    assert names == ["align", "merge", "superres", "contrast", "sharpen"]


def test_neural_mode_upscales(scene: np.ndarray) -> None:
    small = scene[:48, :64].copy()
    result = run_pipeline([small], build_steps(Mode.NEURAL, [small]))
    assert result.errors == []
    assert result.image.shape == (192, 256, 3)
    assert result.applied[0].startswith("нейросеть")
    assert result.applied[-1] == "нерезкое маскирование"


def test_neural_disabled_gives_classic(scene: np.ndarray) -> None:
    steps = build_steps(Mode.NEURAL, [scene], SuperResSettings(enabled=False))
    result = run_pipeline([scene], steps)
    assert result.image.shape == scene.shape
    assert not any("нейросеть" in s for s in result.applied)


def test_neural_over_budget_falls_back_to_classic(scene: np.ndarray) -> None:
    steps = build_steps(Mode.NEURAL, [scene], SuperResSettings(time_budget=1e-6, tile=32))
    result = run_pipeline([scene], steps)
    assert [e.name for e in result.errors] == ["superres"]
    assert "не укладывается" in result.errors[0].message
    assert result.image.shape == scene.shape  # классический результат в исходном размере
    assert "нерезкое маскирование" in result.applied


def test_sharpen_always_last(scene: np.ndarray) -> None:
    order: list[str] = []

    def mark(name: str):
        def f(img: np.ndarray) -> np.ndarray:
            order.append(name)
            return img

        return f

    steps = [
        Step("sharpen", "резкость", "image", mark("sharpen")),
        Step("contrast", "контраст", "image", mark("contrast")),
        Step("superres", "нейросеть", "image", mark("superres")),
    ]
    run_pipeline([scene], steps)
    assert order == ["contrast", "superres", "sharpen"]
    assert [s.name for s in order_steps(steps)] == ["contrast", "superres", "sharpen"]


def test_flags_are_independent(scene: np.ndarray) -> None:
    small = scene[:48, :64].copy()
    steps = toggle(build_steps(Mode.NEURAL, [small]), "contrast", False)
    result = run_pipeline([small], steps)
    assert not any("CLAHE" in s for s in result.applied)
    assert result.image.shape == (192, 256, 3)


def test_auto_neural_order(scene: np.ndarray) -> None:
    names = [s.name for s in build_steps(Mode.AUTO_NEURAL, [scene])]
    assert names == ["align", "merge", "superres", "deblock", "denoise", "color", "contrast", "sharpen"]


def test_auto_neural_picks_model_by_noise(scene: np.ndarray) -> None:
    def superres_title(img: np.ndarray) -> str:
        return next(s.title for s in build_steps(Mode.AUTO_NEURAL, [img]) if s.name == "superres")

    assert superres_title(scene).endswith("(wdn-x4v3)")
    assert "50/50" in superres_title(add_noise(scene, 3))
    assert superres_title(add_noise(scene, 15)).endswith("(x4v3)")


def test_pick_variant_thresholds() -> None:
    assert pick_variant(0.5) == "wdn"
    assert pick_variant(2.5) == "balanced"
    assert pick_variant(8.0) == "general"


def test_auto_neural_skips_classic_denoise_when_model_works() -> None:
    small = add_noise(make_scene(48, 64), 15)
    result = run_pipeline([small], build_steps(Mode.AUTO_NEURAL, [small]))
    assert result.errors == []
    assert result.image.shape == (192, 256, 3)
    assert not any("non-local" in s for s in result.applied)


def test_auto_neural_falls_back_to_auto(scene: np.ndarray) -> None:
    noisy = add_noise(scene, 15)
    steps = build_steps(Mode.AUTO_NEURAL, [noisy], SuperResSettings(time_budget=1e-6, tile=32))
    titles = {s.title: s.name for s in steps}
    result = run_pipeline([noisy], steps)
    assert [e.name for e in result.errors] == ["superres"]
    assert result.image.shape == noisy.shape
    assert [titles[t] for t in result.applied] == ["denoise", "color", "contrast", "sharpen"]


def test_fallback_runs_when_primary_disabled(scene: np.ndarray) -> None:
    noisy = add_noise(scene, 15)
    steps = build_steps(Mode.AUTO_NEURAL, [noisy], SuperResSettings(enabled=False))
    result = run_pipeline([noisy], steps)
    assert any("non-local" in s for s in result.applied)


def test_stop_flag_cancels(scene: np.ndarray) -> None:
    with pytest.raises(PipelineCancelled):
        run_pipeline([scene], build_steps(Mode.FULL, [scene]), should_stop=lambda: True)


def test_empty_frames_rejected() -> None:
    with pytest.raises(ValueError):
        run_pipeline([], [])
