"""Скачивание весов Real-ESRGAN (realesr-general-x4v3 и wdn) и экспорт в ONNX,
плюс смешанная модель dn0.5 (DNI-интерполяция весов, как ``-dn 0.5`` в официальном скрипте).

Готовые .onnx уже лежат в models/, скрипт нужен, только чтобы воспроизвести их.
Официальный репозиторий публикует веса только в формате PyTorch (.pth),
поэтому конвертируем сами. Нужен torch (только для разработки, в бандл не входит):

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime
    python scripts/export_onnx.py

Архитектура повторяет realesrgan/archs/srvgg_arch.py (SRVGGNetCompact):
num_feat=64, num_conv=32, upscale=4, act=PReLU.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("export_onnx")

ROOT = Path(__file__).resolve().parent.parent
RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0"
WEIGHTS = {
    "realesr-general-x4v3": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    "realesr-general-wdn-x4v3": "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
}


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64, num_conv: int = 32, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def download(name: str, sha256: str, folder: Path) -> Path:
    path = folder / f"{name}.pth"
    urllib.request.urlretrieve(f"{RELEASE}/{name}.pth", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{path.name}: sha256 {digest} не совпадает с ожидаемым {sha256}")
    logger.info("скачан %s (%d байт), sha256 совпадает", path.name, path.stat().st_size)
    return path


def load_state(pth: Path) -> dict[str, torch.Tensor]:
    state = torch.load(pth, map_location="cpu", weights_only=True)  # без исполнения pickle-кода
    return state.get("params_ema") or state.get("params") or state


def dni(state_a: dict[str, torch.Tensor], state_b: dict[str, torch.Tensor], weight_a: float) -> dict[str, torch.Tensor]:
    """Deep Network Interpolation, как в официальном RealESRGANer: w·A + (1−w)·B."""
    return {k: weight_a * state_a[k] + (1.0 - weight_a) * state_b[k] for k in state_a}


def export(state: dict[str, torch.Tensor], onnx_path: Path) -> None:
    model = SRVGGNetCompact()
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.onnx.export(
        model,
        torch.rand(1, 3, 64, 64),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h4", 3: "w4"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    x = torch.rand(1, 3, 48, 80)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    logger.info("%s: %d байт, max |torch - onnx| = %.1e", onnx_path.name, onnx_path.stat().st_size, np.abs(ref - got).max())


DNI_WEIGHT = 0.5  # как -dn 0.5, значение по умолчанию в официальном inference_realesrgan.py


def main() -> None:
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        states = {name: load_state(download(name, sha, Path(tmp))) for name, sha in WEIGHTS.items()}
    for name, state in states.items():
        export(state, out_dir / f"{name}.onnx")
    # Смесь: x4v3 (сильное шумоподавление) и wdn (слабое) пополам.
    blended = dni(states["realesr-general-x4v3"], states["realesr-general-wdn-x4v3"], DNI_WEIGHT)
    export(blended, out_dir / "realesr-general-dn05-x4v3.onnx")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
