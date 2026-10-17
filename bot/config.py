"""Чтение конфигурации из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


class ConfigError(RuntimeError):
    """Не задана обязательная переменная окружения."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, "").strip().lower() or default
    if value not in allowed:
        raise ConfigError(f"{name} должна быть одним из: {', '.join(sorted(allowed))}")
    return value


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"не задана переменная окружения {name}")
    return value


@dataclass(frozen=True)
class Config:
    bot_token: str
    webhook_secret: str
    public_url: str
    max_frames: int
    process_timeout: float
    max_megapixels: float
    max_file_mb: int
    allowed_user_ids: frozenset[int]
    kv_url: str
    kv_token: str
    sr_enabled: bool
    sr_timeout: float
    sr_tile: int
    sr_max_side: int
    sr_output: str

    @property
    def kv_enabled(self) -> bool:
        return bool(self.kv_url and self.kv_token)


@lru_cache(maxsize=1)
def get_config() -> Config:
    allowed = os.getenv("ALLOWED_USER_IDS", "")
    return Config(
        bot_token=_required("TELEGRAM_BOT_TOKEN"),
        webhook_secret=_required("WEBHOOK_SECRET"),
        public_url=os.getenv("PUBLIC_URL", "").strip().rstrip("/"),
        max_frames=_int("MAX_FRAMES", 10),
        process_timeout=_float("PROCESS_TIMEOUT", 270.0),
        max_megapixels=_float("MAX_MEGAPIXELS", 16.0),
        max_file_mb=_int("MAX_FILE_MB", 20),
        allowed_user_ids=frozenset(int(x) for x in allowed.replace(" ", "").split(",") if x),
        kv_url=os.getenv("KV_REST_API_URL", "").strip().rstrip("/"),
        kv_token=os.getenv("KV_REST_API_TOKEN", "").strip(),
        sr_enabled=os.getenv("SR_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"},
        sr_timeout=_float("SR_TIMEOUT", 200.0),
        sr_tile=_int("SR_TILE", 384),
        sr_max_side=_int("SR_MAX_SIDE", 1024),
        sr_output=_choice("SR_OUTPUT", "auto", {"auto", "x4", "original"}),
    )
