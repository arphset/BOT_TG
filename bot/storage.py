"""Внешнее хранилище состояния (Upstash Redis / Vercel KV через REST API).

В первой итерации используется для двух вещей, и обе необязательны:

* режим по умолчанию пользователя (``/mode``);
* защита от повторной обработки одного и того же обновления, если
  Telegram повторил запрос webhook.

Если ``KV_REST_API_URL``/``KV_REST_API_TOKEN`` не заданы, хранилище
отключено: бот работает, но ``/mode`` не может запомнить выбор.
Во второй итерации здесь же будет сборка альбомов.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from bot.config import get_config

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=5)


class Storage:
    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._token = token

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._token)

    async def _command(self, *args: str | int) -> Any:
        """Выполняет одну Redis-команду через REST (формат Upstash)."""
        if not self.enabled:
            return None
        headers = {"Authorization": f"Bearer {self._token}"}
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(self._url, json=[str(a) for a in args], headers=headers) as resp:
                data = await resp.json(content_type=None)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"KV error: {data['error']}")
        return data.get("result") if isinstance(data, dict) else None

    async def get_mode(self, user_id: int) -> str | None:
        try:
            value = await self._command("GET", f"photobot:mode:{user_id}")
        except Exception:  # noqa: BLE001 — хранилище не должно ронять бота
            logger.exception("не удалось прочитать режим из KV")
            return None
        return str(value) if value else None

    async def set_mode(self, user_id: int, mode: str | None) -> bool:
        try:
            if mode is None:
                await self._command("DEL", f"photobot:mode:{user_id}")
            else:
                await self._command("SET", f"photobot:mode:{user_id}", mode)
        except Exception:  # noqa: BLE001
            logger.exception("не удалось сохранить режим в KV")
            return False
        return True

    async def claim_update(self, update_id: int, ttl: int = 3600) -> bool:
        """True, если обновление видим впервые (SET NX). Без KV всегда True."""
        if not self.enabled:
            return True
        try:
            res = await self._command("SET", f"photobot:upd:{update_id}", 1, "NX", "EX", ttl)
        except Exception:  # noqa: BLE001
            logger.exception("KV недоступно, дедупликация пропущена")
            return True
        return res == "OK"


def get_storage() -> Storage:
    cfg = get_config()
    return Storage(cfg.kv_url, cfg.kv_token)
