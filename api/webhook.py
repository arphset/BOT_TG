"""Точка входа Vercel: ASGI-приложение, принимающее POST от Telegram.

Vercel находит переменную ``app`` и обслуживает её по пути ``/api/webhook``.
Используется чистый ASGI без веб-фреймворка — меньше зависимостей и
меньше размер функции.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiogram.types import Update  # noqa: E402

from bot import dispatcher as bot_dispatcher  # noqa: E402
from bot.config import get_config  # noqa: E402
from bot.storage import get_storage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("webhook")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

SECRET_HEADER = b"x-telegram-bot-api-secret-token"


async def _respond(send: Send, status: int, body: bytes = b"ok") -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            return b"".join(chunks)


def is_secret_valid(headers: list[tuple[bytes, bytes]], expected: str) -> bool:
    """Проверка заголовка X-Telegram-Bot-Api-Secret-Token за постоянное время."""
    for name, value in headers:
        if name.lower() == SECRET_HEADER:
            return hmac.compare_digest(value, expected.encode())
    return False


_recent_updates: OrderedDict[int, None] = OrderedDict()
_RECENT_LIMIT = 1000


def _seen_locally(update_id: int) -> bool:
    """Защита от повторов внутри одного инстанса (Fluid compute переиспользует инстансы).

    Это не хранение состояния, а страховка: надёжная дедупликация — через KV.
    """
    if update_id in _recent_updates:
        return True
    _recent_updates[update_id] = None
    while len(_recent_updates) > _RECENT_LIMIT:
        _recent_updates.popitem(last=False)
    return False


async def handle_update(data: dict[str, Any]) -> None:
    """Передаёт обновление в диспетчер aiogram. Bot создаётся на один запрос."""
    update_id = data.get("update_id")
    if isinstance(update_id, int) and (_seen_locally(update_id) or not await get_storage().claim_update(update_id)):
        logger.info("обновление %s уже обрабатывалось, пропускаю (повтор от Telegram)", update_id)
        return
    dp = bot_dispatcher.get_dispatcher()
    async with bot_dispatcher.create_bot() as bot:
        update = Update.model_validate(data, context={"bot": bot})
        await dp.feed_update(bot, update)


async def app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["type"] != "http":
        return

    if scope.get("method") != "POST":
        version = os.getenv("VERCEL_GIT_COMMIT_SHA", "local")[:7]
        await _respond(send, 200, f"photo-bot webhook is alive ({version})".encode())
        return

    try:
        cfg = get_config()
    except Exception:  # noqa: BLE001
        logger.exception("ошибка конфигурации")
        await _respond(send, 500, b"misconfigured")
        return

    if not is_secret_valid(scope.get("headers", []), cfg.webhook_secret):
        logger.warning("запрос без верного секретного заголовка отклонён")
        await _respond(send, 403, b"forbidden")
        return

    body = await _read_body(receive)
    try:
        data = json.loads(body)
        await handle_update(data)
    except Exception:  # noqa: BLE001 — всегда 200, чтобы Telegram не повторял запрос бесконечно
        logger.exception("ошибка при обработке обновления")
    await _respond(send, 200)
