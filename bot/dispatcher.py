"""Создание Bot и Dispatcher (aiogram 3)."""

from __future__ import annotations

from functools import lru_cache

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import get_config
from bot.handlers import router


@lru_cache(maxsize=1)
def get_dispatcher() -> Dispatcher:
    """Диспетчер без состояния: создаётся один раз на инстанс функции."""
    dp = Dispatcher()
    dp.include_router(router)
    return dp


def create_bot() -> Bot:
    """Новый Bot на каждый запрос.

    Сессия aiohttp привязана к event loop; в serverless loop между вызовами
    может смениться, поэтому Bot не кэшируется, а закрывается после запроса.
    """
    return Bot(
        token=get_config().bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
