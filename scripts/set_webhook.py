"""Одноразовая регистрация webhook в Telegram.

Запуск:  python scripts/set_webhook.py           — зарегистрировать
         python scripts/set_webhook.py --info    — показать текущий webhook
         python scripts/set_webhook.py --delete  — удалить webhook

Переменные берутся из окружения или из файла .env в корне проекта.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiogram import Bot  # noqa: E402
from aiogram.types import BotCommand  # noqa: E402

logger = logging.getLogger("set_webhook")

COMMANDS = [
    BotCommand(command="start", description="Приветствие и справка"),
    BotCommand(command="mode", description="Режим обработки по умолчанию"),
    BotCommand(command="help", description="Справка"),
]


def load_dotenv(path: Path) -> None:
    """Минимальный загрузчик .env, чтобы не тянуть лишнюю зависимость."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--info", action="store_true", help="показать текущий webhook")
    group.add_argument("--delete", action="store_true", help="удалить webhook")
    parser.add_argument("--url", help="адрес деплоя (по умолчанию PUBLIC_URL)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    from bot.config import get_config

    cfg = get_config()
    async with Bot(cfg.bot_token) as bot:
        if args.info:
            info = await bot.get_webhook_info()
            logger.info("%s", info.model_dump_json(indent=2, exclude_none=True))
            return 0
        if args.delete:
            await bot.delete_webhook(drop_pending_updates=False)
            logger.info("webhook удалён")
            return 0

        base = (args.url or cfg.public_url).rstrip("/")
        if not base.startswith("https://"):
            logger.error("нужен https-адрес деплоя: задайте PUBLIC_URL или --url")
            return 1
        url = f"{base}/api/webhook"
        await bot.set_webhook(
            url=url,
            secret_token=cfg.webhook_secret,
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
            max_connections=10,
        )
        await bot.set_my_commands(COMMANDS)
        info = await bot.get_webhook_info()
        logger.info("webhook установлен: %s (ожидает обновлений: %s)", info.url, info.pending_update_count)
        if info.last_error_message:
            logger.warning("последняя ошибка Telegram: %s", info.last_error_message)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(asyncio.run(main()))
