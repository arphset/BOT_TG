"""Обработчики команд, фото и кнопок."""

from __future__ import annotations

import asyncio
import html
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import PurePath

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, Filter
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
    TelegramObject,
)

from bot.config import get_config
from bot.media import (
    ImageDecodeError,
    decode_image,
    encode_image,
    fit,
    limit_megapixels,
    make_comparison,
    read_meta,
    to_srgb,
)
from bot.pipeline import (
    DEFAULT_LEVEL,
    MODE_TITLES,
    STRENGTH_LEVELS,
    STRENGTH_TITLES,
    Mode,
    PipelineCancelled,
    PipelineResult,
    SuperResSettings,
    build_steps,
    run_pipeline,
)

logger = logging.getLogger(__name__)
router = Router(name="photo")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
PREVIEW_MAX_SIDE = 2560
UPLOAD_LIMIT = 49 * 1024 * 1024

HELP_TEXT = (
    "<b>Как пользоваться</b>\n"
    "Просто пришлите фото — лучше <b>файлом</b> (скрепка → Файл), чтобы Telegram его не сжал. "
    "Я сам разберусь, что с ним делать, и пришлю результат файлом без сжатия, превью и сравнение «до / после».\n\n"
    "Не понравилась сила обработки — под файлом есть кнопки <b>«Слабее»</b> и <b>«Сильнее»</b>: "
    "фото переделывается из оригинала, уровней пять.\n\n"
    "<b>Что я делаю (только то, что нужно этому фото)</b>\n"
    "• убираю смаз и отдельно — смаз лиц;\n"
    "• нейросеть Real-ESRGAN: шум, артефакты сжатия, детали; маленькие фото увеличиваю в 4 раза;\n"
    "• убираю дымку, вытягиваю тени и выбитые света;\n"
    "• баланс белого, уровни, цвет, локальный контраст;\n"
    "• выравниваю завал горизонта (0,5–6°, с небольшой обрезкой);\n"
    "• мягкая резкость в конце.\n"
    "В подписи к файлу видно, какие шаги сработали.\n\n"
    "Обработка — до минуты-двух. Файл до 20 МБ."
)


class AllowedUser(Filter):
    """Пропускает только пользователей из ALLOWED_USER_IDS (если список задан)."""

    async def __call__(self, event: TelegramObject) -> bool:
        allowed = get_config().allowed_user_ids
        user = getattr(event, "from_user", None)
        if not allowed:
            return True
        ok = user is not None and user.id in allowed
        if not ok:
            logger.info("доступ запрещён пользователю %s", user.id if user else None)
        return ok


router.message.filter(AllowedUser())
router.callback_query.filter(AllowedUser())


def strength_keyboard(level: int) -> InlineKeyboardMarkup | None:
    """Кнопки «Слабее» / «Сильнее» под результатом умного режима."""
    row: list[InlineKeyboardButton] = []
    if level > 0:
        row.append(InlineKeyboardButton(text="− Слабее", callback_data=f"lvl:{level - 1}"))
    if level < len(STRENGTH_LEVELS) - 1:
        row.append(InlineKeyboardButton(text="Сильнее +", callback_data=f"lvl:{level + 1}"))
    return InlineKeyboardMarkup(inline_keyboard=[row]) if row else None


@dataclass(frozen=True)
class SourceFile:
    file_id: str
    file_size: int | None
    file_name: str
    compressed: bool  # пришло как фото, а не как файл


def extract_source(message: Message) -> SourceFile | None:
    """Достаёт изображение из сообщения: документ-картинку или фото."""
    doc = message.document
    if doc is not None:
        name = doc.file_name or "image.jpg"
        is_image = (doc.mime_type or "").startswith("image/") or PurePath(name).suffix.lower() in IMAGE_EXTENSIONS
        if not is_image:
            return None
        return SourceFile(doc.file_id, doc.file_size, name, compressed=False)
    if message.photo:
        largest = message.photo[-1]
        return SourceFile(largest.file_id, largest.file_size, f"photo_{message.message_id}.jpg", compressed=True)
    return None


# ---------- команды ----------


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я улучшаю фотографии классическими алгоритмами: шумоподавление, "
        "локальный контраст, резкость, устранение лёгкого размытия.\n\n" + HELP_TEXT
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


# ---------- приём фото ----------


@router.message(F.document | F.photo)
async def on_image(message: Message, bot: Bot) -> None:
    source = extract_source(message)
    if source is None:
        await message.reply("Это не изображение. Пришлите фото файлом: JPEG, PNG, WebP или TIFF.")
        return
    if not _check_size(source):
        await message.reply(_too_big_text())
        return
    status = await message.reply("Обрабатываю… Обычно это занимает до минуты.")
    await process_and_reply(bot, message, source, Mode.SMART, status, level=DEFAULT_LEVEL)


@router.callback_query(F.data.startswith("lvl:"))
async def on_strength(callback: CallbackQuery, bot: Bot) -> None:
    """«Слабее» / «Сильнее»: переделать то же фото из оригинала с другой силой."""
    try:
        level = int((callback.data or "").split(":", 1)[1])
        if not 0 <= level < len(STRENGTH_LEVELS):
            raise ValueError
    except ValueError:
        await callback.answer("Неизвестный уровень", show_alert=True)
        return
    result_message = callback.message
    # Результат — ответ на сообщение с фото, поэтому исходник берём оттуда: без хранилища.
    original = result_message.reply_to_message if isinstance(result_message, Message) else None
    source = extract_source(original) if original is not None else None
    if not isinstance(result_message, Message) or original is None or source is None:
        await callback.answer("Не нашёл исходное фото. Пришлите его ещё раз.", show_alert=True)
        return
    title = STRENGTH_TITLES[level]
    await callback.answer(f"Переделываю: {title}")
    # Снимаем кнопки со старого результата, чтобы повторный тап не запустил вторую обработку.
    try:
        await result_message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass
    status = await original.reply(f"Переделываю ({title})…")
    await process_and_reply(bot, original, source, Mode.SMART, status, level=level)


@router.callback_query(F.data.startswith("mode:"))
async def on_mode_chosen(callback: CallbackQuery, bot: Bot) -> None:
    value = (callback.data or "").split(":", 1)[1]
    try:
        mode = Mode(value)
    except ValueError:
        await callback.answer("Неизвестный режим", show_alert=True)
        return

    menu = callback.message
    # Исходное фото берём из сообщения, на которое бот ответил кнопками:
    # так между запросами не нужно хранить состояние.
    original = menu.reply_to_message if isinstance(menu, Message) else None
    source = extract_source(original) if original is not None else None
    if not isinstance(menu, Message) or original is None or source is None:
        await callback.answer("Не нашёл исходное фото. Пришлите его ещё раз.", show_alert=True)
        return

    # Кнопки режимов остались на сообщениях, отправленных до умного режима, — они продолжают работать.
    await callback.answer(f"Режим: {MODE_TITLES[mode]}")
    # Убираем кнопки сразу, чтобы повторное нажатие не запустило вторую обработку.
    await menu.edit_text(f"Обрабатываю в режиме «{MODE_TITLES[mode]}»…")
    await process_and_reply(bot, original, source, mode, menu)


@router.message()
async def on_other(message: Message) -> None:
    await message.answer("Пришлите фото — лучше файлом, — и я его улучшу. Справка: /help")


@router.errors()
async def on_error(event: ErrorEvent, bot: Bot) -> bool:
    """Последний рубеж: любая непойманная ошибка → понятное сообщение пользователю."""
    logger.exception("необработанная ошибка в обработчике", exc_info=event.exception)
    update = event.update
    chat_id: int | None = None
    if update.message is not None:
        chat_id = update.message.chat.id
    elif update.callback_query is not None and update.callback_query.message is not None:
        chat_id = update.callback_query.message.chat.id
    if chat_id is not None:
        try:
            await bot.send_message(chat_id, "Что-то пошло не так при обработке. Попробуйте ещё раз или пришлите другой файл.")
        except Exception:  # noqa: BLE001
            logger.exception("не удалось отправить сообщение об ошибке")
    return True


# ---------- обработка ----------


def _check_size(source: SourceFile) -> bool:
    limit = get_config().max_file_mb * 1024 * 1024
    return source.file_size is None or source.file_size <= limit


def _too_big_text() -> str:
    return (
        f"Файл больше {get_config().max_file_mb} МБ — Telegram не даёт ботам скачивать такие файлы. "
        "Пришлите, пожалуйста, файл поменьше (например, JPEG вместо PNG/TIFF)."
    )


def _format_report(
    result: PipelineResult,
    total: float,
    downscaled: bool,
    mode: Mode,
    level: int | None = None,
    compressed: bool = False,
) -> str:
    if level is not None:
        head = f"<b>Готово</b> · сила: {STRENGTH_TITLES[level]}"
    else:
        head = f"<b>Готово</b> · режим «{MODE_TITLES[mode]}»"
    lines = [head, f"Время обработки: {result.elapsed:.1f} с (всего {total:.1f} с)"]
    if result.applied:
        lines.append("Шаги: " + ", ".join(result.applied) + ".")
    else:
        lines.append("Ни один шаг не применился — возвращаю исходник.")
    if downscaled:
        lines.append(f"Кадр больше {get_config().max_megapixels:g} Мп, перед обработкой уменьшен.")
    for err in result.errors:
        if err.name == "superres":
            lines.append(
                f"⚠️ Нейросетевой шаг пропущен ({html.escape(err.message)}), прислан классический результат."
            )
        else:
            lines.append(f"⚠️ Шаг «{err.title}» не выполнен ({html.escape(err.message)}), пропущен.")
    if compressed:
        lines.append("<i>Фото пришло сжатым Telegram; если отправить файлом, результат будет лучше.</i>")
    return "\n".join(lines)


def _output_name(file_name: str) -> tuple[str, str]:
    path = PurePath(file_name)
    fmt = "png" if path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"} else "jpg"
    return f"{path.stem}_enhanced.{fmt}", fmt


async def _safe_edit(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramBadRequest:
        await message.answer(text)


async def process_and_reply(
    bot: Bot,
    original: Message,
    source: SourceFile,
    mode: Mode,
    status: Message,
    level: int | None = None,
) -> None:
    """Скачивает, обрабатывает и отправляет результат. Любую ошибку превращает в понятное сообщение."""
    cfg = get_config()
    started = time.perf_counter()

    if not _check_size(source):
        await _safe_edit(status, _too_big_text())
        return
    try:
        buffer = await bot.download(source.file_id)
    except TelegramBadRequest as exc:
        logger.warning("не удалось скачать файл: %s", exc)
        text = _too_big_text() if "too big" in str(exc).lower() else "Не удалось скачать файл из Telegram. Пришлите его ещё раз."
        await _safe_edit(status, text)
        return
    data = buffer.read() if buffer is not None else b""

    try:
        image = decode_image(data)
    except ImageDecodeError:
        await _safe_edit(
            status,
            "Не получилось прочитать файл как изображение. Поддерживаются JPEG, PNG, WebP, TIFF, BMP "
            "(HEIC — сначала сконвертируйте в JPEG).",
        )
        return
    image, downscaled = limit_megapixels(image, cfg.max_megapixels)
    meta = read_meta(data)

    stop = threading.Event()
    frames = [image]
    strength = STRENGTH_LEVELS[level] if level is not None else 1.0
    superres = SuperResSettings(
        enabled=cfg.sr_enabled,
        max_side=cfg.sr_max_side,
        tile=cfg.sr_tile,
        time_budget=cfg.sr_timeout,
        output=cfg.sr_output,  # type: ignore[arg-type]
    )
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: run_pipeline(frames, build_steps(mode, frames, superres, strength), should_stop=stop.is_set)
            ),
            timeout=cfg.process_timeout,
        )
    except (asyncio.TimeoutError, PipelineCancelled):
        stop.set()
        logger.warning("таймаут обработки (%.0f с), режим %s", cfg.process_timeout, mode.value)
        await _safe_edit(
            status,
            f"Не успел обработать фото за {cfg.process_timeout:.0f} с. "
            "Попробуйте «Слабее» или пришлите кадр меньшего размера.",
        )
        return

    out_name, fmt = _output_name(source.file_name)
    # Документ — с исходным ICC-профилем и EXIF; превью — в sRGB, потому что
    # Telegram пережимает фото и выбрасывает профиль.
    encoded, preview, comparison = await asyncio.to_thread(
        lambda: (
            encode_image(result.image, fmt, meta=meta),
            encode_image(to_srgb(fit(result.image, PREVIEW_MAX_SIDE), meta), "jpg", 90),
            encode_image(to_srgb(make_comparison(image, result.image), meta), "jpg", 88),
        )
    )
    if len(encoded) > UPLOAD_LIMIT and fmt == "png":
        # Бот может отправить файл до 50 МБ; большой PNG пересохраняем в JPEG.
        out_name = PurePath(out_name).with_suffix(".jpg").name
        encoded = await asyncio.to_thread(lambda: encode_image(result.image, "jpg", 95, meta=meta))
    report = _format_report(result, time.perf_counter() - started, downscaled, mode, level, source.compressed)

    await original.reply_media_group(
        media=[
            InputMediaPhoto(media=BufferedInputFile(preview, "preview.jpg"), caption="Превью"),
            InputMediaPhoto(media=BufferedInputFile(comparison, "compare.jpg"), caption="До / после"),
        ]
    )
    await original.reply_document(
        document=BufferedInputFile(encoded, out_name),
        caption=report,
        reply_markup=strength_keyboard(level) if level is not None else None,
    )
    try:
        await status.delete()
    except TelegramBadRequest:
        pass
