"""Тесты webhook и обработчиков. Запросы к Telegram подменены фейковой сессией."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import cv2
import numpy as np
import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import (
    AnswerCallbackQuery,
    DeleteMessage,
    EditMessageText,
    GetFile,
    SendDocument,
    SendMediaGroup,
    SendMessage,
    TelegramMethod,
)
from aiogram.types import Chat, File, Message

from api import webhook
from bot import dispatcher as bot_dispatcher
from bot import handlers
from bot.pipeline import Step
from tests.conftest import add_noise, make_scene

CHAT = {"id": 777, "type": "private"}
USER = {"id": 777, "is_bot": False, "first_name": "Tester"}


class FakeSession(BaseSession):
    """Сессия aiogram, которая ничего не отправляет, а запоминает вызовы."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self.files = files or {}

    def _message(self) -> Message:
        return Message(message_id=len(self.calls) + 100, date=int(time.time()), chat=Chat(**CHAT))

    async def make_request(self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None) -> Any:
        self.calls.append(method)
        if isinstance(method, (SendMessage, SendDocument, EditMessageText)):
            return self._message().as_(bot)
        if isinstance(method, SendMediaGroup):
            return [self._message().as_(bot) for _ in method.media]
        if isinstance(method, GetFile):
            data = self.files[method.file_id]
            return File(file_id=method.file_id, file_unique_id="u", file_size=len(data), file_path=method.file_id)
        if isinstance(method, (AnswerCallbackQuery, DeleteMessage)):
            return True
        return True

    async def stream_content(self, url: str, *args: Any, **kwargs: Any) -> AsyncGenerator[bytes, None]:
        yield self.files[url.rsplit("/", 1)[-1]]

    async def close(self) -> None:
        pass

    def of_type(self, kind: type) -> list[Any]:
        return [c for c in self.calls if isinstance(c, kind)]


@pytest.fixture
def session(env: None, monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    fake = FakeSession()
    monkeypatch.setattr(bot_dispatcher, "create_bot", lambda: Bot("42:TEST", session=fake))
    webhook._recent_updates.clear()
    return fake


async def test_repeated_update_is_processed_once(session: FakeSession) -> None:
    await call_app(body=text_update("/help", update_id=99))
    await call_app(body=text_update("/help", update_id=99))
    assert len(session.of_type(SendMessage)) == 1


async def call_app(
    method: str = "POST", body: dict[str, Any] | None = None, secret: str | None = "s3cret"
) -> tuple[int, bytes]:
    headers = [(b"content-type", b"application/json")]
    if secret is not None:
        headers.append((b"x-telegram-bot-api-secret-token", secret.encode()))
    raw = json.dumps(body or {}).encode()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await webhook.app({"type": "http", "method": method, "headers": headers}, receive, send)
    return sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:])


def text_update(text: str, update_id: int = 1) -> dict[str, Any]:
    entities = [{"type": "bot_command", "offset": 0, "length": len(text)}] if text.startswith("/") else []
    return {
        "update_id": update_id,
        "message": {"message_id": 1, "date": 0, "chat": CHAT, "from": USER, "text": text, "entities": entities},
    }


def document_message(file_id: str, name: str = "photo.jpg", mime: str = "image/jpeg", size: int = 1000) -> dict[str, Any]:
    return {
        "message_id": 5,
        "date": 0,
        "chat": CHAT,
        "from": USER,
        "document": {"file_id": file_id, "file_unique_id": "x", "file_name": name, "mime_type": mime, "file_size": size},
    }


def callback_update(data: str, original: dict[str, Any], update_id: int = 3) -> dict[str, Any]:
    menu = {
        "message_id": 6,
        "date": 0,
        "chat": CHAT,
        "from": {"id": 1, "is_bot": True, "first_name": "bot"},
        "text": "Выберите режим обработки:",
        "reply_to_message": original,
    }
    return {
        "update_id": update_id,
        "callback_query": {"id": "cb1", "from": USER, "chat_instance": "ci", "message": menu, "data": data},
    }


def jpeg_bytes(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


# ---------- webhook ----------


async def test_get_is_health_check(session: FakeSession) -> None:
    status, body = await call_app("GET")
    assert status == 200 and b"alive" in body


@pytest.mark.parametrize("secret", [None, "wrong", ""])
async def test_rejects_bad_secret(session: FakeSession, secret: str | None) -> None:
    status, _ = await call_app(body=text_update("/start"), secret=secret)
    assert status == 403
    assert session.calls == []


async def test_returns_200_on_internal_error(session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    async def explode(_: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(webhook, "handle_update", explode)
    status, _ = await call_app(body=text_update("/start"))
    assert status == 200


async def test_returns_200_on_garbage_body(session: FakeSession) -> None:
    status, _ = await call_app(body={"not": "an update"})
    assert status == 200


# ---------- команды ----------


async def test_start(session: FakeSession) -> None:
    status, _ = await call_app(body=text_update("/start"))
    assert status == 200
    [msg] = session.of_type(SendMessage)
    assert msg.chat_id == 777 and "Привет" in msg.text


async def test_help(session: FakeSession) -> None:
    await call_app(body=text_update("/help"))
    [msg] = session.of_type(SendMessage)
    assert "Режимы" in msg.text


async def test_mode_without_storage(session: FakeSession) -> None:
    await call_app(body=text_update("/mode"))
    [msg] = session.of_type(SendMessage)
    assert "хранилище" in msg.text


async def test_allowed_users_filter(session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.config import get_config

    monkeypatch.setenv("ALLOWED_USER_IDS", "1, 2")
    get_config.cache_clear()
    await call_app(body=text_update("/start"))
    assert session.of_type(SendMessage) == []


# ---------- фото ----------


async def test_photo_document_gets_mode_keyboard(session: FakeSession) -> None:
    await call_app(body={"update_id": 2, "message": document_message("f1")})
    [msg] = session.of_type(SendMessage)
    buttons = [b.text for row in msg.reply_markup.inline_keyboard for b in row]
    assert buttons == ["Авто", "Шум", "Резкость", "Размытие", "Полный конвейер", "Нейросеть ×4", "Нейросеть ×4, детали"]
    assert msg.reply_parameters.message_id == 5


async def test_neural_buttons_hidden_when_disabled(session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.config import get_config

    monkeypatch.setenv("SR_ENABLED", "0")
    get_config.cache_clear()
    await call_app(body={"update_id": 2, "message": document_message("f1")})
    [msg] = session.of_type(SendMessage)
    buttons = [b.text for row in msg.reply_markup.inline_keyboard for b in row]
    assert not any("Нейросеть" in b for b in buttons)


async def test_neural_mode_returns_x4_document(session: FakeSession) -> None:
    img = make_scene(60, 80)
    session.files["n1"] = cv2.imencode(".png", img)[1].tobytes()
    await call_app(body=callback_update("mode:neural_wdn", document_message("n1", "tiny.png", "image/png")))
    [doc] = session.of_type(SendDocument)
    result = cv2.imdecode(np.frombuffer(doc.document.data, np.uint8), cv2.IMREAD_COLOR)
    assert result.shape == (240, 320, 3)
    assert "Real-ESRGAN" in doc.caption and "wdn" in doc.caption


async def test_neural_timeout_sends_classic_result(session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.config import get_config

    monkeypatch.setenv("SR_TIMEOUT", "0.000001")
    get_config.cache_clear()
    img = make_scene(60, 80)
    session.files["n1"] = jpeg_bytes(img)
    await call_app(body=callback_update("mode:neural", document_message("n1")))
    [doc] = session.of_type(SendDocument)
    result = cv2.imdecode(np.frombuffer(doc.document.data, np.uint8), cv2.IMREAD_COLOR)
    assert result.shape == img.shape
    assert "Нейросетевой шаг пропущен" in doc.caption and "классический" in doc.caption


async def test_non_image_document_rejected(session: FakeSession) -> None:
    await call_app(body={"update_id": 2, "message": document_message("f1", "doc.pdf", "application/pdf")})
    [msg] = session.of_type(SendMessage)
    assert "не изображение" in msg.text


async def test_too_big_file(session: FakeSession) -> None:
    await call_app(body={"update_id": 2, "message": document_message("f1", size=25 * 1024 * 1024)})
    [msg] = session.of_type(SendMessage)
    assert "20 МБ" in msg.text


async def test_auto_mode_returns_document(session: FakeSession) -> None:
    img = add_noise(make_scene(), 12)
    session.files["f1"] = jpeg_bytes(img)
    status, _ = await call_app(body=callback_update("mode:auto", document_message("f1", "IMG_1.jpg")))
    assert status == 200

    assert session.of_type(AnswerCallbackQuery)
    [doc] = session.of_type(SendDocument)
    assert doc.document.filename == "IMG_1_enhanced.jpg"
    assert "Время обработки" in doc.caption and "шумоподавление" in doc.caption
    result = cv2.imdecode(np.frombuffer(doc.document.data, np.uint8), cv2.IMREAD_COLOR)
    assert result.shape == img.shape

    [group] = session.of_type(SendMediaGroup)
    assert len(group.media) == 2  # превью и «до / после»
    assert session.of_type(DeleteMessage)  # статус «Обрабатываю…» удалён


async def test_png_stays_png(session: FakeSession) -> None:
    session.files["f2"] = cv2.imencode(".png", make_scene())[1].tobytes()
    await call_app(body=callback_update("mode:sharpen", document_message("f2", "scan.png", "image/png")))
    [doc] = session.of_type(SendDocument)
    assert doc.document.filename == "scan_enhanced.png"


async def test_unreadable_file_reports_error(session: FakeSession) -> None:
    session.files["bad"] = b"definitely not an image"
    await call_app(body=callback_update("mode:auto", document_message("bad")))
    assert session.of_type(SendDocument) == []
    edits = [c.text for c in session.of_type(EditMessageText)]
    assert any("Не получилось прочитать" in t for t in edits)


async def test_failing_step_reported_but_result_sent(session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: np.ndarray) -> np.ndarray:
        raise RuntimeError("сбой шага")

    monkeypatch.setattr(
        handlers,
        "build_steps",
        lambda mode, frames, *_: [Step("x", "тестовый шаг", "image", boom), Step("y", "инверсия", "image", lambda i: 255 - i)],
    )
    session.files["f1"] = jpeg_bytes(make_scene())
    await call_app(body=callback_update("mode:auto", document_message("f1")))
    [doc] = session.of_type(SendDocument)
    assert "тестовый шаг" in doc.caption and "пропущен" in doc.caption


async def test_timeout_reports_failure(session: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot.config import get_config

    monkeypatch.setenv("PROCESS_TIMEOUT", "0.2")
    get_config.cache_clear()

    def slow(img: np.ndarray) -> np.ndarray:
        time.sleep(0.6)
        return img

    monkeypatch.setattr(handlers, "build_steps", lambda mode, frames, *_: [Step("slow", "медленно", "image", slow)])
    session.files["f1"] = jpeg_bytes(make_scene())
    status, _ = await call_app(body=callback_update("mode:auto", document_message("f1")))
    assert status == 200
    assert session.of_type(SendDocument) == []
    assert any("Не успел" in c.text for c in session.of_type(EditMessageText))


async def test_callback_without_original_message(session: FakeSession) -> None:
    update = callback_update("mode:auto", document_message("f1"))
    del update["callback_query"]["message"]["reply_to_message"]
    await call_app(body=update)
    [answer] = session.of_type(AnswerCallbackQuery)
    assert answer.show_alert and "ещё раз" in answer.text
