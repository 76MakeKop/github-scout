"""Извлечение интента: валидный ответ, невалидный с повтором, 5xx.

Сети нет — транспорт DeepSeek подменён. Пауз нет — sleep пишет в список.
"""

import json
from uuid import uuid4

import pytest

from scout.config import MissingCredential
from scout.deepseek import DeepSeekClient, DeepSeekUnavailable, strip_code_fence
from scout.http import HttpResponse
from scout.intent import MAX_ATTEMPTS, extract_intent, load_system_prompt

REQUEST_ID = uuid4()

GOOD_PAYLOAD = {
    "task": "извлечение таблиц из PDF в структурированный вид",
    "domain": ["pdf", "data-extraction"],
    "languages": ["python"],
    "must_have": ["сохранение строк и столбцов"],
    "nice_to_have": ["экспорт в CSV"],
    "exclude": ["платные SaaS-обёртки"],
    "synonyms": ["pdf table extraction", "extract tables from pdf", "pdf table parser"],
    "known_libraries": ["camelot", "tabula-py", "pdfplumber"],
}


def chat_response(payload, *, usage=None) -> HttpResponse:
    body = {
        "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
        "usage": usage or {"prompt_tokens": 900, "completion_tokens": 210},
    }
    return HttpResponse(status=200, headers={}, body=json.dumps(body).encode())


class FakeTransport:
    def __init__(self, *responses: HttpResponse) -> None:
        self._responses = list(responses)
        self.bodies: list[dict] = []

    def __call__(self, method, url, headers, body, timeout) -> HttpResponse:
        self.bodies.append(json.loads(body))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def make_client(transport):
    slept: list[float] = []
    return DeepSeekClient(api_key="sk-test", transport=transport, sleep=slept.append), slept


# --- валидный ответ -------------------------------------------------------


def test_valid_json_gives_valid_intent():
    client, _ = make_client(FakeTransport(chat_response(GOOD_PAYLOAD)))

    result = extract_intent("нужен парсер PDF-таблиц", request_id=REQUEST_ID, client=client)

    assert result.status == "ok"
    assert result.attempts == 1
    assert result.intent.synonyms == GOOD_PAYLOAD["synonyms"]
    assert result.intent.known_libraries == GOOD_PAYLOAD["known_libraries"]


def test_code_owns_request_id_model_and_prompt_version():
    """Модель эти поля не заполняет: UUID выдумывать ей незачем."""
    payload = {**GOOD_PAYLOAD, "request_id": "00000000-0000-0000-0000-000000000000"}
    client, _ = make_client(FakeTransport(chat_response(payload)))

    result = extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert result.intent.request_id == REQUEST_ID
    assert result.intent.model == "deepseek-v4-flash"
    assert result.intent.prompt_version == "intent-1"


def test_temperature_is_zero_and_model_is_flash():
    transport = FakeTransport(chat_response(GOOD_PAYLOAD))
    client, _ = make_client(transport)

    extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert transport.bodies[0]["temperature"] == 0.0
    assert transport.bodies[0]["model"] == "deepseek-v4-flash"
    assert transport.bodies[0]["response_format"] == {"type": "json_object"}


def test_usage_counters_are_returned():
    client, _ = make_client(FakeTransport(chat_response(GOOD_PAYLOAD)))

    result = extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert result.usage["input_tokens"] == 900
    assert result.usage["output_tokens"] == 210


# --- невалидный ответ: один повтор, затем failed --------------------------


def test_invalid_payload_retries_once_then_fails():
    """Два синонима вместо трёх — схема не пропустит ни с первого, ни со второго раза."""
    bad = {**GOOD_PAYLOAD, "synonyms": ["pdf table extraction"]}
    transport = FakeTransport(chat_response(bad))
    client, _ = make_client(transport)

    result = extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert result.status == "failed"
    assert result.intent is None
    assert result.attempts == MAX_ATTEMPTS
    assert len(transport.bodies) == 2
    assert any("synonyms" in e for e in result.errors)


def test_retry_message_carries_validation_error():
    bad = {**GOOD_PAYLOAD, "synonyms": []}
    transport = FakeTransport(chat_response(bad))
    client, _ = make_client(transport)

    extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    retry_prompt = transport.bodies[1]["messages"][1]["content"]
    assert "не прошёл валидацию" in retry_prompt
    assert "synonyms" in retry_prompt


def test_retry_succeeds_on_second_attempt():
    bad = {**GOOD_PAYLOAD, "synonyms": ["one"]}
    transport = FakeTransport(chat_response(bad), chat_response(GOOD_PAYLOAD))
    client, _ = make_client(transport)

    result = extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert result.status == "ok"
    assert result.attempts == 2


def test_extra_field_from_model_is_rejected():
    """extra="forbid": выдуманное моделью поле не должно просочиться в контракт."""
    transport = FakeTransport(chat_response({**GOOD_PAYLOAD, "confidence": 0.9}))
    client, _ = make_client(transport)

    result = extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert result.status == "failed"


# --- 5xx ------------------------------------------------------------------


def test_server_error_retries_three_times_then_raises():
    transport = FakeTransport(HttpResponse(status=503, headers={}, body=b"unavailable"))
    client, slept = make_client(transport)

    with pytest.raises(DeepSeekUnavailable):
        extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert len(slept) == 3
    assert slept[0] < slept[1] < slept[2]


def test_429_is_retried():
    transport = FakeTransport(
        HttpResponse(status=429, headers={}, body=b"rate limited"),
        chat_response(GOOD_PAYLOAD),
    )
    client, slept = make_client(transport)

    result = extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    assert result.status == "ok"
    assert len(slept) == 1


# --- ключ и промпт --------------------------------------------------------


def test_missing_key_raises_only_at_call_time(monkeypatch):
    """CLAUDE.md: отсутствие ключа не мешает дойти до места, где он нужен."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient(transport=FakeTransport(chat_response(GOOD_PAYLOAD)))

    with pytest.raises(MissingCredential):
        extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)


def test_system_prompt_is_loaded_and_versioned():
    transport = FakeTransport(chat_response(GOOD_PAYLOAD))
    client, _ = make_client(transport)

    extract_intent("парсер PDF", request_id=REQUEST_ID, client=client)

    system = transport.bodies[0]["messages"][0]["content"]
    assert system == load_system_prompt()
    assert "synonyms" in system


def test_code_fence_is_stripped():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
