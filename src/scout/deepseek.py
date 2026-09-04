"""Клиент DeepSeek API.

Ключ читается лениво — в момент вызова, а не при импорте: CLAUDE.md требует,
чтобы отсутствие ключей не мешало дойти до места, где они реально нужны.

Подсчёта стоимости здесь нет: по ROADMAP.md это день 7.
"""

import json
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from scout import config
from scout.http import HttpError, HttpResponse, Transport, urllib_transport
from scout.log import RunLogger

API_URL = "https://api.deepseek.com/chat/completions"
BACKOFF_SECONDS = (1.0, 4.0, 16.0)

LLM_TIMEOUT = 60.0
"""Модели отвечают на порядок медленнее GitHub: у V4-Flash на 1,5k входа уходят
десятки секунд, часть из них — на рассуждения. Общий таймаут HTTP в 20 с
рассчитан на GitHub и для модели означает гарантированный обрыв на полпути."""


class DeepSeekError(RuntimeError):
    """Ответ DeepSeek, который не лечится повтором."""


class DeepSeekUnavailable(DeepSeekError):
    """Сервис не ответил за отведённые попытки — повод для fallback на Qwen."""


class DeepSeekAuth(DeepSeekError):
    """Ключ отклонён (401 или 403) — ошибка конфигурации, а не сбой сервиса.

    Повторять нечего: пока ключ тот же, ответ будет тот же. Живой прогон дня 5
    начинался ровно с этого — ключ с мусорным хвостом давал 401.
    """


def _jitter(base: float) -> float:
    return base * random.uniform(0.8, 1.2)


def _retry_after(response: HttpResponse) -> float | None:
    """Пауза, названная сервером в `Retry-After`. None — заголовка нет."""
    raw = response.header("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def strip_code_fence(text: str) -> str:
    """Модели любят оборачивать JSON в ```json ... ``` даже в режиме json_object."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    without_open = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return without_open.rsplit("```", 1)[0].strip()


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: Transport = urllib_transport,
        logger: RunLogger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._transport = transport
        self._log = logger
        self._sleep = sleep
        self._timeout = timeout
        self.calls = 0
        # Слой 1 держит один клиент на пять потоков: `+=` на счётчике
        # не атомарен, и без замка часть вызовов терялась бы в учёте.
        self._counter_lock = threading.Lock()

    def _key(self) -> str:
        """Ленивое разрешение ключа: без него падаем здесь, а не на старте CLI."""
        return self._api_key or config.deepseek_api_key()

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Возвращает разобранный JSON-ответ и счётчики токенов."""
        payload = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()

        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
            "User-Agent": "github-scout/0.1",
        }

        attempt = 0
        while True:
            try:
                response = self._transport("POST", API_URL, headers, payload, self._timeout)
            except HttpError as exc:
                # ARCHITECTURE.md ставит таймаут в одну строку с 5xx. Без этой
                # ветки обрыв связи уходил наружу голым исключением и ронял скан.
                if attempt >= len(BACKOFF_SECONDS):
                    raise DeepSeekUnavailable(
                        f"DeepSeek не отвечает после {len(BACKOFF_SECONDS)} повторов: {exc}"
                    ) from exc
                delay = _jitter(BACKOFF_SECONDS[attempt])
                attempt += 1
                if self._log:
                    self._log.info(
                        "deepseek_retry",
                        reason="transport",
                        attempt=attempt,
                        wait_seconds=round(delay, 1),
                    )
                self._sleep(delay)
                continue

            with self._counter_lock:
                self.calls += 1

            if response.status == 200:
                break

            if response.status in (401, 403):
                raise DeepSeekAuth(
                    f"DeepSeek отклонил ключ ({response.status}): проверьте DEEPSEEK_API_KEY в .env"
                )

            retryable = response.status == 429 or 500 <= response.status < 600
            if retryable and attempt < len(BACKOFF_SECONDS):
                # Названная сервером пауза важнее нашей лесенки: при 429 DeepSeek
                # знает, когда окно откроется, а мы только гадаем.
                delay = _retry_after(response) or _jitter(BACKOFF_SECONDS[attempt])
                attempt += 1
                if self._log:
                    self._log.info(
                        "deepseek_retry",
                        status=response.status,
                        attempt=attempt,
                        wait_seconds=round(delay, 1),
                    )
                self._sleep(delay)
                continue

            if retryable:
                raise DeepSeekUnavailable(
                    f"DeepSeek отдаёт {response.status} после {len(BACKOFF_SECONDS)} повторов"
                )
            raise DeepSeekError(f"{response.status} от DeepSeek: {response.text()[:200]}")

        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(f"неожиданная форма ответа DeepSeek: {exc}") from exc

        usage = body.get("usage") or {}
        counters = {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "cached_input_tokens": int(
                usage.get("prompt_cache_hit_tokens") or usage.get("prompt_tokens_cached") or 0
            ),
        }

        try:
            parsed = json.loads(strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise DeepSeekError(f"модель вернула не-JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise DeepSeekError(f"ожидался JSON-объект, получен {type(parsed).__name__}")

        return parsed, counters
