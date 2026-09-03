"""Тонкий HTTP-слой поверх стандартной библиотеки.

Новых зависимостей проект не заводит: запросов на один скан порядка пятнадцати,
пул соединений не нужен. Транспорт внедряется через конструктор — тесты
подставляют свой и не ходят в сеть.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT = 20.0


class HttpError(RuntimeError):
    """Транспортная ошибка, которую не имеет смысла разбирать выше."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""

    def header(self, name: str) -> str | None:
        """Заголовки HTTP регистронезависимы."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None

    def json(self) -> Any:
        if not self.body:
            return None
        try:
            return json.loads(self.body)
        except json.JSONDecodeError as exc:
            raise HttpError(f"ответ не является JSON: {exc}") from exc

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], HttpResponse]


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    """Единственное место, где проект реально ходит в сеть.

    Ошибочные статусы не поднимают исключение: заголовки 403/429 нужны
    вызывающему коду, чтобы вычислить паузу до сброса лимита.
    """
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=exc.code,
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=exc.read(),
        )
    except urllib.error.URLError as exc:
        raise HttpError(f"сеть недоступна: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpError(f"таймаут {timeout} с при обращении к {url}") from exc
