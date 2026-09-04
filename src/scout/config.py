"""Конфигурация: ключи из окружения и константы лимитов.

Ключи читаются **лениво** — только в момент реального обращения к API.
Отсутствие ключей не мешает CLI дойти до заглушки конвейера.

`.env` из корня проекта подхватывается при импорте модуля, до первого чтения
`os.environ`. Своего парсера хватает: формат — `KEY=value`, зависимость ради
пятнадцати строк не нужна.
"""

import os
from collections.abc import MutableMapping
from datetime import UTC, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Загрузка .env (README.md → «Установка»)
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(
    path: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    """Переносит `KEY=value` из файла в окружение. Возвращает число выставленных ключей.

    Уже заданная переменная окружения не перезаписывается: в CI ключи приходят
    из секретов, и забытый локальный `.env` не должен их перебивать.

    Отсутствие файла — не ошибка, а обычный случай: в CI его и не должно быть.
    """
    path = path or DOTENV_PATH
    environ = os.environ if environ is None else environ

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return 0

    loaded = 0
    for line in content.splitlines():
        entry = line.strip().removeprefix("export ").strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        name, _, value = entry.partition("=")
        name = name.strip()
        value = value.strip().strip("\"'")
        if not name or name in environ:
            continue
        environ[name] = value
        loaded += 1
    return loaded


load_dotenv()


# --------------------------------------------------------------------------
# Лимиты конвейера (CLAUDE.md, запрет 5)
# --------------------------------------------------------------------------

MAX_SEARCH_QUERIES = 10
MAX_CANDIDATES = 50
MAX_AUDIT_CANDIDATES = 10
MAX_REPORT_CANDIDATES = 5

# --------------------------------------------------------------------------
# Промпты (CLAUDE.md → «Где живут промпты»)
# --------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_VERSIONS = {
    "intent": "intent-1",
    "l1": "l1-1",
    "l2": "l2-1",
}


def prompt_path(stage: str) -> Path:
    """`intent-1` → `prompts/intent/v1.md`. Версия = имя файла."""
    if stage not in PROMPT_VERSIONS:
        raise KeyError(f"неизвестная стадия промпта: {stage!r}")
    number = PROMPT_VERSIONS[stage].rsplit("-", 1)[1]
    return PROMPTS_DIR / stage / f"v{number}.md"


# --------------------------------------------------------------------------
# Генератор поисковых запросов (QUERIES.md, шаг 2)
# --------------------------------------------------------------------------

QUERY_GENERATOR_VERSION = "qg-1"
"""Версия правил сборки запросов. Меняется вместе с правкой шаблонов в QUERIES.md:
по ней в архиве прогонов видно, каким кодом собран конкретный SearchQuerySet."""


# --------------------------------------------------------------------------
# Peak-часы DeepSeek (CLAUDE.md → «Peak-часы»)
# --------------------------------------------------------------------------

PEAK_WINDOWS_UTC = ((1, 4), (6, 10))

PEAK_WEEKDAYS = frozenset(range(5))
"""Пн–Пт по `datetime.weekday()`. Прайс DeepSeek ограничивает peak буднями,
и без этой проверки выходные считались бы вдвое дороже, чем их выставит
провайдер (`decisions_log.md`, 2026-09-04)."""

# --------------------------------------------------------------------------
# Fallback Слоя 2 (ARCHITECTURE.md → «Обработка ошибок»)
# --------------------------------------------------------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_FALLBACK_MODEL = "qwen/qwen3.8-max"
"""Идентификатор той же модели у OpenRouter. В схемах и отчётах она называется
`qwen3.8-max` (`SCHEMAS.md`, enum `ModelName`) — префикс `qwen/` относится
к маршрутизации OpenRouter, а не к модели. Кода переключения здесь нет:
по ROADMAP.md это день 14."""


def is_peak(moment: datetime | None = None) -> bool:
    """Peak: 01:00–04:00 и 06:00–10:00 UTC по будням. Считается строго по UTC.

    День недели берётся тоже по UTC, а не по местному времени: в Актобе (UTC+5)
    суббота начинается в 19:00 пятницы, и без приведения к UTC граница выходных
    сдвинулась бы на пять часов.
    """
    moment = (moment or datetime.now(UTC)).astimezone(UTC)

    if moment.weekday() not in PEAK_WEEKDAYS:
        return False

    return any(start <= moment.hour < end for start, end in PEAK_WINDOWS_UTC)


# --------------------------------------------------------------------------
# Ключи окружения (таблица в README.md)
# --------------------------------------------------------------------------


class MissingCredential(RuntimeError):
    """Обязательный ключ не найден в окружении."""


def _required(name: str, purpose: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingCredential(f"{name} не задан в окружении ({purpose}). См. README.md")
    return value


def deepseek_api_key() -> str:
    return _required("DEEPSEEK_API_KEY", "интент, Слой 1, Слой 2")


def github_token() -> str:
    return _required("GITHUB_TOKEN", "поиск по GitHub")


def openrouter_api_key() -> str | None:
    """Необязателен: нужен только для fallback Слоя 2 (ROADMAP.md, день 14)."""
    return os.environ.get("OPENROUTER_API_KEY") or None
