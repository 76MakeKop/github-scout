"""Конфигурация: ключи из окружения и константы лимитов.

Ключи читаются **лениво** — только в момент реального обращения к API.
Отсутствие ключей не мешает CLI дойти до заглушки конвейера.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

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
# Peak-часы DeepSeek (CLAUDE.md → «Peak-часы»)
# --------------------------------------------------------------------------

PEAK_WINDOWS_UTC = ((1, 4), (6, 10))


def is_peak(moment: datetime | None = None) -> bool:
    """Peak: 01:00–04:00 и 06:00–10:00 UTC. Считается строго по UTC."""
    moment = moment or datetime.now(UTC)
    hour = moment.astimezone(UTC).hour
    return any(start <= hour < end for start, end in PEAK_WINDOWS_UTC)


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


def qwen_api_key() -> str | None:
    """Необязателен: нужен только для fallback Слоя 2."""
    return os.environ.get("QWEN_API_KEY") or None
