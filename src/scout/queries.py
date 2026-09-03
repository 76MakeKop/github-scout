"""Генератор поисковых запросов: `Intent` → `SearchQuerySet` (QUERIES.md, шаг 2).

Строки собирает код, а не модель: модель, пишущая запрос целиком, ломает
воспроизводимость и регулярно порождает невалидный синтаксис квалификаторов
(`decisions_log.md`). Здесь нет ни сети, ни обращений к LLM — чистая функция
от `Intent` и момента времени.

Момент передаётся параметром `generated_at`, а не берётся из часов внутри:
от него зависит окно `pushed:` семейства `recent`, и без явного входа golden-тест
был бы невоспроизводим.

Двух вещей, нужных шаблонам QUERIES.md, в контракте `Intent` (SCHEMAS.md §2) нет,
поэтому их выводит код — см. `_core_noun` и `_topic_slug`.
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from scout import config
from scout.log import RunLogger
from scout.schemas import Intent, QueryFamily, SearchQuery, SearchQuerySet, SortOrder

# --------------------------------------------------------------------------
# Константы сборки (QUERIES.md, шаг 2)
# --------------------------------------------------------------------------

PER_PAGE = 30
"""Одна страница на запрос: глубже Search API всё равно отдаёт не больше 1000."""

RECENT_WINDOW_DAYS = 540
BROAD_MIN_STARS = 50
BROAD_MIN_STARS_WITHOUT_LANGUAGE = 100
"""Без `language:` невод шире, и порог звёзд поднимается — вырожденный случай QUERIES.md."""

MAX_QUERY_LENGTH = 256  # SCHEMAS.md §3: maxLength поля `q`
MIN_QUERIES = 5  # SCHEMAS.md §3: minItems

CORE_NOUN_WORDS = 2

# Служебные слова английских формулировок: в `broad` они только сужают невод.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)

_SORT_BY_FAMILY = {
    QueryFamily.EXACT: SortOrder.BEST_MATCH,
    QueryFamily.SYNONYM: SortOrder.BEST_MATCH,
    QueryFamily.README: SortOrder.BEST_MATCH,
    QueryFamily.TOPIC: SortOrder.STARS,
    QueryFamily.LIBRARY: SortOrder.STARS,
    QueryFamily.BROAD: SortOrder.STARS,
    QueryFamily.RECENT: SortOrder.UPDATED,
}

_NOT_ARCHIVED = "archived:false"


class QueryGenerationError(ValueError):
    """Из интента не собралось пяти различимых запросов — меньше схема не примет."""


@dataclass(frozen=True)
class _Draft:
    """Запрос до присвоения `id`: часть черновиков отсеется, и нумерация сдвинется."""

    family: QueryFamily
    q: str


# --------------------------------------------------------------------------
# Производные величины, которых нет в контракте `Intent`
# --------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return text.lower().split()


def _core_noun(synonym: str) -> str:
    """Первые два значимых слова главного синонима.

    В `Intent` поля `core_noun` нет, а `broad` по QUERIES.md должен быть шире
    остальных семейств. Меньше слов — шире выдача, поэтому именно усечение,
    а не переформулировка: код не выдумывает морфологию, которой ему неоткуда взять.
    """
    words = _words(synonym)
    meaningful = [word for word in words if word not in _STOPWORDS] or words
    return " ".join(meaningful[:CORE_NOUN_WORDS])


def _topic_slug(value: str) -> str:
    """`Data Extraction` → `data-extraction`. Пусто, если латиницы не осталось."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _quoted(value: str) -> str:
    """Пробел внутри значения квалификатора разорвал бы его на два токена."""
    return f'"{value}"' if " " in value else value


def _join(*parts: str) -> str:
    return " ".join(part for part in parts if part)


# --------------------------------------------------------------------------
# Семейства
# --------------------------------------------------------------------------


def _language_qualifier(intent: Intent) -> str:
    """Пустой, если язык не важен.

    Берётся только первый язык: квалификаторы GitHub объединяются по И,
    и два `language:` дали бы гарантированно пустую выдачу.
    """
    if not intent.languages:
        return ""
    return f"language:{_quoted(intent.languages[0].strip().lower())}"


def _synonym(intent: Intent, index: int) -> str:
    if index >= len(intent.synonyms):
        return ""
    return intent.synonyms[index].strip()


def _library_clause(intent: Intent) -> str:
    """OR-список гипотез имён, укороченный с хвоста до влезающего в лимит.

    Гипотезы идут в порядке убывания уверенности модели, поэтому отбрасывается хвост.
    """
    names = [_quoted(name.strip()) for name in intent.known_libraries if name.strip()]
    while names:
        clause = _join(" OR ".join(names), "in:name,description")
        if len(clause) <= MAX_QUERY_LENGTH:
            return clause
        names.pop()
    return ""


def _drafts(intent: Intent, today: date) -> list[_Draft]:
    """Семь черновиков в порядке примера QUERIES.md. Пустая строка `q` = семейство отпало."""
    language = _language_qualifier(intent)
    topics = [slug for slug in (_topic_slug(item) for item in intent.domain[:2]) if slug]
    pushed_since = today - timedelta(days=RECENT_WINDOW_DAYS)
    stars = BROAD_MIN_STARS if language else BROAD_MIN_STARS_WITHOUT_LANGUAGE

    first, second, third = (_synonym(intent, i) for i in range(3))

    drafts = [
        (QueryFamily.EXACT, _join(first, language, _NOT_ARCHIVED) if first else ""),
        (QueryFamily.SYNONYM, _join(second, language, _NOT_ARCHIVED) if second else ""),
        (
            QueryFamily.TOPIC,
            _join(*(f"topic:{slug}" for slug in topics), _NOT_ARCHIVED) if topics else "",
        ),
        (QueryFamily.LIBRARY, _library_clause(intent)),
        (
            QueryFamily.README,
            _join(f'"{third}"', "in:readme", language, _NOT_ARCHIVED) if third else "",
        ),
        (
            QueryFamily.BROAD,
            _join(_core_noun(first), language, f"stars:>{stars}", _NOT_ARCHIVED) if first else "",
        ),
        (
            QueryFamily.RECENT,
            _join(first, language, f"pushed:>{pushed_since.isoformat()}", _NOT_ARCHIVED)
            if first
            else "",
        ),
    ]
    return [_Draft(family, q) for family, q in drafts if q]


def _broad_fallback(intent: Intent) -> _Draft | None:
    """Тот же невод без `language:` и без `stars:` — добор, когда запросов меньше пяти.

    Это ровно та «широкая» переформулировка, которую QUERIES.md предписывает
    в вырожденных случаях; здесь она включается не по бедности выдачи, а по
    бедности самого интента.
    """
    first = _synonym(intent, 0)
    if not first:
        return None
    return _Draft(QueryFamily.BROAD, _join(_core_noun(first), _NOT_ARCHIVED))


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


def build_query_set(
    intent: Intent,
    *,
    generated_at: datetime | None = None,
    logger: RunLogger | None = None,
) -> SearchQuerySet:
    """5–10 запросов из интента. Одинаковые строки схлопываются, слишком длинные отбрасываются."""
    generated_at = generated_at or datetime.now(UTC)

    kept: list[_Draft] = []
    seen: set[str] = set()

    def offer(draft: _Draft | None) -> None:
        if draft is None or len(kept) >= config.MAX_SEARCH_QUERIES:
            return
        if len(draft.q) > MAX_QUERY_LENGTH:
            if logger:
                logger.info("query_dropped", family=draft.family.value, reason="too_long")
            return
        if draft.q in seen:
            if logger:
                logger.info("query_dropped", family=draft.family.value, reason="duplicate")
            return
        seen.add(draft.q)
        kept.append(draft)

    for draft in _drafts(intent, generated_at.date()):
        offer(draft)

    if len(kept) < MIN_QUERIES:
        offer(_broad_fallback(intent))

    if len(kept) < MIN_QUERIES:
        raise QueryGenerationError(
            f"из интента собралось {len(kept)} различимых запросов, схема требует {MIN_QUERIES}"
        )

    return SearchQuerySet(
        request_id=intent.request_id,
        generated_at=generated_at,
        generator_version=config.QUERY_GENERATOR_VERSION,
        queries=[
            SearchQuery(
                id=f"q{number}",
                family=draft.family,
                q=draft.q,
                sort=_SORT_BY_FAMILY[draft.family],
                per_page=PER_PAGE,
            )
            for number, draft in enumerate(kept, start=1)
        ],
    )
