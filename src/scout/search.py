"""Стык поиска: `SearchQuerySet` → выдачи GitHub → дедуп и RRF → `Candidate[]`.

Этого шага нет отдельной строкой в `ROADMAP.md` — дыра в плане, а не забытая
работа: между «запросы сгенерированы» (день 3) и «Слой 1 получает `Candidate[]`»
(день 5) кто-то должен выполнить сами запросы. Здесь это и делается.

Модуль ничего не решает про ранжирование (это `rank.py`) и про строки запросов
(это `queries.py`). Его работа — порядок вызовов, поведение при отказах и то,
для какого среза кандидатов спрашивается голова ветки.

Голова ветки спрашивается **после** отбора: `head_sha` нужен каждому `Candidate`
по `SCHEMAS.md` §4, но стоит вызова `core` на репозиторий. До 210 результатов
поиска против 50 отобранных — разница в четыре раза на ровном месте.
"""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from scout import config
from scout.github import (
    GitHubAuth,
    GitHubClient,
    GitHubError,
    RateLimitExhausted,
    SearchQuotaExceeded,
)
from scout.log import RunLogger
from scout.queries import build_fallback_query
from scout.rank import RankedRepo, RankingError, rank_candidates, to_candidate
from scout.schemas import Candidate, DroppedCandidate, DropStage, Intent, SearchQuerySet

MIN_UNIQUE_CANDIDATES = 10
"""Порог вырожденного случая QUERIES.md, шаг 5: ниже него добираем широким неводом."""


@dataclass
class SearchOutcome:
    """Кандидаты и то, при каких условиях они собраны.

    `partial` означает «часть данных потеряна из-за сбоя»: невыполненная выдача,
    отказ GitHub на дозапросе. Такой скан ранжирует то, что успел собрать,
    а отчёт помечается флагом (`Report.partial`, SCHEMAS.md §8).

    `dropped` — про другое: кандидат выбыл штатно, и это не сбой. Репозиторий
    удалён или стал приватным между поиском и аудитом, выдача не складывается
    в `Candidate`. Такие строки едут в отчёт как `dropped`, но `partial`
    не поднимают: результат полон настолько, насколько мир позволил.
    """

    candidates: list[Candidate] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    partial: bool = False
    dropped: list[DroppedCandidate] = field(default_factory=list)


def _event(logger: RunLogger | None, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.info(event, **fields)


def _search(
    github: GitHubClient,
    query: Any,
    *,
    logger: RunLogger | None,
) -> tuple[list[dict[str, Any]] | None, bool]:
    """Одна выдача. Второй элемент — «дальше искать нельзя».

    403/429 на одном запросе — не авария скана: остальные семейства независимы,
    и RRF отработает по тому, что есть. Исчерпанный лимит — другое дело: следующие
    вызовы упрутся в ту же стену, и продолжать значит только тратить время.
    """
    try:
        items = github.search_repositories(query.q, sort=query.sort.value, per_page=query.per_page)
    except GitHubAuth:
        # Отклонённый токен одинаков для всех запросов: глушить его как «одна
        # выдача не пришла» значит вернуть пустой скан вместо «почините .env».
        raise
    except (RateLimitExhausted, SearchQuotaExceeded) as exc:
        _event(logger, "search_stopped", query_id=query.id, detail=str(exc))
        return None, True
    except GitHubError as exc:
        _event(logger, "search_failed", query_id=query.id, detail=str(exc))
        return None, False
    return items, False


def collect_candidates(
    query_set: SearchQuerySet,
    *,
    intent: Intent,
    github: GitHubClient,
    now: datetime | None = None,
    limit: int = config.MAX_CANDIDATES,
    logger: RunLogger | None = None,
) -> SearchOutcome:
    """Выполняет запросы, склеивает выдачи и собирает топ-`limit` в `Candidate[]`."""
    now = now or datetime.now(UTC)
    results: dict[str, list[dict[str, Any]]] = {}
    outcome = SearchOutcome()

    for query in query_set.queries:
        outcome.queries_used.append(query.q)
        items, stop = _search(github, query, logger=logger)
        if items is not None:
            results[query.id] = items
        else:
            outcome.partial = True
        if stop:
            break

    ranked = rank_candidates(results, now=now, limit=limit, logger=logger)

    if len(ranked) < min(MIN_UNIQUE_CANDIDATES, limit) and not outcome.partial:
        ranked = _add_broad_fallback(
            ranked,
            results,
            intent=intent,
            github=github,
            query_count=len(query_set.queries),
            now=now,
            limit=limit,
            outcome=outcome,
            logger=logger,
        )

    _event(
        logger,
        "candidates_ranked",
        unique_found=len(ranked),
        queries_executed=len(results),
        partial=outcome.partial,
        top=[item.repo.get("full_name") for item in ranked[:10]],
    )

    outcome.candidates = _assemble(ranked, github=github, now=now, outcome=outcome, logger=logger)
    return outcome


def _add_broad_fallback(
    ranked: list[RankedRepo],
    results: dict[str, list[dict[str, Any]]],
    *,
    intent: Intent,
    github: GitHubClient,
    query_count: int,
    now: datetime,
    limit: int,
    outcome: SearchOutcome,
    logger: RunLogger | None,
) -> list[RankedRepo]:
    """Добор широким неводом без `language:` и `stars:` (QUERIES.md, шаг 5).

    Добор идёт после ранжирования, а не вместо него: бедной может оказаться
    выдача при вполне богатом интенте — например, когда `language:` отсекает
    половину экосистемы, как это делает `language:javascript` с Node-библиотеками
    на TypeScript.
    """
    fallback = build_fallback_query(intent, query_id=f"q{query_count + 1}")
    if fallback is None or fallback.q in outcome.queries_used:
        return ranked

    _event(logger, "broad_fallback_added", query_id=fallback.id, q=fallback.q, found=len(ranked))
    outcome.queries_used.append(fallback.q)

    items, _ = _search(github, fallback, logger=logger)
    if items is None:
        outcome.partial = True
        return ranked

    results[fallback.id] = items
    return rank_candidates(results, now=now, limit=limit, logger=logger)


def _drop(
    outcome: SearchOutcome,
    logger: RunLogger | None,
    *,
    full_name: str | None,
    reason: str,
    detail: str | None = None,
) -> None:
    """Кандидат выбыл: строка в отчёт (`Report.dropped`) и событие в лог."""
    outcome.dropped.append(
        DroppedCandidate(
            full_name=full_name or "<неизвестный репозиторий>",
            stage=DropStage.SEARCH,
            reason=reason if detail is None else f"{reason}: {detail}"[:200],
        )
    )
    _event(logger, "candidate_dropped", full_name=full_name, reason=reason, detail=detail)


def _assemble(
    ranked: list[RankedRepo],
    *,
    github: GitHubClient,
    now: datetime,
    outcome: SearchOutcome,
    logger: RunLogger | None,
) -> list[Candidate]:
    """Голова ветки на каждого отобранного — и сборка в контракт `Candidate`.

    Выбывший кандидат не оставляет дырки в нумерации: `rank` присваивается заново
    по факту сборки, иначе Слою 1 приехал бы список, чьи ранги описывают не его.

    Отказ GitHub и отсутствие головы ветки выглядят одинаково — кандидата нет, —
    но означают разное. Пустой ответ на `/commits/{branch}`: репозиторий удалён
    или стал приватным между поиском и аудитом, строка таблицы `ARCHITECTURE.md`,
    результат от этого неполным не становится. Ошибка GitHub: репозиторий,
    возможно, жив, а данных мы не получили — вот это `partial`.
    """
    candidates: list[Candidate] = []

    for item in ranked:
        full_name = item.repo.get("full_name")
        branch = item.repo.get("default_branch") or ""

        try:
            head_sha = github.get_head_sha(full_name, branch)
        except GitHubAuth:
            raise
        except GitHubError as exc:
            outcome.partial = True
            _drop(outcome, logger, full_name=full_name, reason="head_sha_error", detail=str(exc))
            continue

        if not head_sha:
            _drop(outcome, logger, full_name=full_name, reason="head_sha_missing")
            continue

        try:
            candidates.append(
                to_candidate(
                    replace(item, rank=len(candidates) + 1),
                    head_sha=head_sha,
                    retrieved_at=now,
                )
            )
        except RankingError as exc:
            _drop(outcome, logger, full_name=full_name, reason="invalid_metadata", detail=str(exc))

    return candidates
