"""Дедуп и переранжирование: сырая выдача GitHub → `Candidate[]` (QUERIES.md, шаги 3–4).

Ни сети, ни моделей: чистая функция от выдачи и момента времени. Момент передаётся
параметром `now` по той же причине, что и `generated_at` в генераторе запросов —
от него зависит слагаемое свежести в приоре, и без явного входа тест приора
переставал бы проходить со временем.

Веса RRF и приора — гиперпараметры из QUERIES.md, шаг 4. Менять их разрешено
только после прогона golden-set, автоматическая подстройка запрещена
(CLAUDE.md, запрет 7), поэтому здесь они константы модуля, а не аргументы.

Чего в выдаче Search API нет и что поэтому приходит извне:

- `parent.id` — в элементах поиска поля `parent` не бывает, оно есть только
  в ответе `/repos/{full_name}`. Схлопывание форков поэтому двухступенчатое:
  по `parent.id`, если он всё же известен, иначе по совпадению имени с более
  популярным не-форком (см. `_fork_targets`).
- `head_sha` — головы ветки поиск не отдаёт, а `Candidate` без неё невалиден.
  Её подставляет вызывающий код в `to_candidate`: один вызов `/repos/.../commits/{branch}`
  на кандидата, и делать его имеет смысл только для того среза, который поедет
  на Слой 1, а не для всех двух сотен результатов поиска.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from scout import config
from scout.log import RunLogger
from scout.schemas import Candidate

# --------------------------------------------------------------------------
# Константы переранжирования (QUERIES.md, шаг 4)
# --------------------------------------------------------------------------

K_RRF = 60
"""Общепринятое значение: гасит разницу между 1-м и 2-м местом и не даёт
одному запросу продавить в топ весь свой список."""

RRF_WEIGHT = 0.70
PRIOR_WEIGHT = 0.30

PRIOR_STARS_WEIGHT = 0.40
PRIOR_FRESHNESS_WEIGHT = 0.30
PRIOR_LICENSE_WEIGHT = 0.15
PRIOR_NOT_FORK_WEIGHT = 0.15

STARS_LOG_CEILING = 4.0
"""log10(stars + 1) / 4: потолок слагаемого — 10 000 звёзд."""

FRESHNESS_WINDOW_DAYS = 540.0

MAX_DESCRIPTION = 500  # SCHEMAS.md §4: maxLength
MAX_TOPICS = 20  # SCHEMAS.md §4: maxItems

UNDETERMINED_LICENSE = "NOASSERTION"
"""GitHub так помечает найденный, но не опознанный файл лицензии: определённой
лицензии у репозитория нет, и слагаемое приора за неё не начисляется."""


class RankingError(ValueError):
    """Выдача GitHub не складывается в `Candidate` — поломка контракта, не пустой результат."""


@dataclass(frozen=True)
class RankedRepo:
    """Репозиторий после дедупа и скоринга, до сборки в контракт `Candidate`.

    `rrf_raw` хранится рядом с нормализованным `rrf_score`: нормализация делит
    на максимум по выдаче, и без исходной суммы вклад отдельного запроса
    в логах уже не восстановить.
    """

    repo: dict[str, Any]
    found_by: tuple[str, ...]
    rrf_raw: float
    rrf_score: float
    prior_score: float
    score: float
    rank: int


@dataclass
class _Merged:
    """Один репозиторий и его позиции в выдачах разных запросов."""

    repo: dict[str, Any]
    ranks: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Чтение полей выдачи
# --------------------------------------------------------------------------


def _stars(repo: Mapping[str, Any]) -> int:
    return max(int(repo.get("stargazers_count") or 0), 0)


def _short_name(repo: Mapping[str, Any]) -> str:
    """Имя репозитория без владельца: `camelot-dev/camelot` → `camelot`."""
    name = repo.get("name")
    if isinstance(name, str) and name:
        return name.lower()
    full_name = repo.get("full_name")
    return full_name.rsplit("/", 1)[-1].lower() if isinstance(full_name, str) else ""


def _parent_id(repo: Mapping[str, Any]) -> int | None:
    parent = repo.get("parent")
    if isinstance(parent, Mapping) and isinstance(parent.get("id"), int):
        return parent["id"]
    return None


def _license_spdx(repo: Mapping[str, Any]) -> str | None:
    license_ = repo.get("license")
    if not isinstance(license_, Mapping):
        return None
    spdx = license_.get("spdx_id")
    if not isinstance(spdx, str) or not spdx or spdx == UNDETERMINED_LICENSE:
        return None
    return spdx


def _moment(repo: Mapping[str, Any], key: str) -> datetime | None:
    value = repo.get(key)
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _query_number(query_id: str) -> tuple[int, str]:
    """Ключ сортировки `found_by`: q10 после q2, а не между q1 и q2."""
    digits = query_id[1:]
    return (int(digits), query_id) if digits.isdigit() else (10**6, query_id)


# --------------------------------------------------------------------------
# Шаг 3. Дедупликация
# --------------------------------------------------------------------------


def _merge(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    logger: RunLogger | None,
) -> dict[int, _Merged]:
    """Ключ — `repo_id`: репозиторий могли переименовать между запросами."""
    merged: dict[int, _Merged] = {}

    for query_id, items in results.items():
        for position, item in enumerate(items, start=1):
            repo_id = item.get("id")
            if not isinstance(repo_id, int):
                if logger:
                    logger.info("search_item_skipped", query_id=query_id, reason="no_repo_id")
                continue
            entry = merged.get(repo_id)
            if entry is None:
                merged[repo_id] = _Merged(dict(item), {query_id: position})
            else:
                entry.ranks.setdefault(query_id, position)

    return merged


def _fork_targets(merged: Mapping[int, _Merged]) -> dict[int, int]:
    """Форк → репозиторий, к которому он схлопывается.

    Родитель по `parent.id` — истина; если он известен, но в выдаче его нет,
    форк остаётся сам собой (QUERIES.md, шаг 3). Эвристика по имени включается
    только когда `parent` вообще нет в данных — то есть почти всегда, потому что
    выдача поиска этого поля не содержит. Ограничение «звёзд не меньше, чем
    у форка» отсекает случай однофамильцев: форк популярнее родителя — редкость,
    а два несвязанных проекта с одинаковым именем — нет.
    """
    non_forks_by_name: dict[str, list[int]] = {}
    for repo_id, entry in merged.items():
        if not entry.repo.get("fork"):
            non_forks_by_name.setdefault(_short_name(entry.repo), []).append(repo_id)

    targets: dict[int, int] = {}
    for repo_id, entry in merged.items():
        if not entry.repo.get("fork"):
            continue

        parent_id = _parent_id(entry.repo)
        if parent_id is not None:
            if parent_id in merged and parent_id != repo_id:
                targets[repo_id] = parent_id
            continue

        stars = _stars(entry.repo)
        siblings = [
            other
            for other in non_forks_by_name.get(_short_name(entry.repo), [])
            if other != repo_id and _stars(merged[other].repo) >= stars
        ]
        if siblings:
            targets[repo_id] = min(siblings)

    return targets


def _resolve(start: int, targets: Mapping[int, int]) -> int:
    """Конец цепочки форк → форк → родитель. Цикл в данных не зациклит нас."""
    seen = {start}
    node = start
    while (nxt := targets.get(node)) is not None and nxt not in seen:
        seen.add(nxt)
        node = nxt
    return node


def _collapse_forks(merged: dict[int, _Merged], logger: RunLogger | None) -> None:
    """Форки схлопываются к родителю, отдавая ему свои позиции в выдачах.

    Позиции переходят по минимуму: репозиторий, найденный запросом через форк,
    этим запросом всё-таки найден, и вклад в RRF терять незачем.
    """
    targets = _fork_targets(merged)
    for repo_id, target in targets.items():
        final = _resolve(target, targets)
        if final == repo_id or final not in merged or repo_id not in merged:
            continue

        for query_id, position in merged[repo_id].ranks.items():
            current = merged[final].ranks.get(query_id)
            merged[final].ranks[query_id] = position if current is None else min(current, position)

        if logger:
            logger.info(
                "fork_collapsed",
                fork=merged[repo_id].repo.get("full_name"),
                parent=merged[final].repo.get("full_name"),
            )
        del merged[repo_id]


def _rejection_reason(repo: Mapping[str, Any]) -> str | None:
    """Жёсткое отсечение до ранжирования (QUERIES.md, шаг 3, пункт 4)."""
    if repo.get("archived"):
        return "archived"

    description = repo.get("description")
    has_description = isinstance(description, str) and description.strip()
    if not repo.get("language") and not has_description and not repo.get("topics"):
        return "no_metadata"

    if int(repo.get("size") or 0) == 0:
        return "empty"

    return None


# --------------------------------------------------------------------------
# Шаг 4. RRF и метаданный приор
# --------------------------------------------------------------------------


def _rrf(ranks: Mapping[str, int]) -> float:
    return sum(1.0 / (K_RRF + position) for position in ranks.values())


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _prior(repo: Mapping[str, Any], *, now: datetime) -> float:
    """Против свежесозданных пустышек, которые хорошо совпадают текстом.

    Пуш в будущем (часы GitHub впереди наших) и отсутствие `pushed_at`
    обрабатываются одинаково — через clamp, а не исключением: приор не то место,
    где скан должен падать.
    """
    stars = _clamp(math.log10(_stars(repo) + 1) / STARS_LOG_CEILING)

    pushed_at = _moment(repo, "pushed_at")
    if pushed_at is None:
        freshness = 0.0
    else:
        days = (now - pushed_at).total_seconds() / 86400.0
        freshness = _clamp(1.0 - days / FRESHNESS_WINDOW_DAYS)

    licensed = 1.0 if _license_spdx(repo) else 0.0
    not_fork = 0.0 if repo.get("fork") else 1.0

    return (
        PRIOR_STARS_WEIGHT * stars
        + PRIOR_FRESHNESS_WEIGHT * freshness
        + PRIOR_LICENSE_WEIGHT * licensed
        + PRIOR_NOT_FORK_WEIGHT * not_fork
    )


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


def rank_candidates(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now: datetime,
    limit: int = config.MAX_CANDIDATES,
    logger: RunLogger | None = None,
) -> list[RankedRepo]:
    """Выдачи запросов (`q1` → список элементов Search API) → топ-`limit` по score.

    Порядок ровно как в QUERIES.md: дедуп, схлопывание форков, жёсткое отсечение,
    RRF, приор, сортировка по `(score DESC, repo_id ASC)`. Позиции для RRF берутся
    из исходной выдачи и после отсечения не пересчитываются: ранг — это то, каким
    его увидел GitHub, а не то, что осталось после наших фильтров.
    """
    merged = _merge(results, logger)
    _collapse_forks(merged, logger)

    for repo_id in list(merged):
        reason = _rejection_reason(merged[repo_id].repo)
        if reason is not None:
            if logger:
                logger.info(
                    "candidate_dropped",
                    full_name=merged[repo_id].repo.get("full_name"),
                    reason=reason,
                )
            del merged[repo_id]

    if not merged:
        return []

    raw = {repo_id: _rrf(entry.ranks) for repo_id, entry in merged.items()}
    best = max(raw.values())

    scored: list[tuple[float, int, RankedRepo]] = []
    for repo_id, entry in merged.items():
        rrf_score = raw[repo_id] / best
        prior_score = _prior(entry.repo, now=now)
        score = RRF_WEIGHT * rrf_score + PRIOR_WEIGHT * prior_score
        scored.append(
            (
                score,
                repo_id,
                RankedRepo(
                    repo=entry.repo,
                    found_by=tuple(sorted(entry.ranks, key=_query_number)),
                    rrf_raw=raw[repo_id],
                    rrf_score=rrf_score,
                    prior_score=prior_score,
                    score=score,
                    rank=0,
                ),
            )
        )

    scored.sort(key=lambda row: (-row[0], row[1]))

    top = scored[:limit]
    return [
        RankedRepo(
            repo=item.repo,
            found_by=item.found_by,
            rrf_raw=item.rrf_raw,
            rrf_score=item.rrf_score,
            prior_score=item.prior_score,
            score=item.score,
            rank=position,
        )
        for position, (_, _, item) in enumerate(top, start=1)
    ]


def to_candidate(ranked: RankedRepo, *, head_sha: str, retrieved_at: datetime) -> Candidate:
    """`RankedRepo` + голова ветки → контракт `Candidate` (SCHEMAS.md §4).

    Итоговый `score` в контракт не входит — его место занимает `rank`. Веса
    фиксированы, поэтому `score = 0.70 * rrf_score + 0.30 * prior_score`
    восстанавливается из записанных полей; `rrf_score` пишется нормализованным
    именно поэтому.
    """
    repo = ranked.repo

    pushed_at = _moment(repo, "pushed_at")
    if pushed_at is None:
        raise RankingError(f"нет pushed_at у {repo.get('full_name')!r}")

    default_branch = repo.get("default_branch")
    if not isinstance(default_branch, str) or not default_branch:
        raise RankingError(f"нет default_branch у {repo.get('full_name')!r}")

    description = repo.get("description")
    topics = repo.get("topics")

    try:
        return Candidate(
            repo_id=repo["id"],
            full_name=repo["full_name"],
            html_url=repo["html_url"],
            description=description[:MAX_DESCRIPTION] if isinstance(description, str) else None,
            language=repo.get("language"),
            topics=list(topics)[:MAX_TOPICS] if isinstance(topics, list) else [],
            stars=_stars(repo),
            forks=repo.get("forks_count"),
            open_issues=repo.get("open_issues_count"),
            archived=bool(repo.get("archived")),
            is_fork=bool(repo.get("fork")),
            created_at=_moment(repo, "created_at"),
            pushed_at=pushed_at,
            default_branch=default_branch,
            head_sha=head_sha,
            license_spdx=_license_spdx(repo),
            found_by=list(ranked.found_by),
            rrf_score=ranked.rrf_score,
            prior_score=ranked.prior_score,
            rank=ranked.rank,
            retrieved_at=retrieved_at,
        )
    except (KeyError, ValueError) as exc:
        raise RankingError(f"выдача GitHub не складывается в Candidate: {exc}") from exc
