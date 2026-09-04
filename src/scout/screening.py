"""Слой 1: скрининг кандидатов на V4-Flash (SCHEMAS.md §5, ARCHITECTURE.md).

Вход — метаданные `Candidate` и первые 4000 символов README. Задача слоя —
отсечь очевидно чужое дёшево, до дорогого аудита. Отсечение по лицензии
и по звёздам запрещено (CLAUDE.md, запрет 4) и не поддержано ни кодом, ни промптом.

Один вызов на кандидата, а не один на всех: системный промпт при этом идёт
префиксом каждого сообщения и попадает в cache hit, а невалидный ответ по одному
репозиторию стоит одного повтора, а не пересборки всего списка.

Кандидаты разбираются пулом из `SCREENING_WORKERS` потоков. Последовательный
разбор давал ~16 с на репозиторий и ~13 минут на полсотни — против цели
«полный цикл ≤ 5 минут» (`CLAUDE.md`). Порядок результатов от этого не зависит:
`map` отдаёт их в порядке входа, а `passed` всё равно пересчитывается сортировкой.

Код не доверяет модели там, где решение принадлежит коду:

- `repo_id`, `full_name`, `evidence` проставляются из `Candidate`, что бы модель
  ни прислала;
- `passed` **пересчитывается** из `results`, а не берётся из ответа: список,
  который решает, кто поедет на дорогой слой, не должен зависеть от того,
  сумела ли модель отсортировать десять чисел.

Подсчёта стоимости здесь нет — по ROADMAP.md это день 7; `cost_usd` пока 0.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from scout import config
from scout.config import MissingCredential
from scout.cost import token_usage
from scout.deepseek import DeepSeekAuth, DeepSeekClient
from scout.github import GitHubAuth, GitHubClient, GitHubError
from scout.log import RunLogger
from scout.schemas import (
    Candidate,
    Intent,
    ModelName,
    Provenance,
    ProvenanceApi,
    ScreeningItem,
    ScreeningResult,
    ScreeningVerdict,
)

MAX_README_CHARS = 4000
"""ARCHITECTURE.md: Слой 1 видит метаданные и первые 4000 символов README."""

MAX_ATTEMPTS = 2  # первая попытка + один повтор с текстом ошибки
README_PATH = "README.md"

SCREENING_WORKERS = 5
"""Одновременных разборов кандидата. Больше пяти брать незачем: упор придёт не
в DeepSeek, а в `core`-лимит GitHub на чтение README, и рост параллелизма начнёт
грозить вторичным лимитом (`decisions_log.md`, 2026-09-04)."""

# Эти три поля модель не заполняет — их проставляет код.
_CODE_OWNED_FIELDS = ("repo_id", "full_name", "evidence")


@dataclass
class ScreeningRun:
    """Результат слоя. `failed` — кандидаты, чей ответ дважды не прошёл схему.

    Места для них в `ScreeningResult` (SCHEMAS.md §5) нет: контракт описывает
    разобранные результаты. В `passed` они не попадают, и по ARCHITECTURE.md
    в отчёт тоже не попадут.
    """

    result: ScreeningResult
    failed: list[str] = field(default_factory=list)


def load_system_prompt() -> str:
    return config.prompt_path("l1").read_text(encoding="utf-8")


def _format_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]


def _readme(
    github: GitHubClient,
    candidate: Candidate,
    *,
    logger: RunLogger | None,
) -> str:
    """Первые 4000 символов README на зафиксированной ревизии кандидата.

    Читаем по `head_sha`, а не по подвижной ветке: иначе скрининг и последующий
    аудит смотрели бы на разный код при том же ключе кэша.

    README нет — не ошибка: скринингу остаются метаданные. Отсутствие описания
    само по себе не повод отсеивать (запрет 4 и промпт `l1-1`).
    """
    try:
        text = github.get_file(candidate.full_name, README_PATH, ref=candidate.head_sha)
    except GitHubAuth:
        raise
    except GitHubError as exc:
        text = None
        if logger:
            logger.info("readme_failed", full_name=candidate.full_name, detail=str(exc))

    if not text:
        if logger:
            logger.info("readme_missing", full_name=candidate.full_name, path=README_PATH)
        return ""

    return text[:MAX_README_CHARS]


def _user_message(intent: Intent, candidate: Candidate, readme: str) -> str:
    """Задача, метаданные, README. Порядок постоянный — как и системный префикс."""
    metadata = [
        f"full_name: {candidate.full_name}",
        f"description: {candidate.description or '—'}",
        f"language: {candidate.language or '—'}",
        f"topics: {', '.join(candidate.topics) or '—'}",
        f"stars: {candidate.stars}",
        f"pushed_at: {candidate.pushed_at.date().isoformat()}",
        f"is_fork: {'да' if candidate.is_fork else 'нет'}",
    ]

    return "\n".join(
        [
            "## Задача пользователя",
            intent.task,
            f"Обязательно: {'; '.join(intent.must_have) or '—'}",
            f"Желательно: {'; '.join(intent.nice_to_have) or '—'}",
            f"Исключить: {'; '.join(intent.exclude) or '—'}",
            f"Языки: {', '.join(intent.languages) or 'не важно'}",
            "",
            "## Кандидат",
            *metadata,
            "",
            "## README (начало)",
            readme or "README отсутствует — оценивай по метаданным.",
        ]
    )


def _evidence(
    candidate: Candidate, *, has_readme: bool, retrieved_at: datetime
) -> list[Provenance]:
    """Provenance проставляет код: дозапись provenance — его автономное право."""
    evidence = [
        Provenance(
            path="metadata",
            commit_sha=candidate.head_sha,
            retrieved_at=candidate.retrieved_at,
            api=ProvenanceApi.SEARCH,
        )
    ]
    if has_readme:
        evidence.append(
            Provenance(
                path=README_PATH,
                commit_sha=candidate.head_sha,
                retrieved_at=retrieved_at,
                api=ProvenanceApi.CONTENTS,
            )
        )
    return evidence


def _assemble(
    payload: dict[str, Any],
    candidate: Candidate,
    *,
    has_readme: bool,
    retrieved_at: datetime,
) -> ScreeningItem:
    body = {key: value for key, value in payload.items() if key not in _CODE_OWNED_FIELDS}
    return ScreeningItem(
        **body,
        repo_id=candidate.repo_id,
        full_name=candidate.full_name,
        evidence=_evidence(candidate, has_readme=has_readme, retrieved_at=retrieved_at),
    )


def _screen_one(
    candidate: Candidate,
    *,
    intent: Intent,
    system: str,
    client: DeepSeekClient,
    github: GitHubClient,
    logger: RunLogger | None,
) -> tuple[ScreeningItem | None, dict[str, int]]:
    """Разбор одного кандидата, из которого исключение наружу не выходит.

    Слой 1 идёт пулом потоков, и `map` поднимает исключение потока в вызывающем
    коде: любая неожиданная поломка на одном кандидате из пятидесяти уронила бы
    весь скан. Цена дня 8 — «сбой одного кандидата стоит одного кандидата»,
    поэтому здесь ловится всё, а не только предвиденное.

    Два исключения не глушатся намеренно: отсутствующий и отклонённый ключ.
    Они одинаковы для всех пятидесяти кандидатов, и молчаливый провал каждого
    выглядел бы как «модель не справилась», хотя чинить надо `.env`.
    """
    totals: dict[str, int] = {}
    try:
        return _screen_candidate(
            candidate,
            intent=intent,
            system=system,
            client=client,
            github=github,
            logger=logger,
            totals=totals,
        )
    except (MissingCredential, DeepSeekAuth, GitHubAuth):
        raise
    except Exception as exc:  # ловим всё: живучесть пула важнее точности типа
        if logger:
            logger.error(
                "screening_crash",
                full_name=candidate.full_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        return None, totals


def _screen_candidate(
    candidate: Candidate,
    *,
    intent: Intent,
    system: str,
    client: DeepSeekClient,
    github: GitHubClient,
    logger: RunLogger | None,
    totals: dict[str, int],
) -> tuple[ScreeningItem | None, dict[str, int]]:
    """Один кандидат: README, вызов модели, один повтор при невалидном ответе.

    Счётчики токенов копятся в переданном словаре, а не в общем на все потоки:
    функция работает в потоке пула, и разделяемый словарь пришлось бы держать
    под замком. Словарь приходит снаружи, чтобы токены, потраченные до поломки,
    не терялись вместе с кандидатом — они уже оплачены.
    """
    retrieved_at = datetime.now(UTC)
    readme = _readme(github, candidate, logger=logger)
    user = _user_message(intent, candidate, readme)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        payload, counters = client.chat_json(
            system=system,
            user=user,
            model=ModelName.FLASH.value,
            temperature=0.0,
        )
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + value

        try:
            item = _assemble(payload, candidate, has_readme=bool(readme), retrieved_at=retrieved_at)
            return item, totals
        except ValidationError as exc:
            problems = _format_errors(exc)
            if logger:
                logger.info(
                    "screening_invalid",
                    full_name=candidate.full_name,
                    attempt=attempt,
                    problems=problems,
                )
            if attempt == MAX_ATTEMPTS:
                if logger:
                    logger.error(
                        "screening_failed", full_name=candidate.full_name, problems=problems
                    )
                return None, totals
            user = (
                f"{user}\n\n"
                "Предыдущий ответ не прошёл валидацию схемы:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\nВерни исправленный JSON."
            )

    return None, totals


def _passed(results: Sequence[ScreeningItem], limit: int) -> list[int]:
    """`repo_id` прошедших, по убыванию relevance. Считает код, а не модель.

    Второй ключ сортировки — `repo_id`: при равном relevance порядок обязан быть
    воспроизводимым, иначе на дорогой слой при повторе скана поедет другая десятка.
    """
    passing = [item for item in results if item.verdict is ScreeningVerdict.PASS]
    passing.sort(key=lambda item: (-item.relevance, item.repo_id))
    return [item.repo_id for item in passing[:limit]]


def screen(
    candidates: Sequence[Candidate],
    intent: Intent,
    *,
    request_id: UUID,
    github: GitHubClient,
    client: DeepSeekClient | None = None,
    logger: RunLogger | None = None,
    limit: int = config.MAX_AUDIT_CANDIDATES,
    workers: int = SCREENING_WORKERS,
) -> ScreeningRun:
    """Кандидаты → `ScreeningResult` с пересчитанным кодом списком `passed`."""
    client = client or DeepSeekClient(logger=logger)
    system = load_system_prompt()
    started_at = datetime.now(UTC)

    results: list[ScreeningItem] = []
    failed: list[str] = []
    totals: dict[str, int] = {}

    def work(candidate: Candidate) -> tuple[ScreeningItem | None, dict[str, int]]:
        return _screen_one(
            candidate,
            intent=intent,
            system=system,
            client=client,
            github=github,
            logger=logger,
        )

    # `map` отдаёт результаты в порядке входа, а не завершения: кандидаты
    # разбираются одновременно, но список остаётся тем же при любом раскладе
    # задержек. Разбор идёт в потоках пула, склейка — здесь, в одном.
    pool_size = max(1, min(workers, len(candidates) or 1))
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="l1") as pool:
        outcomes = list(pool.map(work, candidates))

    for candidate, (item, counters) in zip(candidates, outcomes, strict=True):
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + value

        if item is None:
            failed.append(candidate.full_name)
            continue

        if item.verdict is ScreeningVerdict.REJECT and logger:
            # Без этой строки в логе виден только `passed`, и на разборе recall
            # (день 10) нельзя отличить «эталон не нашёлся поиском» от
            # «нашёлся, но Слой 1 его отбросил» — а это разные починки.
            logger.info(
                "candidate_rejected",
                repo_id=item.repo_id,
                full_name=item.full_name,
                relevance=item.relevance,
                verdict=item.verdict.value,
                red_flags=[flag.value for flag in item.red_flags],
                reasons=item.reasons,
            )

        results.append(item)

    result = ScreeningResult(
        request_id=request_id,
        layer=1,
        model=ModelName.FLASH,
        prompt_version=config.PROMPT_VERSIONS["l1"],
        results=results,
        passed=_passed(results, limit),
        # Окно тарификации берётся по началу слоя, а не по каждому вызову:
        # скрининг длится минуты и мог бы пересечь границу peak-часа посередине.
        # Дробить счёт по вызовам ради этого незачем — разница меньше цены одного репо.
        token_usage=token_usage(ModelName.FLASH, totals, moment=started_at),
    )

    return ScreeningRun(result=result, failed=failed)
