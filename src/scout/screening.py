"""Слой 1: скрининг кандидатов на V4-Flash (SCHEMAS.md §5, ARCHITECTURE.md).

Вход — метаданные `Candidate` и первые 4000 символов README. Задача слоя —
отсечь очевидно чужое дёшево, до дорогого аудита. Отсечение по лицензии
и по звёздам запрещено (CLAUDE.md, запрет 4) и не поддержано ни кодом, ни промптом.

Один вызов на кандидата, а не один на всех: системный промпт при этом идёт
префиксом каждого сообщения и попадает в cache hit, а невалидный ответ по одному
репозиторию стоит одного повтора, а не пересборки всего списка.

Код не доверяет модели там, где решение принадлежит коду:

- `repo_id`, `full_name`, `evidence` проставляются из `Candidate`, что бы модель
  ни прислала;
- `passed` **пересчитывается** из `results`, а не берётся из ответа: список,
  который решает, кто поедет на дорогой слой, не должен зависеть от того,
  сумела ли модель отсортировать десять чисел.

Подсчёта стоимости здесь нет — по ROADMAP.md это день 7; `cost_usd` пока 0.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from scout import config
from scout.deepseek import DeepSeekClient
from scout.github import GitHubClient, GitHubError
from scout.log import RunLogger
from scout.schemas import (
    Candidate,
    Intent,
    ModelName,
    PricingWindow,
    Provenance,
    ProvenanceApi,
    ScreeningItem,
    ScreeningResult,
    ScreeningVerdict,
    TokenUsage,
)

MAX_README_CHARS = 4000
"""ARCHITECTURE.md: Слой 1 видит метаданные и первые 4000 символов README."""

MAX_ATTEMPTS = 2  # первая попытка + один повтор с текстом ошибки
README_PATH = "README.md"

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
    totals: dict[str, int],
    logger: RunLogger | None,
) -> ScreeningItem | None:
    """Один кандидат: README, вызов модели, один повтор при невалидном ответе."""
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
            return _assemble(payload, candidate, has_readme=bool(readme), retrieved_at=retrieved_at)
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
                return None
            user = (
                f"{user}\n\n"
                "Предыдущий ответ не прошёл валидацию схемы:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\nВерни исправленный JSON."
            )

    return None


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
) -> ScreeningRun:
    """Кандидаты → `ScreeningResult` с пересчитанным кодом списком `passed`."""
    client = client or DeepSeekClient(logger=logger)
    system = load_system_prompt()

    results: list[ScreeningItem] = []
    failed: list[str] = []
    totals: dict[str, int] = {}

    for candidate in candidates:
        item = _screen_one(
            candidate,
            intent=intent,
            system=system,
            client=client,
            github=github,
            totals=totals,
            logger=logger,
        )
        if item is None:
            failed.append(candidate.full_name)
            continue
        results.append(item)

    result = ScreeningResult(
        request_id=request_id,
        layer=1,
        model=ModelName.FLASH,
        prompt_version=config.PROMPT_VERSIONS["l1"],
        results=results,
        passed=_passed(results, limit),
        token_usage=TokenUsage(
            model=ModelName.FLASH,
            input_tokens=totals.get("input_tokens", 0),
            cached_input_tokens=totals.get("cached_input_tokens", 0),
            output_tokens=totals.get("output_tokens", 0),
            # Подсчёт стоимости — день 7 ROADMAP.md. Поле обязательно по схеме,
            # поэтому стоит ноль, а не выдуманное число.
            cost_usd=0.0,
            pricing_window=PricingWindow.PEAK if config.is_peak() else PricingWindow.OFF_PEAK,
        ),
    )

    return ScreeningRun(result=result, failed=failed)
