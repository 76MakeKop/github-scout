"""Конвейер скана одним вызовом: интент → запросы → поиск → Слой 1.

Вынесен из `cli.py`, потому что читателей у конвейера стало двое. CLI печатает
результат человеку и переводит отказы в коды возврата; `evaluate.py` считает по
тем же прогонам recall и в печати не нуждается вовсе. Пока код жил в CLI,
второму читателю пришлось бы либо разбирать stdout, либо повторять конвейер
своей копией — и та копия неминуемо разошлась бы с первой.

Здесь нет ни печати, ни кодов возврата: наружу выходят объекты и исключения.
Решение, что показать человеку и чем ответить оболочке, принимает `cli.py`
(таблица кодов — в его докстринге).

Слой 2 встал между скринингом и сборкой отчёта на дне 11. Отчёта пока нет:
конвейер отдаёт `AuditResult[]`, а превращение их в `Report` с пятёркой
кандидатов — день 13.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from scout import config
from scout.audit import audit_candidates
from scout.cache import DEFAULT_CACHE_PATH, AuditCache
from scout.cost import token_usage
from scout.deepseek import DeepSeekClient
from scout.github import GitHubClient
from scout.intent import extract_intent
from scout.log import RunLogger
from scout.queries import build_query_set
from scout.schemas import (
    AuditResult,
    Candidate,
    DroppedCandidate,
    DropStage,
    Intent,
    ModelName,
    ScanRequest,
    TokenUsage,
)
from scout.screening import ScreeningRun, screen
from scout.search import collect_candidates

CACHE_PATH = DEFAULT_CACHE_PATH
"""Куда Слой 2 кладёт аудиты. Отдельным именем модуля, а не литералом внутри
функции: тесты обязаны уводить кэш во временный каталог, иначе прогон писал бы
в рабочую базу проекта — тем же приёмом, что `cli.DEFAULT_QUEUE_PATH`."""


class IntentUnparsed(RuntimeError):
    """Модель дважды вернула интент не по схеме — дальше идти не с чем."""


class ScanStatus(StrEnum):
    """Чем закончился конвейер. Все три исхода — законные, ни один не ошибка."""

    OK = "ok"
    NO_CANDIDATES = "no_candidates"
    NONE_PASSED = "none_passed"


@dataclass
class ScanOutcome:
    """Всё, что скан узнал. Читается и печатью, и метрикой.

    `screening` пуст, когда поиск ничего не нашёл: Слою 1 нечего было разбирать.
    """

    request: ScanRequest
    status: ScanStatus
    intent: Intent
    intent_usage: TokenUsage
    queries_used: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    screening: ScreeningRun | None = None
    audits: list[AuditResult] = field(default_factory=list)
    audit_usage: TokenUsage | None = None
    dropped: list[DroppedCandidate] = field(default_factory=list)
    partial: bool = False
    duration_sec: float = 0.0

    @property
    def screening_usage(self) -> TokenUsage | None:
        return self.screening.result.token_usage if self.screening else None

    @property
    def total_cost_usd(self) -> float:
        """Сумма по этапам: интент, Слой 1, Слой 2."""
        screening = self.screening_usage
        return (
            self.intent_usage.cost_usd
            + (screening.cost_usd if screening else 0.0)
            + (self.audit_usage.cost_usd if self.audit_usage else 0.0)
        )

    @property
    def audited_full_names(self) -> list[str]:
        """Прошедшие Слой 2, по убыванию `score.total`.

        Порядок считает код, а не модель, и второй ключ — `repo_id`: при равном
        total два прогона обязаны дать одну и ту же пятёрку в отчёте.
        """
        ordered = sorted(self.audits, key=lambda a: (-a.score.total, a.repo_id))
        return [audit.full_name for audit in ordered]

    @property
    def passed_full_names(self) -> list[str]:
        """Прошедшие Слой 1, в порядке `passed` — то есть по убыванию relevance.

        `passed` хранит `repo_id`, а человеку и метрике нужны имена; сопоставление
        идёт по разобранным результатам скрининга, а не по списку кандидатов:
        кандидат, чей ответ не разобрался, в `passed` попасть и не мог.
        """
        if self.screening is None:
            return []
        by_id = {item.repo_id: item.full_name for item in self.screening.result.results}
        return [by_id[repo_id] for repo_id in self.screening.result.passed if repo_id in by_id]


def _run_audit(
    candidates: Sequence[Candidate],
    *,
    screening: ScreeningRun,
    intent: Intent,
    github: GitHubClient,
    log: RunLogger,
    refresh: bool,
) -> tuple[list[AuditResult], list[str], dict[str, int]]:
    """Слой 2 по прошедшим Слой 1, в порядке `passed` (по убыванию relevance).

    Кэш открывается здесь и на время одного скана: `AuditCache` держит соединение
    SQLite, и оставлять его открытым на весь процесс незачем — скан либо
    завершится, либо упадёт, и в обоих случаях файл должен быть закрыт.

    Сюда же приходит `--refresh` (техдолг 5): флаг разбирался в `ScanOptions`
    и доходил до CLI, но передать его было некуда, пока Слоя 2 не существовало.
    """
    by_id = {candidate.repo_id: candidate for candidate in candidates}
    to_audit = [by_id[repo_id] for repo_id in screening.result.passed if repo_id in by_id]

    log.info("reached_stub", stage="audit", note="Слой 2: аудит прошедших скрининг")

    with AuditCache(CACHE_PATH, refresh=refresh) as cache:
        results, failed, counters = audit_candidates(
            to_audit,
            intent,
            client=DeepSeekClient(logger=log),
            github=github,
            cache=cache,
            logger=log,
        )

    log.info(
        "audit_layer_done",
        audited=len(results),
        failed=len(failed),
        requested=len(to_audit),
        refresh=refresh,
        verdicts={
            verdict.value: sum(1 for a in results if a.verdict is verdict)
            for verdict in {a.verdict for a in results}
        },
    )
    return results, failed, counters


def run_scan(
    request: ScanRequest,
    *,
    log: RunLogger,
    github: GitHubClient | None = None,
    now: datetime | None = None,
) -> ScanOutcome:
    """Один скан целиком. Бросает доменные исключения, не ловит их за вызывающего.

    `github` принимается снаружи, чтобы прогон golden-set мог переиспользовать
    один клиент на все задачи: троттлинг поиска и счётчик запросов живут в нём,
    и новый клиент на каждую задачу означал бы обнуление обоих.
    """
    started = time.monotonic()
    now = now or datetime.now(UTC)

    log.info(
        "start",
        request_id=str(request.request_id),
        query_text=request.query_text,
        options=request.options.model_dump(),
        pricing_window="peak" if config.is_peak(now) else "off-peak",
    )

    extraction = extract_intent(request.query_text, request_id=request.request_id, logger=log)
    if extraction.status == "failed":
        log.error("intent_failed", attempts=extraction.attempts, problems=extraction.errors)
        raise IntentUnparsed("модель дважды вернула интент не по схеме")

    intent = extraction.intent
    intent_usage = token_usage(ModelName.FLASH, extraction.usage, moment=now)
    log.info(
        "intent_extracted",
        synonyms_count=len(intent.synonyms),
        hypothesis_count=len(intent.known_libraries),
        task=intent.task,
        languages=intent.languages,
        attempts=extraction.attempts,
        prompt_version=intent.prompt_version,
        token_usage=intent_usage.model_dump(mode="json"),
    )

    query_set = build_query_set(intent, logger=log)
    log.info(
        "queries_generated",
        query_count=len(query_set.queries),
        generator_version=query_set.generator_version,
        families=[query.family.value for query in query_set.queries],
        queries=[query.q for query in query_set.queries],
    )

    github = github or GitHubClient(token=config.github_token(), logger=log)

    found = collect_candidates(
        query_set,
        intent=intent,
        github=github,
        limit=request.options.max_candidates,
        logger=log,
    )

    if not found.candidates:
        log.info(
            "scan_build_recommended",
            reason="no_candidates",
            queries=found.queries_used,
            partial=found.partial,
        )
        return ScanOutcome(
            request=request,
            status=ScanStatus.NO_CANDIDATES,
            intent=intent,
            intent_usage=intent_usage,
            queries_used=found.queries_used,
            dropped=found.dropped,
            partial=found.partial,
            duration_sec=time.monotonic() - started,
        )

    log.info("reached_stub", stage="screening", note="Слой 1: скрининг кандидатов")

    screening = screen(
        found.candidates,
        intent,
        request_id=request.request_id,
        github=github,
        logger=log,
        limit=request.options.audit_limit,
    )

    dropped = found.dropped + [
        DroppedCandidate(
            full_name=full_name,
            stage=DropStage.SCREENING,
            reason="ответ модели дважды не прошёл схему",
        )
        for full_name in screening.failed
    ]
    partial = found.partial or bool(screening.failed)

    usage = screening.result.token_usage
    log.info(
        "screening_done",
        screened=len(screening.result.results),
        passed=len(screening.result.passed),
        failed=len(screening.failed),
        prompt_version=screening.result.prompt_version,
        token_usage=usage.model_dump(mode="json"),
    )

    audits: list[AuditResult] = []
    audit_usage: TokenUsage | None = None
    if screening.result.passed:
        audits, audit_failed, audit_counters = _run_audit(
            found.candidates,
            screening=screening,
            intent=intent,
            github=github,
            log=log,
            refresh=request.options.refresh,
        )
        if audit_counters:
            audit_usage = token_usage(ModelName.PRO, audit_counters, moment=now)
        if audit_failed:
            partial = True
            dropped += [
                DroppedCandidate(
                    full_name=full_name,
                    stage=DropStage.AUDIT,
                    reason="аудит не удался: ответ модели дважды не прошёл схему",
                )
                for full_name in audit_failed
            ]

    outcome = ScanOutcome(
        request=request,
        status=ScanStatus.OK if screening.result.passed else ScanStatus.NONE_PASSED,
        intent=intent,
        intent_usage=intent_usage,
        queries_used=found.queries_used,
        candidates=found.candidates,
        screening=screening,
        audits=audits,
        audit_usage=audit_usage,
        dropped=dropped,
        partial=partial,
        duration_sec=time.monotonic() - started,
    )

    log.info(
        "scan_cost",
        intent_usd=round(intent_usage.cost_usd, 6),
        screening_usd=round(usage.cost_usd, 6),
        audit_usd=round(audit_usage.cost_usd, 6) if audit_usage else 0.0,
        total_cost_usd=round(outcome.total_cost_usd, 6),
        pricing_window=usage.pricing_window.value,
    )
    log.info(
        "scan_finished",
        status=outcome.status.value,
        partial=partial,
        passed=len(screening.result.passed),
        dropped=[item.model_dump(mode="json") for item in dropped],
        duration_sec=round(outcome.duration_sec, 1),
    )

    if outcome.status is ScanStatus.NONE_PASSED:
        log.info(
            "scan_build_recommended",
            reason="none_passed",
            screened=len(found.candidates),
            queries=found.queries_used,
            partial=partial,
        )

    return outcome
