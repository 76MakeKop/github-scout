"""Прогон golden-set и метрика recall (ROADMAP.md → «Оценка качества»).

Неделя 2 заканчивается числом, а не ощущением. Число здесь считается по двум
срезам сразу, и это не избыточность:

- **recall@50** — доля эталонов, которые вообще нашлись поиском. Потолок всего
  остального: чего не нашёл поиск, того не спасёт никакой слой.
- **recall@10** — доля эталонов, доживших до конца Слоя 1. Целевая метрика
  недели 2 и потолок для recall@5, который появится вместе с отчётом.

Разница между ними и есть диагноз. Низкий recall@50 — болит `QUERIES.md`;
recall@50 высокий, а recall@10 низкий — болит промпт скрининга. Без обоих чисел
эти две починки неразличимы, и на дне 10 пришлось бы гадать.

Задачи-ловушки (`expected_repos: []`) считаются наоборот: единица, если агент
никого не пропустил дальше, и ноль, если пропустил. Пока Слоя 2 нет, «рекомендация
BUILD» — это и есть пустой `passed`.
"""

import json
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from scout import config
from scout.config import MissingCredential
from scout.deepseek import DeepSeekAuth
from scout.github import GitHubAuth, GitHubClient
from scout.log import RunLogger
from scout.pipeline import ScanOutcome, run_scan
from scout.schemas import ScanOptions, ScanRequest, SchemaModel, Verdict

DEFAULT_GOLDEN_DIR = Path("tests/golden")
DEFAULT_EVAL_DIR = Path("eval")

RECALL_TARGET = 0.60
"""Цель недели 2 (`ROADMAP.md`): ниже — неделя 3 начинается с починки поиска,
а не с постройки Слоя 2 поверх плохой выдачи."""


class EvalError(RuntimeError):
    """Набор не читается. Это ошибка данных, а не результат замера."""


class GoldenCase(SchemaModel):
    """Одна задача набора. Файл `tests/golden/{slug}.json`, имя файла и есть slug."""

    slug: str
    query_text: str
    expected_repos: list[str]
    expected_verdict: Verdict
    note: str

    @property
    def is_trap(self) -> bool:
        """Ловушка: готового решения нет, правильный ответ — BUILD.

        Без таких задач метрика поощряет выдавать что угодно: пять строк в отчёте
        всегда лучше пустого ответа, если пустой ответ не засчитывается никогда.
        """
        return not self.expected_repos


@dataclass
class CaseResult:
    """Результат одной задачи. `error` заполнен, если прогон не состоялся."""

    slug: str
    query_text: str
    status: str
    expected: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    partial: bool = False
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "query_text": self.query_text,
            "status": self.status,
            "expected": self.expected,
            "found": self.found,
            "passed": self.passed,
            "recall_at_10": round(self.recall_at_10, 4),
            "recall_at_50": round(self.recall_at_50, 4),
            "cost_usd": round(self.cost_usd, 6),
            "duration_sec": round(self.duration_sec, 1),
            "partial": self.partial,
            "error": self.error,
        }


@dataclass
class EvalRun:
    """Весь прогон: строки по задачам и сводка."""

    results: list[CaseResult]
    options: ScanOptions
    generated_at: datetime

    @property
    def measured(self) -> list[CaseResult]:
        """Задачи, которые действительно прогнались. Упавшие в среднее не берём.

        Иначе сбой сети занижал бы recall и выглядел бы как плохое качество поиска —
        ровно та подмена, ради предотвращения которой день 8 и делался.
        """
        return [result for result in self.results if result.error is None]

    def summary(self) -> dict[str, Any]:
        measured = self.measured
        durations = [result.duration_sec for result in measured]
        return {
            "cases_total": len(self.results),
            "cases_measured": len(measured),
            "cases_failed": len(self.results) - len(measured),
            "traps": sum(1 for result in measured if not result.expected),
            "recall_at_10": _mean(result.recall_at_10 for result in measured),
            "recall_at_50": _mean(result.recall_at_50 for result in measured),
            "target_met": _mean(result.recall_at_10 for result in measured) >= RECALL_TARGET,
            "cost_usd_total": round(sum(result.cost_usd for result in measured), 6),
            "duration_sec_median": round(statistics.median(durations), 1) if durations else 0.0,
            "partial_runs": sum(1 for result in measured if result.partial),
        }

    def as_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat().replace("+00:00", "Z"),
            "options": self.options.model_dump(),
            "summary": self.summary(),
            "cases": [result.as_json() for result in self.results],
        }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return round(sum(collected) / len(collected), 4) if collected else 0.0


def _normalize(full_names: Iterable[str]) -> set[str]:
    """GitHub нечувствителен к регистру имён, а эталон пишет человек."""
    return {name.strip().lower() for name in full_names if name.strip()}


def load_cases(directory: Path = DEFAULT_GOLDEN_DIR) -> list[GoldenCase]:
    """Читает `*.json` из каталога. Порядок по slug — чтобы прогоны сравнивались."""
    if not directory.is_dir():
        raise EvalError(f"каталог golden-set не найден: {directory}")

    cases: list[GoldenCase] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path.name}: не разбирается как JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise EvalError(f"{path.name}: ожидался объект, получен {type(payload).__name__}")

        try:
            cases.append(GoldenCase(slug=path.stem, **payload))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise EvalError(f"{path.name}: {problems}") from exc

    if not cases:
        raise EvalError(f"в каталоге {directory} нет ни одного файла задачи")
    return cases


def recall(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Доля эталонов, попавших в срез. Пустой эталон — задача-ловушка.

    Для ловушки правильный ответ «никого не нашлось»: единица за пустой срез
    и ноль за непустой. Считать её через долю нельзя — деление на ноль, — а
    выбрасывать из набора значит потерять единственную защиту от агента,
    который всегда отвечает пятёркой кандидатов.
    """
    wanted = _normalize(expected)
    if not wanted:
        return 0.0 if _normalize(actual) else 1.0
    return len(wanted & _normalize(actual)) / len(wanted)


def evaluate_case(
    case: GoldenCase,
    *,
    options: ScanOptions,
    log: RunLogger,
    github: GitHubClient | None = None,
    runner: Callable[..., ScanOutcome] = run_scan,
) -> CaseResult:
    """Один прогон и его метрики. Сбой задачи не роняет набор (день 8)."""
    request = ScanRequest(
        request_id=uuid4(),
        query_text=case.query_text,
        created_at=datetime.now(UTC),
        options=options,
    )
    log.info("eval_case_started", slug=case.slug, query_text=case.query_text)

    try:
        outcome = runner(request, log=log, github=github)
    except (MissingCredential, DeepSeekAuth, GitHubAuth):
        # Ключ одинаков для всех задач набора: продолжать значит потратить час
        # на двадцать пять одинаковых отказов.
        raise
    except Exception as exc:  # ловим всё: набор дороже одной задачи
        log.error("eval_case_failed", slug=case.slug, error_type=type(exc).__name__, error=str(exc))
        return CaseResult(
            slug=case.slug,
            query_text=case.query_text,
            status="failed",
            expected=list(case.expected_repos),
            error=f"{type(exc).__name__}: {exc}",
        )

    found = [candidate.full_name for candidate in outcome.candidates]
    passed = outcome.passed_full_names

    result = CaseResult(
        slug=case.slug,
        query_text=case.query_text,
        status=outcome.status.value,
        expected=list(case.expected_repos),
        found=found,
        passed=passed,
        recall_at_10=recall(case.expected_repos, passed),
        recall_at_50=recall(case.expected_repos, found),
        cost_usd=outcome.total_cost_usd,
        duration_sec=outcome.duration_sec,
        partial=outcome.partial,
    )

    log.info(
        "eval_case_done",
        slug=case.slug,
        status=result.status,
        recall_at_10=result.recall_at_10,
        recall_at_50=result.recall_at_50,
        cost_usd=round(result.cost_usd, 6),
    )
    return result


def evaluate(
    cases: Sequence[GoldenCase],
    *,
    options: ScanOptions | None = None,
    log: RunLogger,
    github: GitHubClient | None = None,
    runner: Callable[..., ScanOutcome] = run_scan,
) -> EvalRun:
    """Прогоняет набор целиком одним клиентом GitHub.

    Клиент один на все задачи намеренно: в нём живут троттлинг поиска и счётчик
    запросов, и новый клиент на каждую задачу означал бы обнуление обоих — набор
    из двадцати пяти задач упёрся бы во вторичный лимит на середине.
    """
    options = options or ScanOptions()
    github = github or GitHubClient(token=config.github_token(), logger=log)

    results: list[CaseResult] = []
    for position, case in enumerate(cases, start=1):
        log.info("eval_progress", case=position, of=len(cases), slug=case.slug)
        # Счётчик поисковых запросов принадлежит одному скану, а не набору:
        # запрет 5 CLAUDE.md ограничивает скан, и без сброса набор упёрся бы
        # в него на второй задаче.
        github.reset_search_budget()
        results.append(evaluate_case(case, options=options, log=log, github=github, runner=runner))

    run = EvalRun(results=results, options=options, generated_at=datetime.now(UTC))
    log.info("eval_done", **run.summary())
    return run


def write_report(run: EvalRun, directory: Path = DEFAULT_EVAL_DIR) -> Path:
    """Пишет `eval/{дата}.json`. Второй прогон за день не затирает первый."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = run.generated_at.date().isoformat()
    path = directory / f"{stem}.json"

    attempt = 2
    while path.exists():
        path = directory / f"{stem}-{attempt}.json"
        attempt += 1

    path.write_text(
        json.dumps(run.as_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
