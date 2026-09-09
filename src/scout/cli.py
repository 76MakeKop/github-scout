"""CLI. Конвейер доходит до конца Слоя 1 и останавливается перед аудитом.

Слоя 2, кэша и отчёта здесь ещё нет — следующие пункты ROADMAP.md.

Коды возврата. Ноль означает «ответ получен», даже если он неполный: скан,
потерявший двух кандидатов из пятидесяти, полезен, и заставлять вызывающий
скрипт считать это провалом незачем. Ненулевые коды различают, что именно
сломалось, — по ним видно, чинить конфигурацию, ждать сервис или править запрос:

| Код | Что произошло |
|---|---|
| 0 | результат есть: полный, частичный (`partial`) или обоснованный BUILD |
| 1 | скан не дал ничего: сервис не ответил или упал неожиданный сбой |
| 2 | запрос не проходит схему `ScanRequest` |
| 3 | ошибка конфигурации: ключа нет либо он отклонён (401/403) |
| 4 | DeepSeek недоступен после всех повторов |
| 5 | модель дважды вернула интент не по схеме |
| 6 | из интента не собрался ни один поисковый запрос |
| 7 | GitHub отказал так, что поиск не состоялся |
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from scout import config
from scout.cache import DEFAULT_CACHE_PATH, AuditCache
from scout.config import MissingCredential
from scout.deepseek import DeepSeekAuth, DeepSeekError
from scout.evaluate import (
    DEFAULT_EVAL_DIR,
    DEFAULT_GOLDEN_DIR,
    EvalError,
    EvalRun,
    evaluate,
    load_cases,
    write_report,
)
from scout.github import GitHubAuth, GitHubError
from scout.log import RunLogger, new_run_id
from scout.pipeline import IntentUnparsed, ScanStatus, run_scan
from scout.queries import QueryGenerationError
from scout.scheduler import DEFAULT_QUEUE_PATH, PendingScans, next_offpeak_start
from scout.schemas import (
    AuditResult,
    DroppedCandidate,
    PricingWindow,
    ScanOptions,
    ScanRequest,
    ScreeningItem,
    TokenUsage,
)
from scout.screening import ScreeningRun

BUILD_ADVICE = "Кандидатов нет — рекомендация BUILD: подходящего открытого решения не нашлось."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scout",
        description="GitHub Scout — поиск и оценка открытых решений под задачу",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="найти решения под задачу")
    scan.add_argument("query", help="описание задачи обычными словами")
    scan.add_argument(
        "--max-candidates",
        type=int,
        default=config.MAX_CANDIDATES,
        help=f"кандидатов на Слой 1 (по умолчанию {config.MAX_CANDIDATES})",
    )
    scan.add_argument(
        "--audit-limit",
        type=int,
        default=config.MAX_AUDIT_CANDIDATES,
        help=f"кандидатов на Слой 2 (по умолчанию {config.MAX_AUDIT_CANDIDATES})",
    )
    scan.add_argument(
        "--report-limit",
        type=int,
        default=config.MAX_REPORT_CANDIDATES,
        help=f"кандидатов в отчёте (по умолчанию {config.MAX_REPORT_CANDIDATES})",
    )
    scan.add_argument(
        "--off-peak",
        action="store_true",
        help="отложить до ближайшего off-peak окна (дешевле, но не сразу)",
    )
    scan.add_argument(
        "--refresh",
        action="store_true",
        help="игнорировать кэш аудитов для этого запуска",
    )

    cache = sub.add_parser("cache", help="управление кэшем аудитов")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("list", help="показать записи кэша")
    drop = cache_sub.add_parser("drop", help="удалить записи по репозиторию")
    drop.add_argument("repo_id", type=int, help="repo_id из GitHub")

    worker = sub.add_parser("worker", help="выполнить отложенные сканы, чьё время пришло")
    worker_sub = worker.add_subparsers(dest="worker_command", required=True)
    worker_sub.add_parser("run", help="разгрести очередь --off-peak")

    evaluation = sub.add_parser("eval", help="прогнать golden-set и посчитать recall")
    evaluation.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_DIR),
        help=f"каталог с задачами (по умолчанию {DEFAULT_GOLDEN_DIR})",
    )
    evaluation.add_argument(
        "--max-candidates",
        type=int,
        default=config.MAX_CANDIDATES,
        help=f"кандидатов на Слой 1 в каждой задаче (по умолчанию {config.MAX_CANDIDATES})",
    )
    evaluation.add_argument(
        "--audit-limit",
        type=int,
        default=config.MAX_AUDIT_CANDIDATES,
        help="срез, по которому считается recall@10",
    )
    evaluation.add_argument(
        "--limit", type=int, default=None, help="прогнать только первые N задач набора"
    )
    evaluation.add_argument(
        "--out", default=str(DEFAULT_EVAL_DIR), help="куда положить JSON прогона"
    )
    evaluation.add_argument(
        "--journal",
        default=None,
        help="черновик прогона: задача пишется на диск сразу, как отработала. "
        "Повторный запуск с тем же файлом досчитывает только недостающее "
        "и не платит за уже измеренное",
    )
    evaluation.add_argument(
        "--dry-run",
        action="store_true",
        help="проверить набор и выйти: ни одного вызова модели, ни одного цента",
    )

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    log = RunLogger(new_run_id())

    try:
        request = ScanRequest(
            request_id=uuid4(),
            query_text=args.query,
            created_at=datetime.now(UTC),
            options=ScanOptions(
                max_candidates=args.max_candidates,
                audit_limit=args.audit_limit,
                report_limit=args.report_limit,
                off_peak=args.off_peak,
                refresh=args.refresh,
            ),
        )
    except ValidationError as exc:
        problems = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        log.error("invalid_request", problems=problems)
        print("Запрос отклонён:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    # Момент берётся один раз и передаётся дальше: иначе очередь, тарификация
    # и проверка peak-часа считали бы время каждая по своим часам.
    now = datetime.now(UTC)

    # Очередь разгребается перед своей работой: демона нет, и это единственный
    # момент, когда отложенная задача может дождаться исполнения.
    drain_pending(log, moment=now)

    if request.options.off_peak and config.is_peak(now):
        return _defer(request, log, now)

    return _execute_scan(request, log)


def _defer(request: ScanRequest, log: RunLogger, now: datetime) -> int:
    """Сейчас peak и попросили подождать — ставим в очередь (CLAUDE.md, запрет 8)."""
    starts_at = next_offpeak_start(now)

    with PendingScans(DEFAULT_QUEUE_PATH) as queue:
        task_id = queue.add(request.query_text, request.options, scheduled_at=starts_at)

    log.info(
        "scan_deferred",
        task_id=task_id,
        scheduled_at=starts_at.isoformat().replace("+00:00", "Z"),
        reason="peak_hours",
    )
    print(
        f"⏳ Задача в очереди. Запустится автоматически "
        f"в {starts_at.strftime('%H:%M')} UTC — в off-peak вдвое дешевле."
    )
    print("   Разгрести очередь вручную: python -m scout worker run")
    return 0


def drain_pending(log: RunLogger, *, moment: datetime | None = None) -> int:
    """Выполняет отложенные задачи, чьё время пришло. Возвращает их число."""
    moment = moment or datetime.now(UTC)

    with PendingScans(DEFAULT_QUEUE_PATH) as queue:
        due = queue.due(moment)
        if not due:
            return 0

        log.info("pending_scans_due", count=len(due))
        for task in due:
            print(f"▶ Выполняю отложенную задачу #{task.id}: «{task.query}»")
            request = ScanRequest(
                request_id=uuid4(),
                query_text=task.query,
                created_at=datetime.now(UTC),
                # Флаг снят намеренно: время пришло, второй раз откладывать нечего.
                options=task.options.model_copy(update={"off_peak": False}),
            )
            _execute_scan(request, log)
            queue.remove(task.id)

    return len(due)


def _execute_scan(request: ScanRequest, log: RunLogger) -> int:
    """Один скан: конвейер в `pipeline.py`, здесь — печать и коды возврата.

    Отказы разбираются тут, а не в конвейере: у оболочки и у прогона golden-set
    на один и тот же отказ разные ответы — первой нужен код возврата, второму
    строка в таблице результатов.
    """
    try:
        outcome = run_scan(request, log=log)
    except (MissingCredential, DeepSeekAuth, GitHubAuth) as exc:
        return _credential_error(log, exc)
    except IntentUnparsed:
        print(
            "Не удалось разобрать задачу: модель дважды вернула невалидный ответ.", file=sys.stderr
        )
        return 5
    except DeepSeekError as exc:
        log.error("deepseek_failed", detail=str(exc))
        print(f"DeepSeek недоступен: {exc}", file=sys.stderr)
        return 4
    except QueryGenerationError as exc:
        log.error("queries_failed", detail=str(exc))
        print(f"Не удалось собрать поисковые запросы: {exc}", file=sys.stderr)
        return 6
    except GitHubError as exc:
        log.error("search_failed_hard", detail=str(exc))
        print(f"Поиск по GitHub не удался: {exc}", file=sys.stderr)
        return 7

    if outcome.status is ScanStatus.NO_CANDIDATES:
        # Пустой поиск — это ответ, а не сбой: подходящего решения не нашлось,
        # и BUILD здесь обоснован (ARCHITECTURE.md, последняя строка таблицы).
        # Сбой отличается от него флагом `partial`, а не кодом возврата.
        print(BUILD_ADVICE)
        _print_queries(outcome.queries_used)
        _print_degradation(outcome.partial, outcome.dropped)
        return 0

    screening = outcome.screening
    usage = outcome.screening_usage

    if outcome.status is ScanStatus.NONE_PASSED:
        print("Ни один кандидат не прошёл скрининг — рекомендация BUILD.")
        _print_queries(outcome.queries_used)
        _print_degradation(outcome.partial, outcome.dropped)
        _print_cost(outcome.intent_usage, usage, outcome.total_cost_usd)
        return 0

    _print_passed(screening, len(outcome.candidates))
    _print_audits(outcome.audits)
    _print_degradation(outcome.partial, outcome.dropped)
    _print_cost(outcome.intent_usage, usage, outcome.total_cost_usd, outcome.audit_usage)

    log.info("reached_stub", stage="report", note="Отчёт: следующий пункт плана")
    return 0


def _credential_error(log: RunLogger, exc: Exception) -> int:
    """Ключа нет или он отклонён — чинится в `.env`, повторять нечего."""
    log.error("credential_rejected", error_type=type(exc).__name__, detail=str(exc))
    print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
    return 3


def _print_queries(queries: list[str]) -> None:
    """Список проверенных запросов при рекомендации BUILD.

    Без него BUILD выглядит как «ничего не нашлось», хотя обычно означает
    «искали вот так и не нашлось» — а это проверяемое утверждение.
    """
    print(f"Проверено запросов: {len(queries)}.")
    for query in queries:
        print(f"  - {query}")


def _print_degradation(partial: bool, dropped: list[DroppedCandidate]) -> None:
    """Что скан потерял по дороге. Молчать об этом нельзя: неполный результат,
    выданный как полный, — худший из возможных ответов."""
    if not dropped and not partial:
        return

    if partial:
        print("\n⚠ Результат неполный: часть данных потеряна из-за сбоев (partial).")

    if dropped:
        print(f"Выбыло кандидатов: {len(dropped)}")
        for item in dropped:
            print(f"  - {item.full_name} ({item.stage.value}): {item.reason}")


def _print_audits(audits: list[AuditResult]) -> None:
    """Аудит по каждому кандидату, по убыванию `total`.

    Это ещё не отчёт (день 13): здесь нет ни пятёрки, ни provenance построчно —
    только то, что Слой 2 успел выяснить, в порядке, который посчитал код.
    """
    if not audits:
        return

    print(f"\nСлой 2 — аудит ({len(audits)}):")
    for audit in sorted(audits, key=lambda a: (-a.score.total, a.repo_id)):
        passport = audit.license_passport
        licence = passport.spdx_id or "лицензия не определена"
        reuse = "можно брать код" if passport.code_reuse_allowed else "код брать нельзя"
        effort = audit.fit.integration_effort_days
        print(f"  {audit.verdict.value:5} {audit.score.total:.2f}  {audit.full_name}")
        print(
            f"        {licence} ({passport.copyleft.value}, {reuse}) · "
            f"интеграция {effort.low:g}–{effort.high:g} дн, вероятно {effort.likely:g}"
        )
        if audit.fit.gaps:
            print(f"        не покрывает: {'; '.join(audit.fit.gaps[:2])}")
        for risk in audit.risks[:2]:
            print(f"        риск {risk.type.value}/{risk.severity.value}: {risk.note}")


def _print_cost(
    intent_usage: TokenUsage,
    screening_usage: TokenUsage,
    total: float,
    audit_usage: TokenUsage | None = None,
) -> None:
    """Стоимость по этапам: интент, Слой 1, Слой 2."""
    window = "peak" if screening_usage.pricing_window is PricingWindow.PEAK else "off-peak"

    print(f"\nСтоимость скана ({window}):")
    print(
        f"  интент    {intent_usage.cost_usd:.6f} $"
        f"   ({intent_usage.input_tokens} вход / {intent_usage.output_tokens} выход,"
        f" {intent_usage.cached_input_tokens} из кэша)"
    )
    print(
        f"  Слой 1    {screening_usage.cost_usd:.6f} $"
        f"   ({screening_usage.input_tokens} вход / {screening_usage.output_tokens} выход,"
        f" {screening_usage.cached_input_tokens} из кэша)"
    )
    if audit_usage:
        print(
            f"  Слой 2    {audit_usage.cost_usd:.6f} $"
            f"   ({audit_usage.input_tokens} вход / {audit_usage.output_tokens} выход,"
            f" {audit_usage.cached_input_tokens} из кэша)"
        )
    print(f"  total_cost_usd {total:.6f} $")


def _print_passed(screening: ScreeningRun, screened: int) -> None:
    """Отчёта на дне 5 ещё нет — печатаем то, что уже есть: кто прошёл и почему."""
    by_id: dict[int, ScreeningItem] = {item.repo_id: item for item in screening.result.results}

    print(f"Слой 1: {len(screening.result.passed)} из {screened} кандидатов прошли скрининг.\n")
    for position, repo_id in enumerate(screening.result.passed, start=1):
        item = by_id[repo_id]
        print(f"{position:2}. {item.full_name}  relevance {item.relevance:.2f}")
        print(f"    {item.reasons[0]}")

    if screening.failed:
        print(f"\nНе разобраны моделью: {', '.join(screening.failed)}")


def cmd_cache(args: argparse.Namespace) -> int:
    """Обслуживание кэша аудитов. Сам кэш наполнится, когда появится Слой 2."""
    with AuditCache(DEFAULT_CACHE_PATH) as cache:
        if args.cache_command == "list":
            entries = cache.entries()
            if not entries:
                print("Кэш пуст.")
                return 0

            print(f"Записей в кэше: {len(entries)}")
            for entry in entries:
                print(
                    f"  {entry.key}  {entry.payload.full_name}"
                    f"  вердикт {entry.payload.verdict.value}"
                    f"  попаданий {entry.hits}"
                    f"  от {entry.created_at.date().isoformat()}"
                )
            return 0

        removed = cache.drop(args.repo_id)
        print(f"Удалено записей: {removed}")
        return 0


def cmd_worker(args: argparse.Namespace) -> int:
    """Разгребает очередь `--off-peak`. Демона нет: воркер запускается руками
    или заодно при следующем `scan`."""
    log = RunLogger(new_run_id())
    executed = drain_pending(log)

    if not executed:
        with PendingScans(DEFAULT_QUEUE_PATH) as queue:
            waiting = queue.all()
        if waiting:
            nearest = min(task.scheduled_at for task in waiting)
            print(
                f"В очереди {len(waiting)} задач(и), ближайшая — в {nearest.strftime('%H:%M')} UTC."
            )
        else:
            print("Очередь пуста.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Прогон golden-set. `--dry-run` проверяет набор, не тратя ни цента."""
    log = RunLogger(new_run_id())

    try:
        cases = load_cases(Path(args.golden))
    except EvalError as exc:
        log.error("golden_set_invalid", detail=str(exc))
        print(f"Набор не читается: {exc}", file=sys.stderr)
        return 2

    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"Задач в наборе: {len(cases)}")
    for case in cases:
        target = ", ".join(case.expected_repos) if case.expected_repos else "— (ловушка: BUILD)"
        print(f"  {case.slug:24} {case.query_text}")
        print(f"  {'':24} эталон: {target}")

    if args.dry_run:
        print("\nПроверка набора пройдена. Прогон не запускался (--dry-run).")
        return 0

    options = ScanOptions(max_candidates=args.max_candidates, audit_limit=args.audit_limit)
    print(
        f"\nПрогон: {len(cases)} задач × {options.max_candidates} кандидатов, "
        f"recall@{options.audit_limit} после Слоя 1.\n"
    )

    try:
        run = evaluate(
            cases,
            options=options,
            log=log,
            journal=Path(args.journal) if args.journal else None,
        )
    except (MissingCredential, DeepSeekAuth, GitHubAuth) as exc:
        return _credential_error(log, exc)
    except DeepSeekError as exc:
        log.error("deepseek_failed", detail=str(exc))
        print(f"DeepSeek недоступен: {exc}", file=sys.stderr)
        return 4
    except GitHubError as exc:
        log.error("search_failed_hard", detail=str(exc))
        print(f"Поиск по GitHub не удался: {exc}", file=sys.stderr)
        return 7

    _print_eval(run, options)
    path = write_report(run, Path(args.out))
    print(f"\nПодробности прогона: {path}")
    return 0


def _print_eval(run: EvalRun, options: ScanOptions) -> None:
    """Таблица по задачам и сводка. Оба среза recall — рядом: их разница и есть диагноз."""
    print(f"{'задача':24} {'recall@' + str(options.audit_limit):>10} {'recall@50':>10}  статус")
    for result in run.results:
        if result.error:
            print(f"{result.slug:24} {'—':>10} {'—':>10}  ошибка: {result.error}")
            continue
        flag = "  ⚠ partial" if result.partial else ""
        # У ловушки нет эталона, значит нет и потолка поиска: прочерк честнее нуля.
        ceiling = "—" if result.recall_at_50 is None else f"{result.recall_at_50:.2f}"
        print(f"{result.slug:24} {result.recall_at_10:>10.2f} {ceiling:>10}  {result.status}{flag}")

    summary = run.summary()
    verdict = "цель недели 2 взята" if summary["target_met"] else "ниже цели 0,60 — чинить поиск"
    print(
        f"\nrecall@{options.audit_limit}: {summary['recall_at_10']:.2f} — {verdict}"
        f"\nrecall@50 (потолок поиска): {summary['recall_at_50']:.2f}"
        f"\nЗадач измерено: {summary['cases_measured']} из {summary['cases_total']}"
        f", ловушек {summary['traps']}, неполных прогонов {summary['partial_runs']}"
        f"\nСтоимость: {summary['cost_usd_total']:.4f} $"
        f", медиана времени задачи {summary['duration_sec_median']:.0f} с"
    )


def main(argv: list[str] | None = None) -> int:
    """Последняя застава: наружу выходит код возврата, а не traceback.

    Ниже по конвейеру каждый предвиденный отказ уже разобран и получил свой код.
    Этот `except` — про непредвиденное: он не лечит, а переводит падение
    в понятное сообщение и код 1, оставляя разбор в логе.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan":
            return cmd_scan(args)
        if args.command == "worker":
            return cmd_worker(args)
        if args.command == "eval":
            return cmd_eval(args)
        return cmd_cache(args)
    except Exception as exc:  # ловим всё: traceback пользователю бесполезен
        RunLogger(new_run_id()).error("scan_crashed", error_type=type(exc).__name__, error=str(exc))
        print(f"Скан прерван: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
