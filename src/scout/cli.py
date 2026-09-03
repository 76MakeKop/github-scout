"""CLI. Конвейер доходит до конца Слоя 1 и останавливается перед аудитом.

Слоя 2, кэша и отчёта здесь ещё нет — следующие пункты ROADMAP.md.
"""

import argparse
import sys
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from scout import config
from scout.cache import DEFAULT_CACHE_PATH, AuditCache
from scout.config import MissingCredential
from scout.deepseek import DeepSeekError
from scout.github import GitHubClient, GitHubError
from scout.intent import extract_intent
from scout.log import RunLogger, new_run_id
from scout.queries import QueryGenerationError, build_query_set
from scout.schemas import ScanOptions, ScanRequest, ScreeningItem
from scout.screening import ScreeningRun, screen
from scout.search import collect_candidates

NOT_YET = "реализуется на неделе 2"

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

    log.info(
        "start",
        request_id=str(request.request_id),
        query_text=request.query_text,
        options=request.options.model_dump(),
        pricing_window="peak" if config.is_peak() else "off-peak",
    )

    try:
        extraction = extract_intent(request.query_text, request_id=request.request_id, logger=log)
    except MissingCredential as exc:
        log.error("missing_credential", detail=str(exc))
        print(f"Не хватает ключа: {exc}", file=sys.stderr)
        return 3
    except DeepSeekError as exc:
        log.error("deepseek_failed", detail=str(exc))
        print(f"DeepSeek недоступен: {exc}", file=sys.stderr)
        return 4

    if extraction.status == "failed":
        log.error("intent_failed", attempts=extraction.attempts, problems=extraction.errors)
        print(
            "Не удалось разобрать задачу: модель дважды вернула невалидный ответ.", file=sys.stderr
        )
        return 5

    intent = extraction.intent
    log.info(
        "intent_extracted",
        synonyms_count=len(intent.synonyms),
        hypothesis_count=len(intent.known_libraries),
        task=intent.task,
        languages=intent.languages,
        attempts=extraction.attempts,
        prompt_version=intent.prompt_version,
        **extraction.usage,
    )

    try:
        query_set = build_query_set(intent, logger=log)
    except QueryGenerationError as exc:
        log.error("queries_failed", detail=str(exc))
        print(f"Не удалось собрать поисковые запросы: {exc}", file=sys.stderr)
        return 6

    log.info(
        "queries_generated",
        query_count=len(query_set.queries),
        generator_version=query_set.generator_version,
        families=[query.family.value for query in query_set.queries],
        queries=[query.q for query in query_set.queries],
    )

    try:
        github = GitHubClient(token=config.github_token(), logger=log)
    except MissingCredential as exc:
        log.error("missing_credential", detail=str(exc))
        print(f"Не хватает ключа: {exc}", file=sys.stderr)
        return 3

    try:
        found = collect_candidates(
            query_set,
            intent=intent,
            github=github,
            limit=request.options.max_candidates,
            logger=log,
        )
    except GitHubError as exc:
        log.error("search_failed_hard", detail=str(exc))
        print(f"Поиск по GitHub не удался: {exc}", file=sys.stderr)
        return 7

    if not found.candidates:
        log.info("scan_build_recommended", reason="no_candidates", queries=found.queries_used)
        print(BUILD_ADVICE)
        print(f"Проверено запросов: {len(found.queries_used)}.")
        return 0

    log.info("reached_stub", stage="screening", note="Слой 1: скрининг кандидатов")

    try:
        screening = screen(
            found.candidates,
            intent,
            request_id=request.request_id,
            github=github,
            logger=log,
            limit=request.options.audit_limit,
        )
    except MissingCredential as exc:
        log.error("missing_credential", detail=str(exc))
        print(f"Не хватает ключа: {exc}", file=sys.stderr)
        return 3
    except DeepSeekError as exc:
        log.error("deepseek_failed", detail=str(exc))
        print(f"DeepSeek недоступен: {exc}", file=sys.stderr)
        return 4

    usage = screening.result.token_usage
    log.info(
        "screening_done",
        screened=len(screening.result.results),
        passed=len(screening.result.passed),
        failed=len(screening.failed),
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
        prompt_version=screening.result.prompt_version,
    )

    if not screening.result.passed:
        log.info("scan_build_recommended", reason="none_passed", screened=len(found.candidates))
        print("Ни один кандидат не прошёл скрининг — рекомендация BUILD.")
        return 0

    _print_passed(screening, len(found.candidates))

    log.info("reached_stub", stage="audit", note="Слой 2: следующий пункт плана")
    return 0


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    return cmd_cache(args)
