"""CLI. Конвейер доходит до извлечённого интента и останавливается.

Генерации запросов, поиска и слоёв здесь ещё нет — следующие пункты ROADMAP.md.
"""

import argparse
import sys
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from scout import config
from scout.config import MissingCredential
from scout.deepseek import DeepSeekError
from scout.intent import extract_intent
from scout.log import RunLogger, new_run_id
from scout.schemas import ScanOptions, ScanRequest

NOT_YET = "реализуется на неделе 2"


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

    log.info("reached_stub", stage="queries", note="генерация запросов: следующий пункт плана")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    print(f"cache {args.cache_command}: {NOT_YET}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    return cmd_cache(args)
