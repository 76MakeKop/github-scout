"""CLI. День 1: конвейер доходит до заглушки интента и останавливается.

Ни сети, ни LLM, ни SQLite здесь нет — по ROADMAP.md это дни 2 и 6.
"""

import argparse
import sys
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from scout import config
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

    log.info(
        "reached_stub",
        stage="intent",
        note="intent: stub",
        prompt_version=config.PROMPT_VERSIONS["intent"],
    )
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    print(f"cache {args.cache_command}: {NOT_YET}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    return cmd_cache(args)
