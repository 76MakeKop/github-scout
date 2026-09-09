"""Кэш аудитов: SQLite, одна строка на `repo_id + head_sha` (SCHEMAS.md §9).

Кэшируется **только** `AuditResult` Слоя 2. `ScreeningResult` Слоя 1 не кэшируется:
он дёшев, а после распараллеливания ещё и быстр — хранение обошлось бы дороже
пересчёта. Кэш по хешу текста запроса отвергнут в `decisions_log.md`: NL-запросы
не совпадают побайтово, а один репозиторий попадает в выдачу многих разных задач.

Инвалидация — только по смене `head_sha`. Новый коммит даёт новый ключ и,
значит, автоматический промах; старая строка при этом остаётся жить. TTL по времени
не используется: он либо отдаёт устаревшее, либо жжёт токены на неизменившемся коде.

Слой 2 появился на дне 11 и ходит сюда через `lookup_or_store`: попадание по
ключу `repo:{repo_id}:{head_sha}` означает, что V4-Pro не вызывается вовсе.
Готова политика `lookup_or_store` — ровно то место, куда Слой 2 встанет.
"""

import json
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from scout import config
from scout.log import RunLogger
from scout.schemas import AuditResult, CacheEntry

DEFAULT_CACHE_PATH = config.PROJECT_ROOT / ".cache" / "scout.sqlite3"

PAYLOAD_TYPE = "audit_result_v1"
"""Версия формата полезной нагрузки. При переходе на `_v2` старые строки
не читаются и не удаляются автоматически — чистка через `scout cache drop`."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    key          TEXT PRIMARY KEY,
    repo_id      INTEGER NOT NULL,
    head_sha     TEXT NOT NULL,
    payload_type TEXT NOT NULL DEFAULT 'audit_result_v1',
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    hits         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS audits_repo_id ON audits (repo_id);
"""


def cache_key(repo_id: int, head_sha: str) -> str:
    """`repo:{repo_id}:{head_sha}` — ключ собирает код, а не модель."""
    return f"repo:{repo_id}:{head_sha}"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class AuditCache:
    """Строки `CacheEntry` в SQLite.

    `refresh=True` — режим флага `--refresh`: чтения не видят ничего, записи
    идут как обычно. Так один прогон пересчитывает всё заново и перезаписывает
    строки, не теряя остальной кэш.
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH, *, refresh: bool = False) -> None:
        self.path = Path(path)
        self.refresh = refresh
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Слой 2 пойдёт по кандидатам параллельно, как уже ходит Слой 1, поэтому
        # соединение сразу разрешено делить между потоками — но только под замком:
        # сам по себе объект sqlite3.Connection этого не гарантирует.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()

        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    # -- контекстный менеджер ------------------------------------------------

    def __enter__(self) -> "AuditCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- чтение --------------------------------------------------------------

    def get_entry(self, key: str) -> CacheEntry | None:
        """Строка целиком, со счётчиком попаданий уже увеличенным.

        Единственное место, где растёт `hits`: считать попадания в двух местах
        значило бы получить в отчёте число, которому нельзя верить.
        """
        if self.refresh:
            return None

        with self._lock:
            row = self._connection.execute("SELECT * FROM audits WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            hits = row["hits"] + 1
            self._connection.execute("UPDATE audits SET hits = ? WHERE key = ?", (hits, key))
            self._connection.commit()

        return self._to_entry(row, hits=hits)

    def get(self, key: str) -> AuditResult | None:
        entry = self.get_entry(key)
        return entry.payload if entry is not None else None

    def entries(self) -> list[CacheEntry]:
        """Всё содержимое — для `scout cache list`. Кэш маленький, страниц не нужно."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM audits ORDER BY created_at DESC, key ASC"
            ).fetchall()
        return [self._to_entry(row) for row in rows]

    # -- запись --------------------------------------------------------------

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO audits"
                " (key, repo_id, head_sha, payload_type, payload, created_at, hits)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.key,
                    entry.repo_id,
                    entry.head_sha,
                    entry.payload_type,
                    entry.payload.model_dump_json(),
                    _iso(entry.created_at),
                    entry.hits,
                ),
            )
            self._connection.commit()

    def drop(self, repo_id: int) -> int:
        """Все строки репозитория, включая устаревшие по SHA. Возвращает число удалённых."""
        with self._lock:
            cursor = self._connection.execute("DELETE FROM audits WHERE repo_id = ?", (repo_id,))
            self._connection.commit()
            return cursor.rowcount

    # -- внутреннее ----------------------------------------------------------

    @staticmethod
    def _to_entry(row: sqlite3.Row, *, hits: int | None = None) -> CacheEntry:
        payload: dict[str, Any] = json.loads(row["payload"])
        return CacheEntry(
            key=row["key"],
            repo_id=row["repo_id"],
            head_sha=row["head_sha"],
            payload_type=row["payload_type"],
            payload=AuditResult(**payload),
            created_at=_utc(row["created_at"]),
            hits=row["hits"] if hits is None else hits,
        )


def lookup_or_store(
    cache: AuditCache,
    *,
    repo_id: int,
    full_name: str,
    head_sha: str,
    produce: Callable[[], AuditResult],
    logger: RunLogger | None = None,
) -> AuditResult:
    """Политика «сначала кэш»: попадание — модель не зовём, промах — зовём и запоминаем.

    Сюда встанет Слой 2 (дни 11–12): `produce` — это его вызов модели. Политика
    живёт здесь, а не в слое, чтобы условия попадания и запись события были
    в одном месте с самим хранилищем.
    """
    key = cache_key(repo_id, head_sha)

    entry = cache.get_entry(key)
    if entry is not None:
        if logger:
            logger.info(
                "cache_hit",
                repo_id=repo_id,
                full_name=full_name,
                head_sha=head_sha,
                hits=entry.hits,
            )
        return entry.payload

    if logger:
        logger.info("cache_miss", repo_id=repo_id, full_name=full_name, head_sha=head_sha)

    result = produce()
    cache.put(
        CacheEntry(
            key=key,
            repo_id=repo_id,
            head_sha=head_sha,
            payload_type=PAYLOAD_TYPE,
            payload=result,
            created_at=datetime.now(UTC),
            hits=0,
        )
    )
    return result
