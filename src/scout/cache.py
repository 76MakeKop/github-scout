"""Кэш аудитов: SQLite, одна строка на `repo_id + head_sha + prompt_version` (§9).

Кэшируется **только** `AuditResult` Слоя 2. `ScreeningResult` Слоя 1 не кэшируется:
он дёшев, а после распараллеливания ещё и быстр — хранение обошлось бы дороже
пересчёта. Кэш по хешу текста запроса отвергнут в `decisions_log.md`: NL-запросы
не совпадают побайтово, а один репозиторий попадает в выдачу многих разных задач.

Инвалидация по двум осям. Смена `head_sha` — изменился код. Смена
`prompt_version` — изменился вопрос, который мы про этот код задаём: `AuditResult`
не свойство репозитория, а ответ конкретного промпта о нём. Без версии в ключе
бамп до `l2-2` молча отдавал бы суждения `l2-1`, и замер «до и после», которого
требует `CHECKLIST.md` при любой правке промпта, показывал бы «до» оба раза —
ошибка тихая, числа приходят правдоподобные и неверные.

Новый ключ даёт автоматический промах, старые строки остаются жить: прошлый
прогон должен воспроизводиться. TTL по времени не используется — он либо отдаёт
устаревшее, либо жжёт токены на неизменившемся коде.

Слой 2 ходит сюда через `lookup_or_store`: попадание означает, что V4-Pro
не вызывается вовсе.
"""

import json
import os
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

DEFAULT_CACHE_PATH = Path(
    os.environ.get("SCOUT_CACHE_PATH", config.PROJECT_ROOT / ".cache" / "scout.sqlite3")
)
"""Переопределяется `SCOUT_CACHE_PATH`. Нужно, когда два замера идут параллельно:
прогоны «до» и «после» пишут разные версии промпта и логически не конфликтуют,
но делить один файл SQLite между процессами — значит ловить блокировки на ровном
месте. Разные файлы убирают общее изменяемое состояние совсем."""

PAYLOAD_TYPE = "audit_result_v1"
"""Версия формата полезной нагрузки. При переходе на `_v2` старые строки
не читаются и не удаляются автоматически — чистка через `scout cache drop`."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    key            TEXT PRIMARY KEY,
    repo_id        INTEGER NOT NULL,
    head_sha       TEXT NOT NULL,
    prompt_version TEXT NOT NULL DEFAULT 'l2-1',
    payload_type   TEXT NOT NULL DEFAULT 'audit_result_v1',
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    hits           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS audits_repo_id ON audits (repo_id);
"""


LEGACY_PROMPT_VERSION = "l2-1"
"""Версия, которой сделаны все записи без суффикса в ключе: до дня 13 промпт
Слоя 2 существовал ровно один, поэтому старые строки не выбрасываются,
а домигрируются — иначе честные попадания терялись бы на ровном месте."""


def cache_key(repo_id: int, head_sha: str, prompt_version: str) -> str:
    """`repo:{repo_id}:{head_sha}:{prompt_version}` — ключ собирает код, а не модель."""
    return f"repo:{repo_id}:{head_sha}:{prompt_version}"


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
            self._migrate_legacy_keys()
            self._connection.commit()

    def _migrate_legacy_keys(self) -> None:
        """Дописывает `:l2-1` ключам, созданным до появления версии в ключе.

        Выбросить их было бы проще, но неправильно: все они сделаны единственным
        существовавшим тогда промптом, это законные попадания, за которые уже
        заплачено. Столбца `prompt_version` в старой таблице нет вовсе — его
        добавляет `ALTER TABLE`, а `DEFAULT 'l2-1'` заполняет существующие строки.
        """
        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(audits)").fetchall()
        }
        if "prompt_version" not in columns:
            self._connection.execute(
                f"ALTER TABLE audits ADD COLUMN prompt_version TEXT NOT NULL"
                f" DEFAULT '{LEGACY_PROMPT_VERSION}'"
            )

        self._connection.execute(
            "UPDATE audits SET key = key || ':' || prompt_version WHERE key NOT LIKE '%:l2-%'"
        )

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
                " (key, repo_id, head_sha, prompt_version, payload_type,"
                "  payload, created_at, hits)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.key,
                    entry.repo_id,
                    entry.head_sha,
                    entry.prompt_version,
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
            prompt_version=row["prompt_version"],
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
    prompt_version: str,
    produce: Callable[[], AuditResult],
    logger: RunLogger | None = None,
) -> AuditResult:
    """Политика «сначала кэш»: попадание — модель не зовём, промах — зовём и запоминаем.

    `produce` — вызов модели Слоем 2. Политика живёт здесь, а не в слое, чтобы
    условия попадания и запись события были в одном месте с самим хранилищем.
    """
    key = cache_key(repo_id, head_sha, prompt_version)

    entry = cache.get_entry(key)
    if entry is not None:
        if logger:
            logger.info(
                "cache_hit",
                repo_id=repo_id,
                full_name=full_name,
                head_sha=head_sha,
                prompt_version=prompt_version,
                hits=entry.hits,
            )
        return entry.payload

    if logger:
        logger.info(
            "cache_miss",
            repo_id=repo_id,
            full_name=full_name,
            head_sha=head_sha,
            prompt_version=prompt_version,
        )

    result = produce()
    cache.put(
        CacheEntry(
            key=key,
            repo_id=repo_id,
            head_sha=head_sha,
            prompt_version=prompt_version,
            payload_type=PAYLOAD_TYPE,
            payload=result,
            created_at=datetime.now(UTC),
            hits=0,
        )
    )
    return result
