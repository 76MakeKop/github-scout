"""Очередь отложенных сканов для режима `--off-peak`.

`CLAUDE.md`, запрет 8: по умолчанию скан выполняется сразу, независимо от часа.
Отложить можно только явным флагом — и только когда сейчас действительно peak,
иначе ждать нечего.

Демона нет намеренно. Очередь разгребается при следующем запуске CLI: команда
`scout worker run` делает это явно, а обычный `scan` — заодно, перед своей работой.
Для одного пользователя с 5–6 запусками в месяц фоновая служба стоила бы дороже
пользы, а её отказы пришлось бы отдельно диагностировать.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

from scout import config
from scout.schemas import ScanOptions

DEFAULT_QUEUE_PATH = config.PROJECT_ROOT / ".cache" / "pending_scans.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_scans (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    options_json TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pending_scans_scheduled ON pending_scans (scheduled_at);
"""


@dataclass(frozen=True)
class PendingScan:
    id: int
    query: str
    options: ScanOptions
    scheduled_at: datetime
    created_at: datetime


def next_offpeak_start(moment: datetime | None = None) -> datetime:
    """Ближайший момент, когда цена станет off-peak. Сейчас дёшево — вернёт сейчас.

    Окна заданы часами в `config.PEAK_WINDOWS_UTC`, поэтому «конец окна» — это
    ровный час: минуты и секунды обнуляются. Ждать до конца текущего peak-окна
    достаточно — между окнами 01–04 и 06–10 лежит дешёвый час, и следующее окно
    задачу уже не догонит.
    """
    moment = (moment or datetime.now(UTC)).astimezone(UTC)

    for start, end in config.PEAK_WINDOWS_UTC:
        if start <= moment.hour < end:
            return moment.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=end)

    return moment


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class PendingScans:
    """Отложенные сканы в SQLite. Живёт рядом с кэшем аудитов, но в своём файле:
    у кэша своя политика инвалидации, у очереди — своя, смешивать их незачем."""

    def __init__(self, path: Path | str = DEFAULT_QUEUE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()

        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def __enter__(self) -> "PendingScans":
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

    def add(self, query: str, options: ScanOptions, *, scheduled_at: datetime) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO pending_scans (query, options_json, scheduled_at, created_at)"
                " VALUES (?, ?, ?, ?)",
                (query, options.model_dump_json(), _iso(scheduled_at), _iso(datetime.now(UTC))),
            )
            self._connection.commit()
            return int(cursor.lastrowid or 0)

    def all(self) -> list[PendingScan]:
        return self._select("SELECT * FROM pending_scans ORDER BY scheduled_at ASC, id ASC", ())

    def due(self, moment: datetime | None = None) -> list[PendingScan]:
        """Задачи, чьё время пришло. Граница включительная: ровно назначенный
        момент — это уже «пора», а не «ещё рано»."""
        moment = moment or datetime.now(UTC)
        return self._select(
            "SELECT * FROM pending_scans WHERE scheduled_at <= ? ORDER BY scheduled_at ASC, id ASC",
            (_iso(moment),),
        )

    def remove(self, task_id: int) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM pending_scans WHERE id = ?", (task_id,))
            self._connection.commit()

    def _select(self, sql: str, params: tuple) -> list[PendingScan]:
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [
            PendingScan(
                id=row["id"],
                query=row["query"],
                options=ScanOptions(**json.loads(row["options_json"])),
                scheduled_at=_utc(row["scheduled_at"]),
                created_at=_utc(row["created_at"]),
            )
            for row in rows
        ]
