"""Структурные JSON-логи: одна строка — одно событие.

Обязательные поля каждой строки: ts, level, run_id, event.
Пишем в stderr, чтобы stdout оставался под отчёт.
"""

import json
import sys
import threading
from datetime import UTC, datetime
from typing import Any, TextIO
from uuid import uuid4


def new_run_id() -> str:
    """Идентификатор одного запуска scan."""
    return uuid4().hex[:12]


class RunLogger:
    def __init__(self, run_id: str, stream: TextIO | None = None) -> None:
        self.run_id = run_id
        self._stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()

    def emit(self, event: str, level: str = "info", **fields: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": level,
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        # Слой 1 пишет из пяти потоков сразу. Без замка строки могут переслоиться
        # посередине, и лог перестанет разбираться построчно — а он машинный.
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def info(self, event: str, **fields: Any) -> None:
        self.emit(event, level="info", **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.emit(event, level="error", **fields)
