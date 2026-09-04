"""Очередь отложенных сканов: `--off-peak` ждёт ближайшего дешёвого окна.

Время везде передаётся явным параметром — иначе тесты зависели бы от часа,
в который их запустили, а очередь как раз про часы.
"""

from datetime import UTC, datetime

import pytest

from scout.scheduler import PendingScans, next_offpeak_start
from scout.schemas import ScanOptions

MONDAY_NOON = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def options(**overrides) -> ScanOptions:
    return ScanOptions(off_peak=True, **overrides)


@pytest.fixture
def queue(tmp_path):
    with PendingScans(tmp_path / ".cache" / "pending_scans.sqlite3") as opened:
        yield opened


# --------------------------------------------------------------------------
# Ближайшее off-peak окно
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "expected_hour"),
    [
        (1, 4),  # начало первого peak-окна → его конец
        (3, 4),  # середина первого окна
        (6, 10),  # начало второго окна
        (9, 10),  # середина второго окна
    ],
)
def test_peak_hour_waits_for_the_end_of_its_window(hour, expected_hour):
    moment = datetime(2026, 9, 4, hour, 30, tzinfo=UTC)
    assert next_offpeak_start(moment) == datetime(2026, 9, 4, expected_hour, 0, tzinfo=UTC)


@pytest.mark.parametrize("hour", [0, 4, 5, 10, 15, 23])
def test_offpeak_hour_starts_immediately(hour):
    """Уже дёшево — ждать нечего, «ближайшее окно» это сейчас."""
    moment = datetime(2026, 9, 4, hour, 30, tzinfo=UTC)
    assert next_offpeak_start(moment) == moment


def test_gap_between_peak_windows_is_offpeak():
    """05:00 UTC лежит между окнами 01–04 и 06–10 — это уже дешёвое время."""
    moment = datetime(2026, 9, 4, 5, 0, tzinfo=UTC)
    assert next_offpeak_start(moment) == moment


def test_weekend_never_waits():
    """Суббота 02:00 попадает в окно по часам, но выходные и так off-peak —
    откладывать задачу значило бы ждать двух часов даром."""
    saturday = datetime(2026, 9, 5, 2, 0, tzinfo=UTC)
    assert saturday.weekday() == 5
    assert next_offpeak_start(saturday) == saturday


# --------------------------------------------------------------------------
# Очередь
# --------------------------------------------------------------------------


def test_database_is_created_on_first_use(tmp_path):
    path = tmp_path / ".cache" / "pending_scans.sqlite3"
    assert not path.exists()

    with PendingScans(path):
        pass

    assert path.exists()


def test_queued_scan_survives_reopening(tmp_path):
    path = tmp_path / "pending.sqlite3"
    scheduled = datetime(2026, 9, 4, 4, 0, tzinfo=UTC)

    with PendingScans(path) as queue:
        queue.add("нужен парсер PDF-таблиц", options(), scheduled_at=scheduled)

    with PendingScans(path) as reopened:
        pending = reopened.all()

    assert len(pending) == 1
    assert pending[0].query == "нужен парсер PDF-таблиц"
    assert pending[0].scheduled_at == scheduled
    assert pending[0].options.off_peak is True


def test_options_survive_the_round_trip(queue):
    queue.add(
        "задача",
        options(max_candidates=7, audit_limit=3, report_limit=2, refresh=True),
        scheduled_at=MONDAY_NOON,
    )

    restored = queue.all()[0].options

    assert restored.max_candidates == 7
    assert restored.audit_limit == 3
    assert restored.report_limit == 2
    assert restored.refresh is True


def test_due_returns_only_tasks_whose_time_has_come(queue):
    queue.add("рано", options(), scheduled_at=datetime(2026, 9, 4, 10, 0, tzinfo=UTC))
    queue.add("пора", options(), scheduled_at=datetime(2026, 9, 4, 4, 0, tzinfo=UTC))

    due = queue.due(datetime(2026, 9, 4, 5, 0, tzinfo=UTC))

    assert [task.query for task in due] == ["пора"]


def test_due_includes_the_exact_moment(queue):
    scheduled = datetime(2026, 9, 4, 4, 0, tzinfo=UTC)
    queue.add("ровно", options(), scheduled_at=scheduled)

    assert len(queue.due(scheduled)) == 1


def test_due_is_ordered_oldest_first(queue):
    queue.add("вторая", options(), scheduled_at=datetime(2026, 9, 4, 5, 0, tzinfo=UTC))
    queue.add("первая", options(), scheduled_at=datetime(2026, 9, 4, 4, 0, tzinfo=UTC))

    due = queue.due(datetime(2026, 9, 4, 12, 0, tzinfo=UTC))

    assert [task.query for task in due] == ["первая", "вторая"]


def test_remove_takes_the_task_out_of_the_queue(queue):
    queue.add("задача", options(), scheduled_at=MONDAY_NOON)
    task = queue.all()[0]

    queue.remove(task.id)

    assert queue.all() == []


def test_empty_queue_is_not_an_error(queue):
    assert queue.all() == []
    assert queue.due(MONDAY_NOON) == []
