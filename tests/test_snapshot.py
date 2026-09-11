"""Снимок выдачи и прогон по нему: то, без чего замер A/B меряет погоду.

Ни моделей, ни сети. Проверяется ровно одно свойство, ради которого файл
существует: два прогона по одному снимку получают побайтово один и тот же вход,
и разойтись могут только тем слоем, который между ними правят. 2026-09-11 это
свойство не выполнялось — состав выдачи совпал у одной задачи из двенадцати, —
и весь замер `blocking` оказался непригоден.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from conftest import screening_run
from scout import cli, pipeline, snapshot
from scout import evaluate as evaluate_module
from scout.pipeline import ScanOutcome, ScanStatus, run_scan
from scout.schemas import (
    DroppedCandidate,
    DropStage,
    ModelName,
    ScanOptions,
    ScanRequest,
)
from scout.snapshot import FrozenCase, ReplayStage, SnapshotIncomplete
from test_evaluate import CASE_OK, FakeClient, Recorder, case, outcome
from test_queries import make_intent
from test_screen import candidate

QUERY = "нужен парсер PDF-таблиц на Python"


def frozen_case(slug="pdf-tables", *, candidates=(1, 2, 3), passed=(1, 2), screening=True):
    """Снимок задачи, какой записал бы прогон."""
    request_id = uuid4()
    return FrozenCase(
        slug=slug,
        query_text=QUERY,
        recorded_at=datetime.now(UTC),
        intent=make_intent(request_id=request_id),
        queries_used=["pdf table extraction language:python"],
        candidates=[candidate(n) for n in candidates],
        screening=screening_run(request_id, passed=passed).result if screening else None,
    )


class Sentinel:
    """Клиент GitHub, которого в этих тестах никто не имеет права вызвать."""

    def __getattr__(self, name):
        raise AssertionError(f"прогон по снимку полез в GitHub: {name}")


@pytest.fixture
def no_layer_two(monkeypatch, tmp_path):
    """Слой 2 подменён пустым результатом: здесь проверяется его вход, не он сам."""
    monkeypatch.setattr(pipeline, "CACHE_PATH", tmp_path / "audits.sqlite3")
    monkeypatch.setattr(pipeline, "DeepSeekClient", lambda **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "audit_candidates",
        lambda candidates, intent, *, client, github, cache=None, logger=None, **kwargs: (
            [],
            [],
            {},
        ),
    )


def scan(frozen, *, stage=ReplayStage.SCREENING, options=None, log=None, github=None):
    request = ScanRequest(
        request_id=uuid4(),
        query_text=QUERY,
        created_at=datetime.now(UTC),
        options=options or ScanOptions(),
    )
    return run_scan(
        request,
        log=log or Recorder(),
        github=github or Sentinel(),
        frozen=frozen,
        replay_through=stage,
    )


# --------------------------------------------------------------------------
# Файл снимка
# --------------------------------------------------------------------------


def test_snapshot_keeps_a_line_per_case(tmp_path):
    path = tmp_path / "run.snapshot.jsonl"

    snapshot.write_case(path, frozen_case("first"))
    snapshot.write_case(path, frozen_case("second"))

    assert list(snapshot.read(path)) == ["first", "second"]


def test_frozen_case_round_trips_through_the_file(tmp_path):
    """Кандидат обязан вернуться тем же объектом: он и есть предмет фиксации."""
    path = tmp_path / "run.snapshot.jsonl"
    original = frozen_case()

    snapshot.write_case(path, original)

    restored = snapshot.read(path)["pdf-tables"]
    assert restored.candidates == original.candidates
    assert restored.screening == original.screening
    assert restored.queries_used == original.queries_used


def test_torn_last_line_does_not_cost_the_whole_snapshot(tmp_path):
    """Тот же случай, что у журнала: процесс убили посреди записи строки."""
    path = tmp_path / "run.snapshot.jsonl"
    snapshot.write_case(path, frozen_case("intact"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"slug": "torn", "candi')

    assert list(snapshot.read(path)) == ["intact"]


def test_missing_snapshot_says_so_instead_of_returning_nothing(tmp_path):
    """Пустой словарь означал бы «замораживать нечего» и увёл бы прогон в живой
    поиск — то есть молча вернул бы ровно то, от чего снимок и заводится."""
    with pytest.raises(SnapshotIncomplete):
        snapshot.read(tmp_path / "нет-такого.jsonl")


def test_search_drops_are_kept_and_later_ones_are_not(tmp_path):
    """Потери Слоя 1 и Слоя 2 принадлежат слоям и случатся заново."""
    result = snapshot.freeze(
        "pdf-tables",
        outcome_with_drops(
            DroppedCandidate(full_name="a/b", stage=DropStage.SEARCH, reason="репозиторий удалён"),
            DroppedCandidate(full_name="c/d", stage=DropStage.SCREENING, reason="схема"),
        ),
    )

    assert [item.full_name for item in result.dropped] == ["a/b"]


def outcome_with_drops(*dropped):
    base = outcome()
    return ScanOutcome(
        request=base.request,
        status=base.status,
        intent=base.intent,
        intent_usage=base.intent_usage,
        queries_used=base.queries_used,
        candidates=base.candidates,
        screening=base.screening,
        dropped=list(dropped),
    )


def test_snapshot_of_a_case_without_screening_is_marked_incomplete():
    """Задача, где поиск вернул ноль кандидатов, до Слоя 1 не дошла."""
    empty = ScanOutcome(
        request=outcome().request,
        status=ScanStatus.NO_CANDIDATES,
        intent=outcome().intent,
        intent_usage=outcome().intent_usage,
    )

    frozen = snapshot.freeze("trap-kz-tax-form-200", empty)

    assert frozen.has(ReplayStage.SEARCH) is True
    assert frozen.has(ReplayStage.SCREENING) is False


# --------------------------------------------------------------------------
# Конвейер по снимку
# --------------------------------------------------------------------------


def test_replay_does_not_search_and_does_not_ask_for_an_intent(no_layer_two, monkeypatch):
    """Оба верхних этапа — источник того самого разброса, который чинится."""

    def forbidden(*args, **kwargs):
        raise AssertionError("прогон по снимку выполнил этап, который заморожен")

    monkeypatch.setattr(pipeline, "extract_intent", forbidden)
    monkeypatch.setattr(pipeline, "build_query_set", forbidden)
    monkeypatch.setattr(pipeline, "collect_candidates", forbidden)
    monkeypatch.setattr(pipeline, "screen", forbidden)

    result = scan(frozen_case())

    assert [c.full_name for c in result.candidates] == [
        "owner1/repo1",
        "owner2/repo2",
        "owner3/repo3",
    ]
    assert result.passed_full_names == ["owner1/repo1", "owner2/repo2"]


def test_replay_through_search_runs_layer_one_live(no_layer_two, monkeypatch):
    """Замер промпта Слоя 1: кандидаты те же, скрининг настоящий."""
    seen = {}

    def fake_screen(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        seen["candidates"] = [c.full_name for c in candidates]
        return screening_run(request_id, passed=(2,))

    monkeypatch.setattr(pipeline, "screen", fake_screen)

    result = scan(frozen_case(), stage=ReplayStage.SEARCH)

    assert seen["candidates"] == ["owner1/repo1", "owner2/repo2", "owner3/repo3"]
    assert result.passed_full_names == ["owner2/repo2"]


def test_two_replays_get_the_same_input_whatever_layer_two_says(no_layer_two):
    """Свойство, ради которого всё это делается: разойтись плечи могут только
    Слоем 2. 2026-09-10 два прогона одного набора совпали составом выдачи
    у одной задачи из двенадцати, и прирост recall@5 оказался неотличим от шума."""
    frozen = frozen_case()

    first = scan(frozen)
    second = scan(frozen)

    assert [c.full_name for c in first.candidates] == [c.full_name for c in second.candidates]
    assert first.passed_full_names == second.passed_full_names


def test_replay_does_not_bill_for_the_frozen_layers(no_layer_two):
    """Замороженные слои сегодня не вызывались — платить за них нельзя.

    Иначе плечо A/B показывало бы счёт чужого прогона, и сравнивать стоимость
    двух плеч было бы не с чем.
    """
    result = scan(frozen_case())

    assert result.intent_usage.cost_usd == 0.0
    assert result.screening_usage.cost_usd == 0.0
    assert result.total_cost_usd == 0.0


def test_replayed_intent_belongs_to_this_run(no_layer_two):
    """`request_id` идентифицирует прогон, а не выдачу: чужой в интенте связал бы
    провенанс сегодняшнего аудита с прогоном, которого сегодня не было."""
    result = scan(frozen_case())

    assert result.intent.request_id == result.request.request_id
    assert result.screening.result.request_id == result.request.request_id


def test_audit_limit_of_this_run_narrows_the_frozen_passed(no_layer_two):
    """Снимок говорит, кто прошёл; сколько из них оплачивать — решение запуска."""
    result = scan(frozen_case(passed=(1, 2, 3)), options=ScanOptions(audit_limit=2))

    assert result.screening.result.passed == [1, 2, 3][:2]


def test_replay_through_screening_without_a_frozen_layer_one_is_refused(no_layer_two):
    """Молча съехать на живой Слой 1 нельзя: прогон бы состоялся, стоил денег
    и дал бы несравнимое плечо — ровно тот отказ, который надо заметить сразу."""
    with pytest.raises(SnapshotIncomplete, match="Слоя 1"):
        scan(frozen_case(screening=False))


def test_a_recorded_run_replays_from_its_own_file(no_layer_two, tmp_path):
    """Вся цепочка целиком: прогон → снимок на диск → прогон по снимку.

    По частям это проверено выше; здесь важен стык — то, что записал `freeze`,
    читается `read` и принимается конвейером без единой правки руками.
    """
    path = tmp_path / "run.snapshot.jsonl"
    recorded = outcome(candidates=(1, 2, 3), passed=(1, 2))
    snapshot.write_case(path, snapshot.freeze("pdf-tables", recorded))

    result = scan(snapshot.read(path)["pdf-tables"])

    assert [c.full_name for c in result.candidates] == [c.full_name for c in recorded.candidates]
    assert result.passed_full_names == recorded.passed_full_names


def test_replay_is_visible_in_the_log(no_layer_two):
    log = Recorder()
    scan(frozen_case(), log=log)

    events = dict(log.events)
    assert events["start"]["replay"] == "screening"
    assert events["replay_prepared"]["candidates"] == 3


# --------------------------------------------------------------------------
# Прогон набора: запись и воспроизведение
# --------------------------------------------------------------------------


def test_freeze_writes_a_case_as_soon_as_it_is_measured(tmp_path):
    path = tmp_path / "run.snapshot.jsonl"

    evaluate_module.evaluate(
        [case("first"), case("second")],
        log=Recorder(),
        github=FakeClient(),
        runner=lambda request, **kwargs: outcome(),
        freeze_to=path,
    )

    assert [json.loads(line)["slug"] for line in path.read_text().splitlines()] == [
        "first",
        "second",
    ]


def test_replayed_set_feeds_each_case_its_own_snapshot(tmp_path):
    seen = {}

    def runner(request, *, log, github=None, frozen=None, replay_through=None):
        seen[frozen.slug] = replay_through
        return outcome()

    evaluate_module.evaluate(
        [case("first"), case("second")],
        log=Recorder(),
        github=FakeClient(),
        runner=runner,
        replay={"first": frozen_case("first"), "second": frozen_case("second")},
        replay_through=ReplayStage.SEARCH,
    )

    assert seen == {"first": ReplayStage.SEARCH, "second": ReplayStage.SEARCH}


def test_replay_is_recorded_in_the_run_file():
    """Иначе честный замер на зафиксированной выдаче и случайный, где разошёлся
    поиск, лежат в `eval/` неразличимыми файлами."""
    run = evaluate_module.evaluate(
        [case("pdf-tables")],
        log=Recorder(),
        github=FakeClient(),
        runner=lambda request, **kwargs: outcome(),
        replay={"pdf-tables": frozen_case()},
        replay_source="eval/2026-09-11.snapshot.jsonl",
    )

    assert run.as_json()["replay"] == {
        "snapshot": "eval/2026-09-11.snapshot.jsonl",
        "through": "screening",
    }


def test_a_case_absent_from_the_snapshot_stops_the_set(tmp_path):
    """Молча уйти в живой поиск нельзя: плечо стало бы несравнимым, а заметить
    это можно было бы только по готовым числам, когда деньги уже потрачены."""
    with pytest.raises(evaluate_module.EvalError, match="kazakh-nlp"):
        evaluate_module.evaluate(
            [case("kazakh-nlp")],
            log=Recorder(),
            github=FakeClient(),
            runner=lambda request, **kwargs: outcome(),
            replay={"pdf-tables": frozen_case()},
        )


def test_a_live_run_leaves_the_replay_field_empty():
    run = evaluate_module.evaluate(
        [case()], log=Recorder(), github=FakeClient(), runner=lambda request, **kwargs: outcome()
    )

    assert run.as_json()["replay"] is None


def test_a_live_run_does_not_get_replay_arguments():
    """Подмены конвейера принимают сигнатуру `run_scan`; лишний именованный
    аргумент в живом прогоне заставил бы править каждую из них."""

    def strict_runner(request, *, log, github=None):
        return outcome()

    run = evaluate_module.evaluate(
        [case()], log=Recorder(), github=FakeClient(), runner=strict_runner
    )

    assert run.results[0].error is None


def test_snapshot_with_a_missing_case_is_named_before_anything_is_paid(tmp_path):
    path = tmp_path / "run.snapshot.jsonl"
    snapshot.write_case(path, frozen_case("pdf-tables"))
    cases = snapshot.read(path)

    assert snapshot.missing(cases, ["pdf-tables", "kazakh-nlp"], ReplayStage.SCREENING) == [
        "kazakh-nlp"
    ]


def test_case_frozen_only_to_search_counts_as_missing_for_layer_two(tmp_path):
    path = tmp_path / "run.snapshot.jsonl"
    snapshot.write_case(path, frozen_case("pdf-tables", screening=False))
    cases = snapshot.read(path)

    assert snapshot.missing(cases, ["pdf-tables"], ReplayStage.SEARCH) == []
    assert snapshot.missing(cases, ["pdf-tables"], ReplayStage.SCREENING) == ["pdf-tables"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def golden_dir(tmp_path):
    directory = tmp_path / "golden"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pdf-tables.json").write_text(
        json.dumps(CASE_OK, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def test_cli_refuses_a_snapshot_without_the_asked_case(tmp_path, capsys):
    snap = tmp_path / "run.snapshot.jsonl"
    snapshot.write_case(snap, frozen_case("kazakh-nlp"))

    code = cli.main(
        ["eval", "--golden", str(golden_dir(tmp_path)), "--replay", str(snap), "--dry-run"]
    )

    assert code == 2
    assert "pdf-tables" in capsys.readouterr().err


def test_cli_refuses_to_read_and_write_the_same_snapshot(tmp_path, capsys):
    """Иначе прогон дописывал бы файл, из которого сам же читает."""
    snap = tmp_path / "run.snapshot.jsonl"
    snapshot.write_case(snap, frozen_case("pdf-tables"))

    code = cli.main(
        [
            "eval",
            "--golden",
            str(golden_dir(tmp_path)),
            "--replay",
            str(snap),
            "--freeze",
            str(snap),
            "--dry-run",
        ]
    )

    assert code == 2
    assert "один файл" in capsys.readouterr().err


def test_dry_run_checks_the_snapshot_without_spending(tmp_path, capsys):
    """Проверка снимка обязана быть бесплатной: `--dry-run` для того и есть."""
    snap = tmp_path / "run.snapshot.jsonl"
    snapshot.write_case(snap, frozen_case("pdf-tables"))

    code = cli.main(
        ["eval", "--golden", str(golden_dir(tmp_path)), "--replay", str(snap), "--dry-run"]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "заморожены поиск и Слой 1" in out


def test_frozen_token_counters_stay_in_the_record(no_layer_two):
    """Счётчики записанного Слоя 1 остаются в снимке, а не переезжают в прогон."""
    frozen = frozen_case()
    assert frozen.screening.token_usage.input_tokens == 7500

    result = scan(frozen)

    assert result.screening_usage.input_tokens == 0
    assert result.screening_usage.model is ModelName.FLASH
