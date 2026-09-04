"""Харнесс оценки: загрузка golden-set, метрики recall, прогон набора.

Ни моделей, ни сети: `run_scan` подменяется фейком, который отдаёт заранее
собранный `ScanOutcome`. Проверяется арифметика метрики и поведение набора,
а не качество поиска — качество измеряется живым прогоном дня 10.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from conftest import screening_run
from scout.config import MissingCredential
from scout.evaluate import (
    RECALL_TARGET,
    EvalError,
    GoldenCase,
    evaluate,
    evaluate_case,
    load_cases,
    recall,
    write_report,
)
from scout.github import GitHubAuth
from scout.log import RunLogger
from scout.pipeline import ScanOutcome, ScanStatus
from scout.schemas import ModelName, PricingWindow, ScanOptions, ScanRequest, TokenUsage
from test_queries import make_intent
from test_screen import candidate

CASE_OK = {
    "query_text": "нужен парсер PDF-таблиц на Python",
    "expected_repos": ["camelot-dev/camelot", "jsvine/pdfplumber"],
    "expected_verdict": "USE",
    "note": "две общеизвестные библиотеки под ровно эту задачу",
}

TRAP = {
    "query_text": "нужен генератор отчётов по мясокомбинату в формате 1С",
    "expected_repos": [],
    "expected_verdict": "BUILD",
    "note": "ловушка: готового решения нет",
}


def write_case(directory, slug, payload):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def golden(tmp_path):
    directory = tmp_path / "golden"
    write_case(directory, "pdf-tables", CASE_OK)
    write_case(directory, "meat-plant-1c", TRAP)
    return directory


class FakeClient:
    """Клиент GitHub ровно в той части, которой пользуется прогон набора."""

    def __init__(self, search_calls: int = 0):
        self.search_calls = search_calls

    def reset_search_budget(self):
        self.search_calls = 0


class Recorder(RunLogger):
    """Логгер, который не печатает: тесты смотрят на события, а не на stderr."""

    def __init__(self):
        super().__init__("evaltest0000")
        self.events: list[tuple[str, dict]] = []

    def info(self, event, **fields):
        self.events.append((event, fields))

    def error(self, event, **fields):
        self.events.append((event, fields))

    def names(self):
        return [event for event, _ in self.events]


# --------------------------------------------------------------------------
# Загрузка набора
# --------------------------------------------------------------------------


def test_cases_load_with_slug_from_the_file_name(golden):
    cases = load_cases(golden)

    assert [case.slug for case in cases] == ["meat-plant-1c", "pdf-tables"]
    assert cases[1].expected_repos == ["camelot-dev/camelot", "jsvine/pdfplumber"]


def test_cases_are_sorted_so_two_runs_line_up(golden):
    """Порядок задач фиксирован: иначе таблицы двух прогонов не сравнить построчно."""
    write_case(golden, "aaa-first", CASE_OK)

    assert next(case.slug for case in load_cases(golden)) == "aaa-first"


def test_trap_case_is_recognised_by_the_empty_expected_list(golden):
    trap = next(case for case in load_cases(golden) if case.slug == "meat-plant-1c")

    assert trap.is_trap is True


def test_broken_json_names_the_file(tmp_path):
    directory = tmp_path / "golden"
    directory.mkdir()
    (directory / "broken.json").write_text("{не json", encoding="utf-8")

    with pytest.raises(EvalError, match=r"broken\.json"):
        load_cases(directory)


def test_missing_field_is_a_data_error_not_a_crash(tmp_path):
    directory = tmp_path / "golden"
    incomplete = {key: value for key, value in CASE_OK.items() if key != "expected_verdict"}
    write_case(directory, "incomplete", incomplete)

    with pytest.raises(EvalError, match="expected_verdict"):
        load_cases(directory)


def test_unknown_field_is_rejected(tmp_path):
    """`extra="forbid"`: опечатка в имени поля не должна пройти молча."""
    directory = tmp_path / "golden"
    write_case(directory, "typo", {**CASE_OK, "expected_repo": ["x/y"]})

    with pytest.raises(EvalError):
        load_cases(directory)


def test_empty_directory_is_an_error(tmp_path):
    empty = tmp_path / "golden"
    empty.mkdir()

    with pytest.raises(EvalError):
        load_cases(empty)


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(EvalError):
        load_cases(tmp_path / "нет-такого")


# --------------------------------------------------------------------------
# Метрика
# --------------------------------------------------------------------------


def test_recall_counts_the_share_of_expected_repos():
    assert recall(["a/one", "a/two"], ["a/one", "b/other"]) == 0.5


def test_recall_is_one_when_everything_expected_is_there():
    assert recall(["a/one"], ["b/other", "a/one"]) == 1.0


def test_recall_ignores_letter_case():
    """GitHub нечувствителен к регистру, а эталон пишет человек руками."""
    assert recall(["Camelot-Dev/Camelot"], ["camelot-dev/camelot"]) == 1.0


def test_trap_scores_one_when_nobody_passed():
    assert recall([], []) == 1.0


def test_trap_scores_zero_when_somebody_passed():
    """Иначе метрику выгодно обманывать, выдавая пятёрку кандидатов всегда."""
    assert recall([], ["someone/anything"]) == 0.0


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------


def outcome(*, passed=(1, 2), candidates=(1, 2, 3), status=ScanStatus.OK, partial=False, cost=0.01):
    """`ScanOutcome`, какой отдал бы настоящий конвейер."""
    request = ScanRequest(
        request_id=uuid4(),
        query_text="нужен парсер PDF-таблиц на Python",
        created_at=datetime.now(UTC),
        options=ScanOptions(),
    )
    return ScanOutcome(
        request=request,
        status=status,
        intent=make_intent(request_id=request.request_id),
        intent_usage=TokenUsage(
            model=ModelName.FLASH,
            input_tokens=900,
            output_tokens=210,
            cost_usd=cost,
            pricing_window=PricingWindow.OFF_PEAK,
        ),
        queries_used=["pdf table extraction language:python"],
        candidates=[candidate(n) for n in candidates],
        screening=screening_run(request.request_id, passed=passed),
        partial=partial,
        duration_sec=42.0,
    )


def runner_returning(*outcomes):
    """Фейковый конвейер: по одному исходу на задачу, по порядку."""
    queue = list(outcomes)

    def runner(request, *, log, github=None, **kwargs):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return runner


def case(slug="pdf-tables", **overrides):
    return GoldenCase(slug=slug, **{**CASE_OK, **overrides})


def test_expected_repo_that_passed_screening_counts_towards_recall_at_10():
    log = Recorder()
    result = evaluate_case(
        case(expected_repos=["owner1/repo1"]),
        options=ScanOptions(),
        log=log,
        runner=runner_returning(outcome(passed=(1, 2))),
    )

    assert result.recall_at_10 == 1.0
    assert result.passed == ["owner1/repo1", "owner2/repo2"]


def test_expected_repo_found_but_screened_out_shows_the_gap():
    """Ровно тот диагноз, ради которого считаются оба среза: поиск нашёл,
    Слой 1 отбросил — значит болит промпт скрининга, а не `QUERIES.md`."""
    result = evaluate_case(
        case(expected_repos=["owner3/repo3"]),
        options=ScanOptions(),
        log=Recorder(),
        runner=runner_returning(outcome(passed=(1, 2), candidates=(1, 2, 3))),
    )

    assert result.recall_at_50 == 1.0
    assert result.recall_at_10 == 0.0


def test_expected_repo_never_found_scores_zero_on_both():
    result = evaluate_case(
        case(expected_repos=["nobody/knows"]),
        options=ScanOptions(),
        log=Recorder(),
        runner=runner_returning(outcome()),
    )

    assert result.recall_at_50 == 0.0
    assert result.recall_at_10 == 0.0


def test_failed_case_does_not_stop_the_set():
    """Сбой одной задачи стоит одной задачи — то же правило, что на дне 8."""

    def explode(request, *, log, github=None, **kwargs):
        if "мясокомбинат" in request.query_text:
            raise RuntimeError("сеть отвалилась")
        return outcome()

    log = Recorder()
    run = evaluate(
        [case(), GoldenCase(slug="meat-plant-1c", **TRAP)],
        log=log,
        github=FakeClient(),
        runner=explode,
    )

    assert [result.status for result in run.results] == ["ok", "failed"]
    assert "RuntimeError" in run.results[1].error
    assert "eval_case_failed" in log.names()


def test_failed_case_is_excluded_from_the_average():
    """Иначе обрыв сети выглядел бы как плохое качество поиска."""

    def explode(request, *, log, github=None, **kwargs):
        if "мясокомбинат" in request.query_text:
            raise RuntimeError("сеть отвалилась")
        return outcome(passed=(1, 2))

    run = evaluate(
        [case(expected_repos=["owner1/repo1"]), GoldenCase(slug="trap", **TRAP)],
        log=Recorder(),
        github=FakeClient(),
        runner=explode,
    )

    summary = run.summary()
    assert summary["cases_measured"] == 1
    assert summary["cases_failed"] == 1
    assert summary["recall_at_10"] == 1.0


def test_rejected_key_stops_the_whole_set():
    """Двадцать пять одинаковых отказов — час впустую: ключ чинится один раз."""

    def explode(request, *, log, github=None, **kwargs):
        raise GitHubAuth("GitHub отклонил токен (401)")

    with pytest.raises(GitHubAuth):
        evaluate([case()], log=Recorder(), github=FakeClient(), runner=explode)


def test_missing_key_stops_the_whole_set():
    def explode(request, *, log, github=None, **kwargs):
        raise MissingCredential("нет DEEPSEEK_API_KEY")

    with pytest.raises(MissingCredential):
        evaluate([case()], log=Recorder(), github=FakeClient(), runner=explode)


def test_search_budget_is_reset_between_cases():
    """Запрет 5 CLAUDE.md ограничивает скан, а не набор: без сброса счётчика
    вторая задача упёрлась бы в лимит десяти поисковых запросов."""

    seen = []

    def runner(request, *, log, github=None, **kwargs):
        seen.append(github.search_calls)
        github.search_calls += 7
        return outcome()

    evaluate([case("a"), case("b")], log=Recorder(), github=FakeClient(9), runner=runner)

    assert seen == [0, 0]


def test_summary_averages_recall_and_sums_cost():
    run = evaluate(
        [case("a", expected_repos=["owner1/repo1"]), case("b", expected_repos=["nobody/knows"])],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome(cost=0.02)),
    )

    summary = run.summary()
    assert summary["recall_at_10"] == 0.5
    assert summary["cost_usd_total"] == pytest.approx(0.04)
    assert summary["duration_sec_median"] == 42.0


def test_summary_flags_whether_the_target_is_met():
    """Цель недели 2 — не украшение отчёта, а развилка: чинить поиск или строить Слой 2."""
    met = evaluate(
        [case(expected_repos=["owner1/repo1"])],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome()),
    ).summary()
    missed = evaluate(
        [case(expected_repos=["nobody/knows"])],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome()),
    ).summary()

    assert met["recall_at_10"] >= RECALL_TARGET
    assert met["target_met"] is True
    assert missed["target_met"] is False


def test_partial_runs_are_counted_in_the_summary():
    run = evaluate(
        [case()],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome(partial=True)),
    )

    assert run.summary()["partial_runs"] == 1


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------


def test_report_is_written_as_json(tmp_path):
    run = evaluate(
        [case()], log=Recorder(), github=FakeClient(), runner=runner_returning(outcome())
    )

    path = write_report(run, tmp_path / "eval")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.startswith(run.generated_at.date().isoformat())
    assert payload["summary"]["cases_measured"] == 1
    assert payload["cases"][0]["slug"] == "pdf-tables"
    assert payload["options"]["max_candidates"] == 50


def test_second_run_on_the_same_day_does_not_overwrite_the_first(tmp_path):
    """Замер до и после правки промпта делается в один день — и оба нужны."""
    run = evaluate(
        [case()], log=Recorder(), github=FakeClient(), runner=runner_returning(outcome())
    )

    first = write_report(run, tmp_path / "eval")
    second = write_report(run, tmp_path / "eval")

    assert first != second
    assert first.exists() and second.exists()
