"""Харнесс оценки: загрузка golden-set, метрики recall, прогон набора.

Ни моделей, ни сети: `run_scan` подменяется фейком, который отдаёт заранее
собранный `ScanOutcome`. Проверяется арифметика метрики и поведение набора,
а не качество поиска — качество измеряется живым прогоном дня 10.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from conftest import screening_run
from scout.config import MissingCredential
from scout.evaluate import (
    RECALL_TARGET,
    CaseResult,
    EvalError,
    GoldenCase,
    evaluate,
    evaluate_case,
    load_cases,
    read_journal,
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


# --------------------------------------------------------------------------
# Журнал прогона: остановка не должна сжигать оплаченное
# --------------------------------------------------------------------------


def test_case_result_survives_a_round_trip_through_json():
    """Строка журнала — это оплаченный замер; поднимать его надо тем же объектом."""
    original = CaseResult(
        slug="pdf-tables",
        query_text="нужен парсер PDF-таблиц",
        status="ok",
        expected=["camelot-dev/camelot"],
        found=["camelot-dev/camelot", "jsvine/pdfplumber"],
        passed=["camelot-dev/camelot"],
        recall_at_10=1.0,
        recall_at_50=0.5,
        cost_usd=0.0123,
        duration_sec=81.9,
        partial=True,
    )

    assert CaseResult.from_json(original.as_json()).as_json() == original.as_json()


def test_trap_result_round_trips_with_null_ceiling():
    """У ловушки `recall_at_50` — None, и он обязан остаться None, а не стать нулём:
    ноль means «поиск не нашёл», None — «искать было нечего»."""
    trap = CaseResult(slug="trap", query_text="…", status="ok", recall_at_10=1.0)

    assert CaseResult.from_json(trap.as_json()).recall_at_50 is None


def test_journal_gets_a_line_per_case_as_it_goes(tmp_path):
    """Не в конце набора: полтора часа прогона не должны висеть на одном вызове
    `write_report` в самом хвосте."""
    journal = tmp_path / "run.jsonl"

    evaluate(
        [case("first"), case("second")],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome()),
        journal=journal,
    )

    assert [json.loads(line)["slug"] for line in journal.read_text().splitlines()] == [
        "first",
        "second",
    ]


def test_journal_keeps_finished_cases_when_the_run_is_killed(tmp_path):
    """Ровно случай 2026-09-09: процесс убит на середине набора. Всё, что успело
    отработать, обязано лежать на диске — иначе деньги потрачены впустую."""
    journal = tmp_path / "run.jsonl"

    def killed_on_the_second(request, **kwargs):
        if request.query_text == "убить прогон здесь":
            raise KeyboardInterrupt
        return outcome()

    with pytest.raises(KeyboardInterrupt):
        evaluate(
            [case("done"), case("killed", query_text="убить прогон здесь")],
            log=Recorder(),
            github=FakeClient(),
            runner=killed_on_the_second,
            journal=journal,
        )

    assert [json.loads(line)["slug"] for line in journal.read_text().splitlines()] == ["done"]


def test_resume_reuses_measured_cases_and_pays_only_for_the_rest(tmp_path):
    """Догон недостающего: измеренная задача не должна вызывать модель второй раз."""
    journal = tmp_path / "run.jsonl"
    evaluate(
        [case("already")],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome()),
        journal=journal,
    )

    ran: list[str] = []

    def recording(request, **kwargs):
        ran.append(request.query_text)
        return outcome()

    run = evaluate(
        [case("already"), case("fresh", query_text="новая задача")],
        log=Recorder(),
        github=FakeClient(),
        runner=recording,
        journal=journal,
    )

    assert ran == ["новая задача"]
    assert [result.slug for result in run.results] == ["already", "fresh"]


def test_failed_case_is_not_journalled_so_the_retry_still_happens(tmp_path):
    """У упавшей задачи нет замера. Записать её значило бы навсегда закрепить
    обрыв сети как результат и никогда его не переспросить."""
    journal = tmp_path / "run.jsonl"

    def always_broken(request, **kwargs):
        raise RuntimeError("сеть отвалилась")

    evaluate(
        [case("broken")],
        log=Recorder(),
        github=FakeClient(),
        runner=always_broken,
        journal=journal,
    )

    assert not journal.exists() or journal.read_text().strip() == ""


def test_torn_last_line_does_not_cost_the_whole_journal(tmp_path):
    """Процесс убили посреди записи. Оборванный хвост пропускается, а всё,
    что записалось целиком, остаётся оплаченным и переиспользуется."""
    journal = tmp_path / "run.jsonl"
    evaluate(
        [case("intact")],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome()),
        journal=journal,
    )
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"slug": "torn", "query_text": "обор')

    assert list(read_journal(journal)) == ["intact"]


def test_resume_with_nothing_left_to_do_needs_no_github_client(tmp_path):
    """Возобновление полностью закрытого набора не должно требовать токена:
    ходить в сеть незачем, считать нечего."""
    journal = tmp_path / "run.jsonl"
    evaluate(
        [case("only")],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome()),
        journal=journal,
    )

    run = evaluate([case("only")], log=Recorder(), journal=journal)

    assert run.summary()["cases_measured"] == 1


# --------------------------------------------------------------------------
# Сам набор в репозитории
# --------------------------------------------------------------------------

SHIPPED = Path(__file__).parent / "golden"


def test_shipped_golden_set_loads():
    """Битый JSON в наборе должен падать здесь, а не на середине платного прогона.

    Границы — из `ROADMAP.md` → «Оценка качества»: меньше двадцати задач дают шаг
    измерения крупнее целевой разницы, больше тридцати — час прогона и лишний доллар
    без прироста разрешения.
    """
    cases = load_cases(SHIPPED)

    assert 20 <= len(cases) <= 30


def test_shipped_set_has_two_or_three_traps():
    """Без ловушек метрику выгодно обманывать: пять строк в отчёте всегда лучше
    пустого ответа, если пустой ответ не засчитывается никогда. Больше трёх — уже
    перекос: набор начинает мерить осторожность вместо полноты (`ROADMAP.md`)."""
    traps = [case for case in load_cases(SHIPPED) if case.is_trap]

    assert 2 <= len(traps) <= 3


def test_shipped_cases_have_one_to_three_reference_repos():
    """`ROADMAP.md` задаёт 1–3 эталона на задачу. Ноль — это необъявленная ловушка,
    больше трёх — задача без единственного правильного ответа, и recall на ней
    измеряет широту выдачи, а не попадание."""
    for case in load_cases(SHIPPED):
        if case.is_trap:
            continue
        assert 1 <= len(case.expected_repos) <= 3, f"{case.slug}: {len(case.expected_repos)}"


def test_shipped_expected_repos_have_no_duplicates_inside_a_case():
    """Повтор эталона внутри задачи тихо занижает знаменатель: `recall` сравнивает
    множества, и дубль сократил бы |expected| без единого признака в отчёте."""
    for case in load_cases(SHIPPED):
        lowered = [full_name.lower() for full_name in case.expected_repos]

        assert len(lowered) == len(set(lowered)), f"{case.slug}: дубль в expected_repos"


def test_shipped_queries_are_unique():
    """Две одинаковые формулировки — это одна задача, посчитанная дважды."""
    queries = [case.query_text for case in load_cases(SHIPPED)]

    assert len(queries) == len(set(queries))


def test_shipped_expected_repos_look_like_full_names():
    """`owner/name` — то, что сравнивается с выдачей; ссылка или имя без владельца
    молча дали бы recall 0 и выглядели бы как плохой поиск."""
    for case in load_cases(SHIPPED):
        for full_name in case.expected_repos:
            assert full_name.count("/") == 1, f"{case.slug}: {full_name}"
            assert not full_name.startswith("http"), f"{case.slug}: {full_name}"


# --------------------------------------------------------------------------
# Находки пилотного прогона 2026-09-05
# --------------------------------------------------------------------------


def test_trap_has_no_search_ceiling():
    """Ловушка не участвует в recall@50: там нечего искать.

    Пилот 2026-09-05 дал recall@10 = 0,60 при recall@50 = 0,48 — потолок ниже
    того, что он ограничивает. Причина: ловушке ставился ноль за то, что поиск
    вернул кандидатов, хотя вернуть их он обязан — иначе Слою 1 некого отвергать.
    """
    result = evaluate_case(
        GoldenCase(slug="trap", **TRAP),
        options=ScanOptions(),
        log=Recorder(),
        runner=runner_returning(outcome(passed=())),
    )

    assert result.recall_at_10 == 1.0
    assert result.recall_at_50 is None


def test_ceiling_never_falls_below_the_metric_it_bounds():
    """Инвариант: recall@50 ≥ recall@10 по построению — эталон, дошедший
    до конца Слоя 1, обязан был сначала найтись поиском."""
    run = evaluate(
        [
            case("found-and-passed", expected_repos=["owner1/repo1"]),
            case("found-not-passed", expected_repos=["owner3/repo3"]),
            GoldenCase(slug="trap", **TRAP),
        ],
        log=Recorder(),
        github=FakeClient(),
        runner=runner_returning(outcome(passed=(1, 2), candidates=(1, 2, 3))),
    )

    summary = run.summary()
    assert summary["recall_at_50"] >= summary["recall_at_10"]
