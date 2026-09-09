"""CLI и его окружение: порядок стадий конвейера, коды возврата, загрузка `.env`.

Ни GitHub, ни DeepSeek здесь не дёргаются — `extract_intent`, `collect_candidates`
и `screen` подменяются целиком. Проверяется не качество слоёв (это `test_intent.py`,
`test_search.py`, `test_screen.py`), а то, как CLI их связывает и что делает,
когда очередная стадия ничего не вернула.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import screening_run
from scout import cli, config, pipeline
from scout.github import GitHubError
from scout.intent import IntentExtraction
from scout.queries import QueryGenerationError
from scout.scheduler import PendingScans
from scout.schemas import ScanOptions
from scout.search import SearchOutcome

QUERY = "нужен парсер PDF-таблиц на Python"


# Фикстуры `ok_intent` и `offline` и помощник `screening_run` переехали
# в `conftest.py`: их делят два файла — этот и `test_errors.py`.


def _all_events(capsys) -> list[dict]:
    """Поток JSON-логов по порядку.

    В том же stderr лежат и человекочитаемые сообщения об ошибках — они не JSON
    и пропускаются.
    """
    parsed = []
    for line in capsys.readouterr().err.splitlines():
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed


def events(capsys) -> dict[str, dict]:
    """Последнее событие каждого типа."""
    return {record["event"]: record for record in _all_events(capsys)}


def test_scan_reaches_generated_queries(ok_intent, offline, capsys):
    assert cli.main(["scan", QUERY]) == 0

    emitted = events(capsys)
    assert emitted["queries_generated"]["query_count"] == 7
    assert emitted["queries_generated"]["generator_version"] == "qg-1"
    assert emitted["queries_generated"]["families"] == [
        "exact",
        "synonym",
        "topic",
        "library",
        "readme",
        "broad",
        "recent",
    ]


def test_generated_queries_are_logged_verbatim(ok_intent, offline, capsys):
    """Строки запросов уходят в лог: без них отладку выдачи не провести."""
    cli.main(["scan", QUERY])

    queries = events(capsys)["queries_generated"]["queries"]
    assert queries[0].startswith("pdf table extraction language:python")
    assert len(queries) == 7


def test_stub_now_stands_at_the_report(ok_intent, offline, capsys):
    """Слой 2 закрыт на дне 11 — следующая незакрытая стадия уже отчёт (день 13)."""
    cli.main(["scan", QUERY])
    assert events(capsys)["reached_stub"]["stage"] == "report"


def test_stages_are_announced_in_pipeline_order(ok_intent, offline, capsys):
    cli.main(["scan", QUERY])

    stages = [
        record["stage"] for record in _all_events(capsys) if record["event"] == "reached_stub"
    ]
    assert stages == ["screening", "audit", "report"]


def test_passed_candidates_are_printed_with_relevance_and_reason(ok_intent, offline, capsys):
    assert cli.main(["scan", QUERY]) == 0

    out = capsys.readouterr().out
    assert "owner1/repo1" in out
    assert "relevance 0.90" in out
    assert "причина для 1" in out


def test_screening_totals_reach_the_log(ok_intent, offline, capsys):
    cli.main(["scan", QUERY])

    done = events(capsys)["screening_done"]
    assert done["passed"] == 2
    assert done["token_usage"]["input_tokens"] == 7500
    assert done["prompt_version"] == "l1-1"


def test_scan_prints_and_logs_the_total_cost(ok_intent, offline, capsys):
    """Критерий приёмки дня 7: цена видна и в логе, и в конце отчёта."""
    cli.main(["scan", QUERY])

    captured = capsys.readouterr()
    assert "total_cost_usd" in captured.out

    events_by_name = {
        record["event"]: record
        for line in captured.err.splitlines()
        if (record := _maybe_json(line))
    }
    cost = events_by_name["scan_cost"]
    assert cost["total_cost_usd"] == pytest.approx(
        cost["intent_usd"] + cost["screening_usd"], rel=1e-6
    )
    assert cost["total_cost_usd"] > 0


def _maybe_json(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# Режим --off-peak
# --------------------------------------------------------------------------


class FrozenClock:
    """Подменяет `datetime.now` в модуле CLI — время очереди должно быть управляемым."""

    def __init__(self, moment):
        self.moment = moment

    def now(self, tz=None):
        return self.moment


def freeze(monkeypatch, moment):
    monkeypatch.setattr(cli, "datetime", FrozenClock(moment))


PEAK_MOMENT = datetime(2026, 9, 4, 2, 0, tzinfo=UTC)
OFFPEAK_MOMENT = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def test_offpeak_flag_defers_execution_during_peak(ok_intent, offline, monkeypatch, capsys):
    """02:00 UTC плюс флаг — задача уходит в очередь, скан не выполняется."""
    freeze(monkeypatch, PEAK_MOMENT)

    assert cli.main(["scan", QUERY, "--off-peak"]) == 0

    captured = capsys.readouterr()
    assert "Задача в очереди" in captured.out
    assert "04:00 UTC" in captured.out
    assert "Слой 1" not in captured.out

    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        pending = queue.all()
    assert len(pending) == 1
    assert pending[0].query == QUERY
    assert pending[0].scheduled_at == datetime(2026, 9, 4, 4, 0, tzinfo=UTC)


def test_offpeak_flag_runs_immediately_outside_peak(ok_intent, offline, monkeypatch, capsys):
    """12:00 UTC уже дёшево — ждать нечего, скан идёт сразу."""
    freeze(monkeypatch, OFFPEAK_MOMENT)

    assert cli.main(["scan", QUERY, "--off-peak"]) == 0

    assert "Слой 1" in capsys.readouterr().out
    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        assert queue.all() == []


def test_scan_without_the_flag_runs_during_peak(ok_intent, offline, monkeypatch, capsys):
    """CLAUDE.md, запрет 8: без явного флага откладывать нельзя даже в пик."""
    freeze(monkeypatch, PEAK_MOMENT)

    assert cli.main(["scan", QUERY]) == 0
    assert "Слой 1" in capsys.readouterr().out


def test_worker_executes_a_task_whose_time_has_come(ok_intent, offline, monkeypatch, capsys):
    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        queue.add(QUERY, ScanOptions(off_peak=True), scheduled_at=PEAK_MOMENT)

    freeze(monkeypatch, datetime(2026, 9, 4, 4, 30, tzinfo=UTC))

    assert cli.main(["worker", "run"]) == 0

    out = capsys.readouterr().out
    assert "Выполняю отложенную задачу" in out
    assert "Слой 1" in out

    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        assert queue.all() == []


def test_worker_leaves_a_task_whose_time_has_not_come(ok_intent, offline, monkeypatch, capsys):
    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        queue.add(QUERY, ScanOptions(off_peak=True), scheduled_at=OFFPEAK_MOMENT)

    freeze(monkeypatch, PEAK_MOMENT)

    assert cli.main(["worker", "run"]) == 0
    assert "ближайшая" in capsys.readouterr().out

    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        assert len(queue.all()) == 1


def test_worker_on_empty_queue_says_so(offline, capsys):
    assert cli.main(["worker", "run"]) == 0
    assert "Очередь пуста" in capsys.readouterr().out


def test_scan_drains_the_queue_before_its_own_work(ok_intent, offline, monkeypatch, capsys):
    """Демона нет: отложенная задача исполняется при следующем запуске CLI."""
    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        queue.add("отложенная задача про PDF", ScanOptions(), scheduled_at=PEAK_MOMENT)

    freeze(monkeypatch, OFFPEAK_MOMENT)

    assert cli.main(["scan", QUERY]) == 0

    out = capsys.readouterr().out
    assert out.index("Выполняю отложенную задачу") < out.index("Слой 1")
    with PendingScans(cli.DEFAULT_QUEUE_PATH) as queue:
        assert queue.all() == []


def test_empty_search_gives_build_recommendation_and_exit_zero(
    ok_intent, offline, monkeypatch, capsys
):
    """Ноль кандидатов — это ответ «пиши сам», а не сбой скана (QUERIES.md, шаг 5)."""

    def nothing_found(query_set, *, intent, github, limit, logger=None, **kwargs):
        return SearchOutcome(queries_used=[query.q for query in query_set.queries])

    monkeypatch.setattr(pipeline, "collect_candidates", nothing_found)

    assert cli.main(["scan", QUERY]) == 0

    captured = capsys.readouterr()
    assert "BUILD" in captured.out
    assert "Проверено запросов: 7." in captured.out


def test_nobody_passing_screening_also_gives_build(ok_intent, offline, monkeypatch, capsys):
    def none_passed(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id, passed=())

    monkeypatch.setattr(pipeline, "screen", none_passed)

    assert cli.main(["scan", QUERY]) == 0
    assert "BUILD" in capsys.readouterr().out


def test_github_failure_has_its_own_exit_code(ok_intent, offline, monkeypatch, capsys):
    def explode(query_set, *, intent, github, limit, logger=None, **kwargs):
        raise GitHubError("500 от GitHub")

    monkeypatch.setattr(pipeline, "collect_candidates", explode)

    assert cli.main(["scan", QUERY]) == 7
    assert "search_failed_hard" in events(capsys)


def test_missing_github_token_stops_before_the_search(ok_intent, monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_API_KEY": "sk-fake"})

    assert cli.main(["scan", QUERY]) == 3
    assert "credential_rejected" in events(capsys)


def test_query_generation_failure_has_its_own_exit_code(ok_intent, monkeypatch, capsys):
    def explode(intent, **kwargs):
        raise QueryGenerationError("из интента собралось 4 различимых запроса")

    monkeypatch.setattr(pipeline, "build_query_set", explode)

    assert cli.main(["scan", QUERY]) == 6
    assert "queries_failed" in events(capsys)


def test_intent_failure_does_not_reach_the_generator(monkeypatch, capsys):
    """Провал разбора задачи не должен маскироваться под провал генератора."""

    def failed_extract(task_text, *, request_id, client=None, logger=None):
        return IntentExtraction(status="failed", attempts=2, errors=["synonyms: too short"])

    monkeypatch.setattr(pipeline, "extract_intent", failed_extract)

    assert cli.main(["scan", QUERY]) == 5
    assert "queries_generated" not in events(capsys)


# --------------------------------------------------------------------------
# Загрузка .env
# --------------------------------------------------------------------------


def test_dotenv_in_project_root_is_visible_in_config(tmp_path, monkeypatch):
    """Ключ вписан в `.env` — и `config` его видит, без плясок в терминале.

    `os.environ` подменяется словарём: иначе тест протёк бы настоящим ключом
    в остальные тесты.
    """
    monkeypatch.setattr(os, "environ", {})
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-from-dotenv\n", encoding="utf-8")

    config.load_dotenv(env_file)

    assert config.deepseek_api_key() == "sk-from-dotenv"


def test_dotenv_path_points_at_project_root():
    """Файл ищется рядом с `.env.example`, а не рядом с исходниками пакета."""
    assert config.DOTENV_PATH.name == ".env"
    assert (config.DOTENV_PATH.parent / ".env.example").exists()


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    """В CI ключи уже в окружении, и забытый локальный `.env` не должен их перебить."""
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_API_KEY": "sk-from-ci"})
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-from-dotenv\n", encoding="utf-8")

    config.load_dotenv(env_file)

    assert config.deepseek_api_key() == "sk-from-ci"


def test_missing_dotenv_is_silent():
    """Отсутствие файла — норма: в CI переменные приходят из окружения."""
    assert config.load_dotenv(Path("/nonexistent/.env")) == 0


def test_dotenv_parsing_handles_real_world_lines(tmp_path, monkeypatch):
    environ: dict[str, str] = {}
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# комментарий",
                "",
                "DEEPSEEK_API_KEY=sk-plain",
                'GITHUB_TOKEN="ghp-quoted"',
                "export OPENROUTER_API_KEY = sk-or-spaced ",
                "СЛОМАННАЯ_СТРОКА_БЕЗ_РАВНО",
            ]
        ),
        encoding="utf-8",
    )

    assert config.load_dotenv(env_file, environ=environ) == 3
    assert environ == {
        "DEEPSEEK_API_KEY": "sk-plain",
        "GITHUB_TOKEN": "ghp-quoted",
        "OPENROUTER_API_KEY": "sk-or-spaced",
    }


def test_empty_value_does_not_shadow_a_missing_key(tmp_path, monkeypatch):
    """`.env.example` целиком состоит из пустых значений: копию без правки
    надо считать отсутствием ключа, а не пустым ключом."""
    monkeypatch.setattr(os, "environ", {})
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")

    config.load_dotenv(env_file)

    with pytest.raises(config.MissingCredential):
        config.deepseek_api_key()


# --------------------------------------------------------------------------
# scout eval — прогон golden-set
# --------------------------------------------------------------------------


def golden_dir(tmp_path, count=3):
    directory = tmp_path / "golden"
    directory.mkdir()
    for n in range(1, count + 1):
        (directory / f"case{n}.json").write_text(
            json.dumps(
                {
                    "query_text": f"нужен инструмент номер {n} для разбора данных",
                    "expected_repos": [f"owner{n}/repo{n}"],
                    "expected_verdict": "USE",
                    "note": "тестовая задача",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return directory


def test_dry_run_checks_the_set_without_spending_anything(tmp_path, monkeypatch, capsys):
    """Проверка набора обязана быть бесплатной: иначе опечатку в JSON
    приходится ловить деньгами."""

    def must_not_run(*args, **kwargs):
        raise AssertionError("--dry-run не должен запускать прогон")

    monkeypatch.setattr(cli, "evaluate", must_not_run)

    assert cli.main(["eval", "--golden", str(golden_dir(tmp_path)), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "Задач в наборе: 3" in out
    assert "Прогон не запускался" in out


def test_broken_set_stops_before_the_run(tmp_path, capsys):
    directory = tmp_path / "golden"
    directory.mkdir()
    (directory / "broken.json").write_text("{не json", encoding="utf-8")

    assert cli.main(["eval", "--golden", str(directory), "--dry-run"]) == 2
    assert "Набор не читается" in capsys.readouterr().err


def test_limit_cuts_the_set_for_a_pilot_run(tmp_path, capsys):
    """Пилот гоняет первые N задач: дешёвая проверка харнесса перед полным набором."""
    assert (
        cli.main(["eval", "--golden", str(golden_dir(tmp_path)), "--limit", "2", "--dry-run"]) == 0
    )
    assert "Задач в наборе: 2" in capsys.readouterr().out
