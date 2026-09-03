"""CLI и его окружение: порядок стадий конвейера, коды возврата, загрузка `.env`.

Ни GitHub, ни DeepSeek здесь не дёргаются — `extract_intent`, `collect_candidates`
и `screen` подменяются целиком. Проверяется не качество слоёв (это `test_intent.py`,
`test_search.py`, `test_screen.py`), а то, как CLI их связывает и что делает,
когда очередная стадия ничего не вернула.
"""

import json
import os
from pathlib import Path

import pytest

from scout import cli, config
from scout.github import GitHubError
from scout.intent import IntentExtraction
from scout.queries import QueryGenerationError
from scout.schemas import ModelName, PricingWindow, ScreeningResult, TokenUsage
from scout.screening import ScreeningRun
from scout.search import SearchOutcome

# Эталонный интент живёт в одном месте на все тесты — в модуле генератора запросов.
from test_queries import make_intent
from test_screen import candidate

QUERY = "нужен парсер PDF-таблиц на Python"


def screening_run(request_id, passed=(1, 2), failed=()):
    """Минимальный валидный `ScreeningResult` — стык проверяется, а не содержание."""
    results = [
        {
            "repo_id": repo_id,
            "full_name": f"owner{repo_id}/repo{repo_id}",
            "relevance": 0.9,
            "verdict": "pass",
            "reasons": [f"причина для {repo_id}"],
        }
        for repo_id in passed
    ]
    return ScreeningRun(
        result=ScreeningResult(
            request_id=request_id,
            layer=1,
            model=ModelName.FLASH,
            prompt_version="l1-1",
            results=results,
            passed=list(passed),
            token_usage=TokenUsage(
                model=ModelName.FLASH,
                input_tokens=7500,
                cached_input_tokens=6000,
                output_tokens=600,
                cost_usd=0.0,
                pricing_window=PricingWindow.OFF_PEAK,
            ),
        ),
        failed=list(failed),
    )


@pytest.fixture
def ok_intent(monkeypatch):
    """`extract_intent` без сети: всегда успешный разбор эталонного интента."""

    def fake_extract(task_text, *, request_id, client=None, logger=None):
        return IntentExtraction(
            status="ok",
            intent=make_intent(request_id=request_id),
            attempts=1,
            usage={"input_tokens": 900, "output_tokens": 210},
        )

    monkeypatch.setattr(cli, "extract_intent", fake_extract)


@pytest.fixture
def offline(monkeypatch):
    """Поиск и скрининг без сети: два кандидата, оба прошли."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-fake")

    def fake_collect(query_set, *, intent, github, limit, logger=None, **kwargs):
        return SearchOutcome(
            candidates=[candidate(1), candidate(2)],
            queries_used=[query.q for query in query_set.queries],
        )

    def fake_screen(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id)

    monkeypatch.setattr(cli, "collect_candidates", fake_collect)
    monkeypatch.setattr(cli, "screen", fake_screen)


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


def test_stub_now_stands_at_audit(ok_intent, offline, capsys):
    """Слой 1 пройден — следующая незакрытая стадия конвейера уже аудит."""
    cli.main(["scan", QUERY])
    assert events(capsys)["reached_stub"]["stage"] == "audit"


def test_screening_stage_is_announced_before_layer_one(ok_intent, offline, capsys):
    cli.main(["scan", QUERY])

    stages = [
        record["stage"] for record in _all_events(capsys) if record["event"] == "reached_stub"
    ]
    assert stages == ["screening", "audit"]


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
    assert done["input_tokens"] == 7500
    assert done["prompt_version"] == "l1-1"


def test_empty_search_gives_build_recommendation_and_exit_zero(
    ok_intent, offline, monkeypatch, capsys
):
    """Ноль кандидатов — это ответ «пиши сам», а не сбой скана (QUERIES.md, шаг 5)."""

    def nothing_found(query_set, *, intent, github, limit, logger=None, **kwargs):
        return SearchOutcome(queries_used=[query.q for query in query_set.queries])

    monkeypatch.setattr(cli, "collect_candidates", nothing_found)

    assert cli.main(["scan", QUERY]) == 0

    captured = capsys.readouterr()
    assert "BUILD" in captured.out
    assert "Проверено запросов: 7." in captured.out


def test_nobody_passing_screening_also_gives_build(ok_intent, offline, monkeypatch, capsys):
    def none_passed(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id, passed=())

    monkeypatch.setattr(cli, "screen", none_passed)

    assert cli.main(["scan", QUERY]) == 0
    assert "BUILD" in capsys.readouterr().out


def test_github_failure_has_its_own_exit_code(ok_intent, offline, monkeypatch, capsys):
    def explode(query_set, *, intent, github, limit, logger=None, **kwargs):
        raise GitHubError("500 от GitHub")

    monkeypatch.setattr(cli, "collect_candidates", explode)

    assert cli.main(["scan", QUERY]) == 7
    assert "search_failed_hard" in events(capsys)


def test_missing_github_token_stops_before_the_search(ok_intent, monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_API_KEY": "sk-fake"})

    assert cli.main(["scan", QUERY]) == 3
    assert "missing_credential" in events(capsys)


def test_query_generation_failure_has_its_own_exit_code(ok_intent, monkeypatch, capsys):
    def explode(intent, **kwargs):
        raise QueryGenerationError("из интента собралось 4 различимых запроса")

    monkeypatch.setattr(cli, "build_query_set", explode)

    assert cli.main(["scan", QUERY]) == 6
    assert "queries_failed" in events(capsys)


def test_intent_failure_does_not_reach_the_generator(monkeypatch, capsys):
    """Провал разбора задачи не должен маскироваться под провал генератора."""

    def failed_extract(task_text, *, request_id, client=None, logger=None):
        return IntentExtraction(status="failed", attempts=2, errors=["synonyms: too short"])

    monkeypatch.setattr(cli, "extract_intent", failed_extract)

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
