"""CLI и его окружение: стык интента с генератором запросов, загрузка `.env`.

DeepSeek здесь не дёргается — `extract_intent` подменяется целиком. Проверяется
не качество интента (это дело `test_intent.py`), а то, что конвейер доходит
до сгенерированных запросов и правильно ведёт себя при отказе генератора.
"""

import json
import os
from pathlib import Path

import pytest

from scout import cli, config
from scout.intent import IntentExtraction
from scout.queries import QueryGenerationError

# Эталонный интент живёт в одном месте на все тесты — в модуле генератора запросов.
from test_queries import make_intent

QUERY = "нужен парсер PDF-таблиц на Python"


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


def events(capsys) -> dict[str, dict]:
    """Последнее событие каждого типа из потока JSON-логов.

    В том же stderr лежат и человекочитаемые сообщения об ошибках — они не JSON
    и пропускаются.
    """
    parsed = []
    for line in capsys.readouterr().err.splitlines():
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {record["event"]: record for record in parsed}


def test_scan_reaches_generated_queries(ok_intent, capsys):
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


def test_generated_queries_are_logged_verbatim(ok_intent, capsys):
    """Строки запросов уходят в лог: без них отладку выдачи не провести."""
    cli.main(["scan", QUERY])

    queries = events(capsys)["queries_generated"]["queries"]
    assert queries[0].startswith("pdf table extraction language:python")
    assert len(queries) == 7


def test_stub_now_stands_at_search(ok_intent, capsys):
    cli.main(["scan", QUERY])
    assert events(capsys)["reached_stub"]["stage"] == "search"


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
                "export QWEN_API_KEY = qwen-spaced ",
                "СЛОМАННАЯ_СТРОКА_БЕЗ_РАВНО",
            ]
        ),
        encoding="utf-8",
    )

    assert config.load_dotenv(env_file, environ=environ) == 3
    assert environ == {
        "DEEPSEEK_API_KEY": "sk-plain",
        "GITHUB_TOKEN": "ghp-quoted",
        "QWEN_API_KEY": "qwen-spaced",
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
