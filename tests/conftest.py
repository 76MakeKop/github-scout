"""Общие фикстуры CLI-тестов.

Живут здесь, а не в `test_cli.py`, потому что нужны двум файлам сразу:
`test_cli.py` проверяет порядок стадий, `test_errors.py` — живучесть на тех же
подменах. Импорт фикстуры из соседнего теста pytest формально позволяет, но имя
при этом попадает в модуль дважды — как импорт и как аргумент теста, — и линтер
справедливо ругается на переопределение. `conftest.py` — штатное место.
"""

import pytest

from scout import cli
from scout.intent import IntentExtraction
from scout.schemas import ModelName, PricingWindow, ScreeningResult, TokenUsage
from scout.screening import ScreeningRun
from scout.search import SearchOutcome
from test_queries import make_intent
from test_screen import candidate


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
def offline(monkeypatch, tmp_path):
    """Поиск и скрининг без сети: два кандидата, оба прошли.

    Очередь отложенных сканов тоже уводится во временный файл: `scan` разгребает
    её на старте, и без подмены тесты писали бы в базу проекта.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-fake")
    monkeypatch.setattr(cli, "DEFAULT_QUEUE_PATH", tmp_path / "pending.sqlite3")

    def fake_collect(query_set, *, intent, github, limit, logger=None, **kwargs):
        return SearchOutcome(
            candidates=[candidate(1), candidate(2)],
            queries_used=[query.q for query in query_set.queries],
        )

    def fake_screen(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id)

    monkeypatch.setattr(cli, "collect_candidates", fake_collect)
    monkeypatch.setattr(cli, "screen", fake_screen)
