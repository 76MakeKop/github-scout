"""Интеграция: запросы → поиск → дедуп и RRF → `Candidate[]`.

Сети нет: `GitHubClient` подменяется фейком со списком заготовленных выдач.
Проверяется стык, а не переранжирование (это `test_rank.py`) и не сборка
строк запросов (`test_queries.py`).
"""

from datetime import UTC, datetime

import pytest

from scout.github import GitHubError, RateLimitExhausted
from scout.queries import build_query_set
from scout.search import MIN_UNIQUE_CANDIDATES, collect_candidates
from test_queries import make_intent
from test_rank import repo

NOW = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)


class FakeGitHub:
    """Выдачи по строке запроса; `head_sha` — по `full_name`."""

    def __init__(self, *, pages=None, default_page=None, head_shas=None, fail=None):
        self.pages = pages or {}
        self.default_page = default_page if default_page is not None else []
        self.head_shas = head_shas if head_shas is not None else {}
        self.fail = fail or {}
        self.searched: list[str] = []
        self.head_sha_calls: list[str] = []

    def search_repositories(self, q, *, sort=None, per_page=30):
        self.searched.append(q)
        error = self.fail.get(q)
        if error is not None:
            raise error
        return self.pages.get(q, self.default_page)

    def get_head_sha(self, full_name, branch):
        self.head_sha_calls.append(full_name)
        if not self.head_shas:
            return "a" * 40
        return self.head_shas.get(full_name)


def query_set(**overrides):
    return build_query_set(make_intent(**overrides), generated_at=NOW)


def collect(github, *, intent_overrides=None, **kwargs):
    intent = make_intent(**(intent_overrides or {}))
    return collect_candidates(
        build_query_set(intent, generated_at=NOW),
        intent=intent,
        github=github,
        now=NOW,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Прогон запросов
# --------------------------------------------------------------------------


def test_every_generated_query_is_executed():
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 13)])
    outcome = collect(github)

    assert len(github.searched) == 7
    assert github.searched == [query.q for query in query_set().queries]
    assert outcome.queries_used == github.searched


def test_results_of_all_queries_are_merged_into_candidates():
    """Один и тот же репозиторий из разных запросов — один кандидат с общим found_by."""
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 13)])
    outcome = collect(github)

    assert [c.repo_id for c in outcome.candidates] == list(range(1, 13))
    assert outcome.candidates[0].found_by == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]


def test_candidates_are_valid_and_carry_head_sha():
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 13)], head_shas={})
    outcome = collect(github)

    first = outcome.candidates[0]
    assert first.head_sha == "a" * 40
    assert first.rank == 1
    assert first.retrieved_at.tzinfo is not None


def test_limit_caps_candidates_at_fifty():
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 30)])
    outcome = collect(github, limit=5)
    assert len(outcome.candidates) == 5


# --------------------------------------------------------------------------
# head_sha
# --------------------------------------------------------------------------


def test_missing_head_sha_drops_candidate_without_failing_scan():
    """404 на голову ветки — репозиторий исчез между поиском и сборкой, не авария."""
    events = []
    github = FakeGitHub(
        default_page=[repo(n) for n in range(1, 13)],
        head_shas={f"owner{n}/repo{n}": "b" * 40 for n in range(2, 13)},
    )

    outcome = collect(github, logger=_Recorder(events))

    assert 1 not in [c.repo_id for c in outcome.candidates]
    assert len(outcome.candidates) == 11
    assert any(
        event == "candidate_dropped" and fields.get("reason") == "head_sha_missing"
        for event, fields in events
    )


def test_ranks_stay_dense_after_a_candidate_drops_out():
    """Дырка в нумерации означала бы, что Слою 1 приехал не тот срез, что отобран."""
    github = FakeGitHub(
        default_page=[repo(n) for n in range(1, 13)],
        head_shas={f"owner{n}/repo{n}": "b" * 40 for n in range(2, 13)},
    )
    outcome = collect(github)

    assert [c.rank for c in outcome.candidates] == list(range(1, 12))


def test_head_sha_is_asked_only_for_the_selected_slice():
    """Голова ветки стоит вызова `core` — спрашиваем её после отбора, а не до."""
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 30)])
    collect(github, limit=5)
    assert len(github.head_sha_calls) == 5


# --------------------------------------------------------------------------
# Отказы поиска
# --------------------------------------------------------------------------


def test_failed_query_marks_run_partial_and_keeps_the_rest():
    """403/429 на одном запросе: ранжируем то, что успели собрать (QUERIES.md, шаг 5)."""
    queries = [query.q for query in query_set().queries]
    github = FakeGitHub(
        default_page=[repo(n) for n in range(1, 13)],
        fail={queries[0]: GitHubError("403 от GitHub")},
    )

    outcome = collect(github)

    assert outcome.partial is True
    assert outcome.candidates


def test_exhausted_rate_limit_stops_further_searches():
    queries = [query.q for query in query_set().queries]
    github = FakeGitHub(
        default_page=[repo(1)],
        fail={queries[1]: RateLimitExhausted("лимит не сбросился")},
    )

    outcome = collect(github)

    assert outcome.partial is True
    assert len(github.searched) == 2


def test_successful_run_is_not_partial():
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 13)])
    assert collect(github).partial is False


# --------------------------------------------------------------------------
# Вырожденные случаи (QUERIES.md, шаг 5)
# --------------------------------------------------------------------------


def test_thin_output_triggers_broad_fallback_run():
    """Меньше 10 уникальных — добор широким неводом без `language:` и `stars:`."""
    events = []
    github = FakeGitHub(default_page=[repo(1), repo(2)])

    collect(github, logger=_Recorder(events))

    assert len(github.searched) == 8
    fallback = github.searched[-1]
    assert "language:" not in fallback
    assert "stars:" not in fallback
    assert any(event == "broad_fallback_added" for event, _ in events)


def test_broad_fallback_results_join_the_ranking():
    queries = [query.q for query in query_set().queries]
    github = FakeGitHub(
        pages={q: [repo(1)] for q in queries},
        default_page=[repo(n) for n in range(2, 15)],
    )

    outcome = collect(github)

    assert len(outcome.candidates) >= MIN_UNIQUE_CANDIDATES


def test_rich_output_does_not_trigger_the_fallback():
    github = FakeGitHub(default_page=[repo(n) for n in range(1, 15)])
    github_calls = collect(github)

    assert len(github.searched) == 7
    assert len(github_calls.candidates) == 14


def test_zero_results_give_no_candidates_and_no_crash():
    github = FakeGitHub(default_page=[])
    outcome = collect(github)

    assert outcome.candidates == []
    assert outcome.queries_used


class _Recorder:
    """Логгер-накопитель: события проверяются, а не печатаются."""

    def __init__(self, sink):
        self.sink = sink

    def info(self, event, **fields):
        self.sink.append((event, fields))

    def error(self, event, **fields):
        self.sink.append((event, fields))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """На всякий случай: фейк не спит, но и настоящий троттлинг сюда не должен пролезть."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
