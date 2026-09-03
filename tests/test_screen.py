"""Слой 1: скрининг кандидатов на V4-Flash (SCHEMAS.md §5).

Ни GitHub, ни DeepSeek не дёргаются: оба подменены фейками. Проверяется, что
код не доверяет модели там, где решение принадлежит коду, — `passed`, `repo_id`,
`provenance`, версия промпта.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from scout.schemas import Candidate, ModelName, ScreeningVerdict
from scout.screening import MAX_README_CHARS, load_system_prompt, screen
from test_queries import make_intent

REQUEST_ID = UUID("6f1f1b9c-0000-4000-8000-000000000001")
RETRIEVED_AT = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)

INTENT = make_intent(request_id=REQUEST_ID)


def candidate(n: int, **overrides) -> Candidate:
    fields = {
        "repo_id": n,
        "full_name": f"owner{n}/repo{n}",
        "html_url": f"https://github.com/owner{n}/repo{n}",
        "description": f"repo {n}",
        "language": "Python",
        "topics": ["pdf"],
        "stars": 100 * n,
        "archived": False,
        "is_fork": False,
        "pushed_at": RETRIEVED_AT,
        "default_branch": "main",
        "head_sha": f"{n:040x}",
        "license_spdx": "MIT",
        "found_by": ["q1"],
        "rrf_score": 0.5,
        "prior_score": 0.5,
        "rank": n,
        "retrieved_at": RETRIEVED_AT,
    }
    fields.update(overrides)
    return Candidate(**fields)


def answer(relevance=0.8, verdict="pass", reasons=None, red_flags=None) -> dict:
    return {
        "relevance": relevance,
        "verdict": verdict,
        "reasons": reasons or ["решает ровно эту задачу"],
        "red_flags": red_flags or [],
    }


class FakeDeepSeek:
    """Очередь ответов в порядке вызовов. Исключение в очереди — бросается."""

    def __init__(self, responses=None, usage=None):
        self.responses = list(responses or [])
        self.usage = usage or {
            "input_tokens": 1500,
            "output_tokens": 120,
            "cached_input_tokens": 1200,
        }
        self.calls = []

    def chat_json(self, *, system, user, model, temperature=0.0):
        self.calls.append(
            {"system": system, "user": user, "model": model, "temperature": temperature}
        )
        payload = self.responses.pop(0) if self.responses else answer()
        return payload, dict(self.usage)


class FakeGitHub:
    """README по `full_name`; отсутствие ключа — 404."""

    def __init__(self, readmes=None):
        self.readmes = readmes if readmes is not None else {}
        self.asked: list[tuple[str, str, str | None]] = []

    def get_file(self, full_name, path, ref=None):
        self.asked.append((full_name, path, ref))
        if not self.readmes:
            return f"# {full_name}\n\nБиблиотека для извлечения таблиц из PDF."
        return self.readmes.get(full_name)


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def info(self, event, **fields):
        self.events.append((event, fields))

    def error(self, event, **fields):
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [event for event, _ in self.events]


def run(candidates, *, deepseek=None, github=None, logger=None, **kwargs):
    return screen(
        candidates,
        INTENT,
        request_id=REQUEST_ID,
        github=github or FakeGitHub(),
        client=deepseek or FakeDeepSeek(),
        logger=logger,
        **kwargs,
    )


# --------------------------------------------------------------------------
# 50 → 10
# --------------------------------------------------------------------------


def test_fifty_candidates_are_screened_down_to_ten():
    candidates = [candidate(n) for n in range(1, 51)]
    deepseek = FakeDeepSeek([answer(relevance=n / 100) for n in range(1, 51)])

    run_result = run(candidates, deepseek=deepseek)

    assert len(deepseek.calls) == 50
    assert len(run_result.result.results) == 50
    assert len(run_result.result.passed) == 10


def test_passed_is_sorted_by_relevance_descending():
    candidates = [candidate(n) for n in range(1, 6)]
    relevances = [0.3, 0.9, 0.1, 0.7, 0.5]
    deepseek = FakeDeepSeek([answer(relevance=r) for r in relevances])

    passed = run(candidates, deepseek=deepseek).result.passed

    assert passed == [2, 4, 5, 1, 3]


def test_passed_is_recomputed_by_code_not_taken_from_the_model():
    """Модель может прислать что угодно, включая чужой repo_id — код берёт свой."""
    candidates = [candidate(1), candidate(2)]
    deepseek = FakeDeepSeek(
        [
            {**answer(relevance=0.2), "repo_id": 999_999, "full_name": "чужой/репозиторий"},
            {**answer(relevance=0.9), "repo_id": 999_999, "full_name": "чужой/репозиторий"},
        ]
    )

    result = run(candidates, deepseek=deepseek).result

    assert [item.repo_id for item in result.results] == [1, 2]
    assert [item.full_name for item in result.results] == ["owner1/repo1", "owner2/repo2"]
    assert result.passed == [2, 1]


def test_rejected_candidates_never_reach_passed():
    candidates = [candidate(1), candidate(2)]
    deepseek = FakeDeepSeek(
        [answer(relevance=0.95, verdict="reject", reasons=["учебный пример"]), answer(0.4)]
    )

    result = run(candidates, deepseek=deepseek).result

    assert result.results[0].verdict is ScreeningVerdict.REJECT
    assert result.passed == [2]


def test_rejected_candidate_is_logged_with_flags_and_reasons():
    """День 10 разбирает провалы recall: без причин отказа не отличить потерю
    на поиске от потери на скрининге."""
    candidates = [candidate(1)]
    deepseek = FakeDeepSeek(
        [
            answer(
                relevance=0.1,
                verdict="reject",
                reasons=["подборка ссылок, своего кода нет"],
                red_flags=["demo-or-tutorial", "wrong-domain"],
            )
        ]
    )
    logger = Recorder()

    run(candidates, deepseek=deepseek, logger=logger)

    rejected = next(fields for event, fields in logger.events if event == "candidate_rejected")
    assert rejected["repo_id"] == 1
    assert rejected["full_name"] == "owner1/repo1"
    assert rejected["verdict"] == "reject"
    assert rejected["red_flags"] == ["demo-or-tutorial", "wrong-domain"]
    assert rejected["reasons"] == ["подборка ссылок, своего кода нет"]


def test_passing_candidate_is_not_logged_as_rejected():
    logger = Recorder()
    run([candidate(1)], logger=logger)
    assert "candidate_rejected" not in logger.names()


def test_equal_relevance_is_ordered_by_repo_id():
    """Идемпотентность: при равном relevance порядок не должен зависеть от словаря."""
    candidates = [candidate(9), candidate(4)]
    deepseek = FakeDeepSeek([answer(relevance=0.5), answer(relevance=0.5)])

    assert run(candidates, deepseek=deepseek).result.passed == [4, 9]


def test_passed_cap_is_the_audit_limit():
    candidates = [candidate(n) for n in range(1, 21)]
    deepseek = FakeDeepSeek([answer(relevance=0.9) for _ in range(20)])

    assert len(run(candidates, deepseek=deepseek, limit=3).result.passed) == 3


# --------------------------------------------------------------------------
# Лицензия — не фильтр (CLAUDE.md, запрет 4)
# --------------------------------------------------------------------------


def test_gpl_candidate_passes_screening():
    """Copyleft влияет на вердикт отчёта, но не на попадание в топ-10."""
    candidates = [candidate(1, license_spdx="GPL-3.0"), candidate(2, license_spdx="MIT")]
    deepseek = FakeDeepSeek([answer(relevance=0.9), answer(relevance=0.4)])

    assert run(candidates, deepseek=deepseek).result.passed == [1, 2]


def test_candidate_without_license_passes_screening():
    candidates = [candidate(1, license_spdx=None)]
    assert run(candidates).result.passed == [1]


def test_prompt_says_nothing_about_filtering_by_license_or_stars():
    """Запрет 4 держится не только кодом: промпт не должен подталкивать к отсеву."""
    prompt = load_system_prompt().lower()
    assert "лиценз" not in prompt.split("## что не является основанием")[0]


# --------------------------------------------------------------------------
# Невалидный ответ модели
# --------------------------------------------------------------------------


def test_invalid_answer_is_retried_once_with_the_validation_error():
    candidates = [candidate(1)]
    deepseek = FakeDeepSeek([{"relevance": 1.5, "verdict": "pass", "reasons": ["ok"]}, answer()])

    result = run(candidates, deepseek=deepseek).result

    assert len(deepseek.calls) == 2
    assert "relevance" in deepseek.calls[1]["user"]
    assert len(result.results) == 1


def test_candidate_failing_twice_is_marked_failed_and_excluded():
    candidates = [candidate(1), candidate(2)]
    broken = {"relevance": 0.9, "verdict": "может быть", "reasons": []}
    deepseek = FakeDeepSeek([broken, broken, answer(relevance=0.3)])
    logger = Recorder()

    run_result = run(candidates, deepseek=deepseek, logger=logger)

    assert run_result.failed == ["owner1/repo1"]
    assert run_result.result.passed == [2]
    assert [item.repo_id for item in run_result.result.results] == [2]
    assert "screening_failed" in logger.names()


def test_unknown_red_flag_does_not_pass_validation():
    """Red flags — только из enum RedFlag: свободный текст ломает контракт отчёта."""
    candidates = [candidate(1)]
    deepseek = FakeDeepSeek([answer(red_flags=["выглядит-подозрительно"]), answer()])

    run(candidates, deepseek=deepseek)

    assert len(deepseek.calls) == 2


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def test_missing_readme_screens_by_metadata_and_logs_event():
    candidates = [candidate(1)]
    github = FakeGitHub(readmes={"owner2/repo2": "текст"})
    logger = Recorder()

    result = run(candidates, github=github, logger=logger).result

    assert len(result.results) == 1
    assert "readme_missing" in logger.names()


def test_readme_is_cut_to_four_thousand_characters():
    candidates = [candidate(1)]
    github = FakeGitHub(readmes={"owner1/repo1": "я" * 10_000})
    deepseek = FakeDeepSeek()

    run(candidates, github=github, deepseek=deepseek)

    assert "я" * MAX_README_CHARS in deepseek.calls[0]["user"]
    assert "я" * (MAX_README_CHARS + 1) not in deepseek.calls[0]["user"]


def test_readme_is_read_at_the_candidate_head_sha():
    """Читаем ту же ревизию, которая записана в кандидате, а не подвижный HEAD."""
    candidates = [candidate(1)]
    github = FakeGitHub()

    run(candidates, github=github)

    assert github.asked[0][2] == candidates[0].head_sha


# --------------------------------------------------------------------------
# Контракт и учёт токенов
# --------------------------------------------------------------------------


def test_token_usage_is_summed_across_calls():
    candidates = [candidate(1), candidate(2), candidate(3)]
    usage = {"input_tokens": 1500, "output_tokens": 120, "cached_input_tokens": 1200}
    deepseek = FakeDeepSeek(usage=usage)

    token_usage = run(candidates, deepseek=deepseek).result.token_usage

    assert token_usage.input_tokens == 4500
    assert token_usage.output_tokens == 360
    assert token_usage.cached_input_tokens == 3600
    assert token_usage.model is ModelName.FLASH


def test_layer_model_and_prompt_version_are_owned_by_code():
    result = run([candidate(1)]).result

    assert result.layer == 1
    assert result.model is ModelName.FLASH
    assert result.prompt_version == "l1-1"
    assert result.request_id == REQUEST_ID


def test_evidence_is_filled_by_code_at_the_candidate_sha():
    result = run([candidate(1)]).result
    evidence = result.results[0].evidence

    assert [item.path for item in evidence] == ["metadata", "README.md"]
    assert {item.commit_sha for item in evidence} == {candidate(1).head_sha}


def test_evidence_holds_metadata_only_when_readme_is_missing():
    github = FakeGitHub(readmes={})
    github.readmes = {"нет": "нет"}
    result = run([candidate(1)], github=github).result

    assert [item.path for item in result.results[0].evidence] == ["metadata"]


def test_system_prompt_is_identical_for_every_candidate():
    """Одинаковый префикс — условие cache hit по токеномике ARCHITECTURE.md."""
    deepseek = FakeDeepSeek()
    run([candidate(1), candidate(2), candidate(3)], deepseek=deepseek)

    assert len({call["system"] for call in deepseek.calls}) == 1
    assert {call["temperature"] for call in deepseek.calls} == {0.0}
    assert {call["model"] for call in deepseek.calls} == {ModelName.FLASH.value}


def test_empty_candidate_list_gives_valid_empty_result():
    result = run([]).result

    assert result.results == []
    assert result.passed == []
    assert result.token_usage.input_tokens == 0


@pytest.mark.parametrize("field", ["stars", "language", "description"])
def test_metadata_reaches_the_model(field):
    deepseek = FakeDeepSeek()
    run([candidate(1)], deepseek=deepseek)

    assert str(getattr(candidate(1), field)) in deepseek.calls[0]["user"]
