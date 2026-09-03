"""Дедуп, схлопывание форков, RRF, приор, сортировка (QUERIES.md, шаги 3–4).

Сети здесь нет: переранжирование — чистая функция от выдачи и момента времени.
Момент передаётся явно, иначе свежесть (`prior`) плыла бы вместе с часами.

Числа в тестах RRF посчитаны руками по формуле QUERIES.md, а не сняты с вывода
кода: тест, снятый с реализации, проверяет сам себя.
"""

from datetime import UTC, datetime

import pytest

from scout.rank import (
    K_RRF,
    PRIOR_WEIGHT,
    RRF_WEIGHT,
    RankingError,
    rank_candidates,
    to_candidate,
)

NOW = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)
RETRIEVED_AT = datetime(2026, 9, 4, 12, 0, 5, tzinfo=UTC)


def repo(repo_id: int, **overrides) -> dict:
    """Элемент выдачи GitHub Search. Живой минимум полей плюс точечные правки."""
    item = {
        "id": repo_id,
        "full_name": f"owner{repo_id}/repo{repo_id}",
        "name": f"repo{repo_id}",
        "html_url": f"https://github.com/owner{repo_id}/repo{repo_id}",
        "description": f"repo {repo_id}",
        "language": "Python",
        "topics": ["pdf"],
        "stargazers_count": 100,
        "forks_count": 10,
        "open_issues_count": 3,
        "archived": False,
        "fork": False,
        "size": 500,
        "created_at": "2020-01-01T00:00:00Z",
        "pushed_at": "2026-08-01T00:00:00Z",
        "default_branch": "main",
        "license": {"spdx_id": "MIT"},
    }
    item.update(overrides)
    return item


def ids(ranked) -> list[int]:
    return [item.repo["id"] for item in ranked]


def by_id(ranked, repo_id: int):
    return next(item for item in ranked if item.repo["id"] == repo_id)


def rank(results, **overrides):
    return rank_candidates(results, now=NOW, **overrides)


# --------------------------------------------------------------------------
# RRF на синтетических списках с известным ответом (критерий приёмки дня 4)
# --------------------------------------------------------------------------


def test_rrf_matches_hand_computed_sum():
    """rrf(r) = Σ 1/(k + rank_q(r)), k = 60 — по всем запросам, где r встретился."""
    ranked = rank({"q1": [repo(1), repo(2)], "q2": [repo(2), repo(1)]})

    expected_raw = {
        1: 1 / (K_RRF + 1) + 1 / (K_RRF + 2),
        2: 1 / (K_RRF + 1) + 1 / (K_RRF + 2),
    }
    best = max(expected_raw.values())
    for repo_id, raw in expected_raw.items():
        assert by_id(ranked, repo_id).rrf_score == pytest.approx(raw / best)


def test_rrf_normalized_top_is_exactly_one():
    """Нормализация rrf_norm = rrf / max rrf: у лидера ровно 1.0."""
    ranked = rank({"q1": [repo(1), repo(2), repo(3)]})
    assert ranked[0].rrf_score == pytest.approx(1.0)
    assert ranked[1].rrf_score < 1.0


def test_repo_found_by_two_queries_beats_higher_single_hit():
    """Ради этого RRF и берут: два вторых места весомее одного первого.

    1/62 + 1/62 = 0.03226 против 1/61 = 0.01639 — при равных метаданных
    сумма рангов решает.
    """
    ranked = rank({"q1": [repo(1), repo(2)], "q2": [repo(3), repo(2)]})
    assert ids(ranked)[0] == 2


def test_missing_query_result_does_not_break_ranking():
    """Один запрос вернул 0 — норма, не ошибка (QUERIES.md, шаг 5)."""
    ranked = rank({"q1": [repo(1)], "q2": [], "q3": [repo(2)]})
    assert sorted(ids(ranked)) == [1, 2]


def test_empty_results_give_empty_list():
    assert rank({"q1": [], "q2": []}) == []


# --------------------------------------------------------------------------
# Дедупликация (QUERIES.md, шаг 3)
# --------------------------------------------------------------------------


def test_duplicate_repo_collapses_and_keeps_all_finders():
    ranked = rank({"q1": [repo(7)], "q3": [repo(7)], "q2": [repo(7)]})
    assert len(ranked) == 1
    assert ranked[0].found_by == ("q1", "q2", "q3")


def test_found_by_is_sorted_by_query_number_not_lexicographically():
    """q10 идёт после q2: сортировка по числу, иначе порядок «q1, q10, q2»."""
    results = {f"q{n}": [repo(7)] for n in (2, 10, 1)}
    assert rank(results)[0].found_by == ("q1", "q2", "q10")


def test_dedup_key_is_repo_id_not_full_name():
    """Репозиторий могли переименовать между запросами — id не меняется."""
    ranked = rank({"q1": [repo(7, full_name="old/name")], "q2": [repo(7, full_name="new/name")]})
    assert len(ranked) == 1
    assert ranked[0].found_by == ("q1", "q2")


def test_item_without_integer_id_is_skipped():
    ranked = rank({"q1": [{"full_name": "broken/item"}, repo(1)]})
    assert ids(ranked) == [1]


# --------------------------------------------------------------------------
# Схлопывание форков
# --------------------------------------------------------------------------


def test_fork_collapses_to_parent_by_parent_id():
    parent = repo(1)
    child = repo(2, fork=True, parent={"id": 1})
    ranked = rank({"q1": [parent], "q2": [child]})

    assert ids(ranked) == [1]
    assert ranked[0].found_by == ("q1", "q2")


def test_fork_collapses_by_name_when_parent_id_absent():
    """Выдача Search API поля `parent` не содержит — работает эвристика по имени.

    Случай `camelot-dev/camelot` + `atlanhq/camelot` из PROGRESS.md.
    """
    parent = repo(1, full_name="camelot-dev/camelot", name="camelot", stargazers_count=3000)
    child = repo(2, full_name="atlanhq/camelot", name="camelot", fork=True, stargazers_count=200)
    ranked = rank({"q1": [parent], "q2": [child]})

    assert ids(ranked) == [1]
    assert ranked[0].found_by == ("q1", "q2")


def test_fork_survives_when_same_name_repo_has_fewer_stars():
    """Одноимённый, но менее популярный — не родитель, а совпадение имени."""
    other = repo(1, name="parser", stargazers_count=5)
    child = repo(2, name="parser", fork=True, stargazers_count=900)
    ranked = rank({"q1": [other], "q2": [child]})

    assert sorted(ids(ranked)) == [1, 2]


def test_fork_survives_when_parent_is_not_in_results():
    """Форк без родителя в выдаче остаётся (QUERIES.md, шаг 3)."""
    child = repo(2, fork=True, parent={"id": 999})
    ranked = rank({"q1": [child]})
    assert ids(ranked) == [2]


def test_known_parent_id_outweighs_name_match():
    """Родитель известен и его в выдаче нет — эвристика по имени не подключается."""
    homonym = repo(1, name="repo2", stargazers_count=5000)
    child = repo(2, name="repo2", fork=True, parent={"id": 999})
    ranked = rank({"q1": [homonym], "q2": [child]})
    assert sorted(ids(ranked)) == [1, 2]


def test_fork_of_fork_collapses_to_final_parent():
    root = repo(1)
    middle = repo(2, fork=True, parent={"id": 1})
    leaf = repo(3, fork=True, parent={"id": 2})
    ranked = rank({"q1": [root], "q2": [middle], "q3": [leaf]})

    assert ids(ranked) == [1]
    assert ranked[0].found_by == ("q1", "q2", "q3")


def test_collapsed_fork_contributes_its_best_rank():
    """Ранги форка переходят родителю: найденное форком нашли и родителя."""
    parent = repo(1)
    child = repo(2, fork=True, parent={"id": 1})
    ranked = rank({"q1": [repo(9), repo(9998), parent], "q2": [child]})

    expected = 1 / (K_RRF + 3) + 1 / (K_RRF + 1)
    assert by_id(ranked, 1).rrf_score == pytest.approx(1.0)
    assert by_id(ranked, 1).rrf_raw == pytest.approx(expected)


# --------------------------------------------------------------------------
# Жёсткое отсечение до ранжирования (QUERIES.md, шаг 3, пункт 4)
# --------------------------------------------------------------------------


def test_archived_repo_is_dropped():
    assert ids(rank({"q1": [repo(1, archived=True), repo(2)]})) == [2]


def test_repo_with_zero_size_is_dropped():
    assert ids(rank({"q1": [repo(1, size=0), repo(2)]})) == [2]


def test_repo_without_language_description_and_topics_is_dropped():
    naked = repo(1, language=None, description=None, topics=[])
    assert ids(rank({"q1": [naked, repo(2)]})) == [2]


@pytest.mark.parametrize(
    "surviving",
    [
        {"language": "Rust", "description": None, "topics": []},
        {"language": None, "description": "делает дело", "topics": []},
        {"language": None, "description": None, "topics": ["asr"]},
    ],
)
def test_any_single_metadata_field_saves_repo(surviving):
    assert ids(rank({"q1": [repo(1, **surviving)]})) == [1]


def test_blank_description_does_not_count_as_metadata():
    naked = repo(1, language=None, description="   ", topics=[])
    assert rank({"q1": [naked]}) == []


def test_hard_cut_does_not_shift_ranks_of_survivors():
    """Позиции для RRF — из исходной выдачи, а не из того, что осталось после фильтров.

    Отсеянный лидер не двигает соседа на первое место: ранг — это то, каким его
    увидел GitHub. Иначе фильтр молча подкручивал бы скоры выживших.
    """
    ranked = rank({"q1": [repo(1, archived=True), repo(2)]})

    assert ids(ranked) == [2]
    assert ranked[0].rrf_raw == pytest.approx(1 / (K_RRF + 2))


def test_archived_parent_takes_live_fork_down_with_it():
    """Схлопывание форков идёт до жёсткого отсечения (QUERIES.md, шаг 3): порядок пунктов.

    Следствие: живой форк архивного родителя схлопывается в него и выпадает вместе
    с ним. Тест фиксирует поведение как есть — не чтобы его одобрить, а чтобы смена
    порядка шагов не прошла молча (`decisions_log.md`, 2026-09-04).
    """
    parent = repo(1, archived=True)
    child = repo(2, fork=True, parent={"id": 1})

    assert rank({"q1": [parent], "q2": [child]}) == []


def test_dropped_candidates_are_logged_with_reason():
    events = []

    class Recorder:
        def info(self, event, **fields):
            events.append((event, fields))

        def error(self, event, **fields):
            events.append((event, fields))

    rank({"q1": [repo(1, archived=True)]}, logger=Recorder())
    assert ("candidate_dropped", {"full_name": "owner1/repo1", "reason": "archived"}) in events


# --------------------------------------------------------------------------
# Метаданный приор
# --------------------------------------------------------------------------


def test_prior_is_one_for_ideal_repo():
    """10000 звёзд, свежий пуш, лицензия, не форк — все четыре слагаемых полны."""
    ideal = repo(1, stargazers_count=10_000, pushed_at="2026-09-04T12:00:00Z")
    assert rank({"q1": [ideal]})[0].prior_score == pytest.approx(1.0)


def test_prior_is_zero_for_bare_repo():
    """Ноль звёзд, пуш старше 540 дней, без лицензии, форк — все слагаемые нулевые."""
    bare = repo(
        1,
        stargazers_count=0,
        pushed_at="2020-01-01T00:00:00Z",
        license=None,
        fork=True,
        parent={"id": 999},
    )
    assert rank({"q1": [bare]})[0].prior_score == pytest.approx(0.0)


def test_prior_star_component_is_logarithmic():
    """log10(100 + 1) / 4 ≈ 0.5004 → вклад звёзд 0.40 * 0.5004."""
    item = repo(1, stargazers_count=100, pushed_at="2020-01-01T00:00:00Z", license=None)
    expected = 0.40 * (2.0043213737826426 / 4) + 0.15
    assert rank({"q1": [item]})[0].prior_score == pytest.approx(expected)


def test_prior_freshness_is_linear_over_540_days():
    """Пуш 270 дней назад — ровно половина окна свежести."""
    item = repo(1, stargazers_count=0, license=None, pushed_at="2025-12-08T12:00:00Z")
    assert rank({"q1": [item]})[0].prior_score == pytest.approx(0.30 * 0.5 + 0.15)


def test_license_noassertion_counts_as_no_license():
    """NOASSERTION — «файл есть, но не опознан», определённой лицензии нет."""
    item = repo(1, license={"spdx_id": "NOASSERTION"})
    without = repo(2, license=None)
    ranked = rank({"q1": [item], "q2": [without]})
    assert by_id(ranked, 1).prior_score == pytest.approx(by_id(ranked, 2).prior_score)


def test_future_pushed_at_does_not_exceed_freshness_ceiling():
    """Часы GitHub впереди наших — clamp держит слагаемое в [0, 1]."""
    item = repo(1, pushed_at="2027-01-01T00:00:00Z")
    assert rank({"q1": [item]})[0].prior_score <= 1.0


# --------------------------------------------------------------------------
# Итоговый скор и сортировка
# --------------------------------------------------------------------------


def test_score_is_weighted_sum_of_rrf_and_prior():
    ranked = rank({"q1": [repo(1), repo(2)]})
    for item in ranked:
        expected = RRF_WEIGHT * item.rrf_score + PRIOR_WEIGHT * item.prior_score
        assert item.score == pytest.approx(expected)


def test_prior_breaks_tie_between_equally_found_repos():
    """Одинаковые ранги, разные метаданные — решает приор (против пустышек)."""
    strong = repo(1, stargazers_count=5000)
    weak = repo(2, stargazers_count=0, license=None, pushed_at="2021-01-01T00:00:00Z")
    assert ids(rank({"q1": [weak, strong], "q2": [strong, weak]}))[0] == 1


def test_equal_scores_are_ordered_by_repo_id_ascending():
    """Второй ключ сортировки — гарантия идемпотентности (QUERIES.md, шаг 4)."""
    ranked = rank({"q1": [repo(9), repo(4)], "q2": [repo(4), repo(9)]})
    assert ids(ranked) == [4, 9]


def test_ranks_are_dense_and_start_at_one():
    ranked = rank({"q1": [repo(1), repo(2), repo(3)]})
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_limit_cuts_the_tail_after_sorting():
    ranked = rank({"q1": [repo(n) for n in range(1, 11)]}, limit=3)
    assert len(ranked) == 3
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_ranking_is_deterministic():
    results = {"q1": [repo(3), repo(1)], "q2": [repo(1), repo(2)]}
    first = rank(results)
    second = rank(results)
    assert [(i.repo["id"], i.rank, i.score) for i in first] == [
        (i.repo["id"], i.rank, i.score) for i in second
    ]


# --------------------------------------------------------------------------
# Сборка Candidate (SCHEMAS.md §4)
# --------------------------------------------------------------------------


def test_to_candidate_fills_contract_fields():
    ranked = rank({"q1": [repo(1)], "q2": [repo(1)]})[0]
    candidate = to_candidate(ranked, head_sha="a" * 40, retrieved_at=RETRIEVED_AT)

    assert candidate.repo_id == 1
    assert candidate.full_name == "owner1/repo1"
    assert candidate.found_by == ["q1", "q2"]
    assert candidate.rank == 1
    assert candidate.head_sha == "a" * 40
    assert candidate.retrieved_at == RETRIEVED_AT
    assert candidate.license_spdx == "MIT"
    assert candidate.rrf_score == pytest.approx(ranked.rrf_score)
    assert candidate.prior_score == pytest.approx(ranked.prior_score)


def test_to_candidate_truncates_overlong_description():
    """maxLength 500 в схеме: обрезаем, а не роняем скан из-за чужого описания."""
    ranked = rank({"q1": [repo(1, description="д" * 800)]})[0]
    candidate = to_candidate(ranked, head_sha="b" * 40, retrieved_at=RETRIEVED_AT)
    assert len(candidate.description) == 500


def test_to_candidate_keeps_first_twenty_topics():
    ranked = rank({"q1": [repo(1, topics=[f"t{n}" for n in range(30)])]})[0]
    candidate = to_candidate(ranked, head_sha="c" * 40, retrieved_at=RETRIEVED_AT)
    assert candidate.topics == [f"t{n}" for n in range(20)]


def test_to_candidate_passes_noassertion_license_as_null():
    ranked = rank({"q1": [repo(1, license={"spdx_id": "NOASSERTION"})]})[0]
    candidate = to_candidate(ranked, head_sha="d" * 40, retrieved_at=RETRIEVED_AT)
    assert candidate.license_spdx is None


def test_to_candidate_rejects_repo_without_default_branch():
    """Без ветки нечего спрашивать у git/commits — это баг выдачи, а не норма."""
    ranked = rank({"q1": [repo(1, default_branch=None)]})[0]
    with pytest.raises(RankingError):
        to_candidate(ranked, head_sha="e" * 40, retrieved_at=RETRIEVED_AT)
