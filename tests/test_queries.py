"""Генератор поисковых запросов: семь семейств, правила сборки, golden-тест.

Сети здесь нет и быть не может: генератор — чистая функция от `Intent` и момента
времени. Момент передаётся явно, иначе golden-тест не был бы воспроизводим.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from scout.queries import (
    MAX_BOOLEAN_OPERATORS,
    MAX_QUERY_LENGTH,
    MIN_QUERIES,
    PER_PAGE,
    QueryGenerationError,
    build_query_set,
)
from scout.schemas import Intent, ModelName, QueryFamily, SearchQuerySet, SortOrder

FIXTURES = Path(__file__).parent / "fixtures"

REQUEST_ID = UUID("6f1f1b9c-0000-4000-8000-000000000001")
GENERATED_AT = datetime(2026, 9, 3, 12, 0, 0, tzinfo=UTC)


def make_intent(**overrides) -> Intent:
    """Эталонный интент примера из QUERIES.md; поля перекрываются точечно."""
    fields = {
        "request_id": REQUEST_ID,
        "task": "извлечение таблиц из PDF в структурированный вид",
        "domain": ["pdf", "table-extraction"],
        "languages": ["python"],
        "must_have": ["сохранение строк и столбцов"],
        "nice_to_have": ["экспорт в CSV"],
        "exclude": ["платные SaaS-обёртки"],
        "synonyms": ["pdf table extraction", "extract tables from pdf", "pdf table parser"],
        "known_libraries": ["camelot", "tabula", "pdfplumber", "unstructured"],
        "model": ModelName.FLASH,
        "prompt_version": "intent-1",
    }
    fields.update(overrides)
    return Intent(**fields)


def build(**overrides) -> SearchQuerySet:
    return build_query_set(make_intent(**overrides), generated_at=GENERATED_AT)


def families(query_set: SearchQuerySet) -> list[QueryFamily]:
    return [q.family for q in query_set.queries]


def by_family(query_set: SearchQuerySet, family: QueryFamily):
    return next((q for q in query_set.queries if q.family is family), None)


# --------------------------------------------------------------------------
# Golden-тест (критерий приёмки дня 3 в ROADMAP.md)
# --------------------------------------------------------------------------


def test_golden_query_set_matches_fixture_byte_for_byte():
    """Фиксированный Intent → побайтово тот же SearchQuerySet.

    Ожидаемый файл написан вручную по примеру из QUERIES.md, а не снят с вывода:
    иначе тест проверял бы сам себя.
    """
    expected = (FIXTURES / "query_set_pdf.json").read_text(encoding="utf-8")
    assert build().model_dump_json(indent=2) + "\n" == expected


def test_generation_is_deterministic():
    assert build().model_dump_json() == build().model_dump_json()


def test_queries_md_example_block_matches_generator():
    """Пример в QUERIES.md сверяется с кодом, а не живёт своей жизнью.

    Дважды пример расходился с правилами того же документа (сортировка `topic`,
    затем `archived:false` и `core_noun`). Оба раза расхождение нашлось глазами.
    Этот тест закрывает класс целиком.
    """
    doc = (Path(__file__).parents[1] / "QUERIES.md").read_text(encoding="utf-8")
    block = doc.split("### Пример:")[1].split("```")[1].strip("\n")

    query_set = build()
    width = max(len(q.q) for q in query_set.queries)
    expected = "\n".join(
        f"{q.id} {q.family.value:<9}{q.q:<{width}}  {q.sort.value:<11}{q.per_page}"
        for q in query_set.queries
    )
    assert block == expected


# --------------------------------------------------------------------------
# Состав и порядок семейств
# --------------------------------------------------------------------------


def test_seven_families_in_documented_order():
    assert families(build()) == [
        QueryFamily.EXACT,
        QueryFamily.SYNONYM,
        QueryFamily.TOPIC,
        QueryFamily.LIBRARY,
        QueryFamily.README,
        QueryFamily.BROAD,
        QueryFamily.RECENT,
    ]


def test_ids_are_sequential_without_gaps():
    query_set = build(domain=[], known_libraries=[])
    assert [q.id for q in query_set.queries] == [f"q{i}" for i in range(1, 6)]


def test_per_page_is_thirty_everywhere():
    assert {q.per_page for q in build().queries} == {PER_PAGE}


def test_generator_version_and_request_id_come_from_intent():
    query_set = build()
    assert query_set.request_id == REQUEST_ID
    assert query_set.generator_version == "qg-1"
    assert query_set.generated_at == GENERATED_AT


# --------------------------------------------------------------------------
# Правила сборки (QUERIES.md, шаг 2)
# --------------------------------------------------------------------------


def test_language_qualifier_omitted_when_languages_empty():
    query_set = build(languages=[])
    assert all("language:" not in q.q for q in query_set.queries)


def test_broad_raises_star_threshold_without_language():
    """Вырожденный случай QUERIES.md: пустой languages → broad получает stars:>100."""
    assert "stars:>50" in by_family(build(), QueryFamily.BROAD).q
    assert "stars:>100" in by_family(build(languages=[]), QueryFamily.BROAD).q


def test_only_first_language_is_used():
    """Квалификаторы GitHub объединяются по И: два language: дали бы пустую выдачу."""
    query_set = build(languages=["python", "rust"])
    assert all("language:rust" not in q.q for q in query_set.queries)
    assert "language:python" in by_family(query_set, QueryFamily.EXACT).q


def test_language_with_space_is_quoted():
    query_set = build(languages=["Jupyter Notebook"])
    assert 'language:"jupyter notebook"' in by_family(query_set, QueryFamily.EXACT).q


def test_archived_false_in_every_family_except_library():
    for query in build().queries:
        if query.family is QueryFamily.LIBRARY:
            assert "archived:" not in query.q
        else:
            assert "archived:false" in query.q


def test_license_qualifier_is_never_used():
    """CLAUDE.md, запрет 4: лицензия — атрибут отчёта, а не фильтр поиска."""
    for languages in ([], ["python"]):
        assert all("license:" not in q.q for q in build(languages=languages).queries)


def test_stars_qualifier_only_in_broad():
    for query in build().queries:
        assert ("stars:" in query.q) is (query.family is QueryFamily.BROAD)


def test_sort_mapping_follows_queries_md():
    expected = {
        QueryFamily.EXACT: SortOrder.BEST_MATCH,
        QueryFamily.SYNONYM: SortOrder.BEST_MATCH,
        QueryFamily.README: SortOrder.BEST_MATCH,
        QueryFamily.TOPIC: SortOrder.STARS,
        QueryFamily.LIBRARY: SortOrder.STARS,
        QueryFamily.BROAD: SortOrder.STARS,
        QueryFamily.RECENT: SortOrder.UPDATED,
    }
    assert {q.family: q.sort for q in build().queries} == expected


def test_recent_window_is_540_days_back_from_generated_at():
    assert "pushed:>2025-03-12" in by_family(build(), QueryFamily.RECENT).q


def test_recent_window_moves_with_generated_at():
    query_set = build_query_set(make_intent(), generated_at=datetime(2027, 1, 1, tzinfo=UTC))
    assert "pushed:>2025-07-10" in by_family(query_set, QueryFamily.RECENT).q


# --------------------------------------------------------------------------
# Семейство topic
# --------------------------------------------------------------------------


def test_topic_uses_first_two_domain_entries():
    expected = "topic:pdf topic:table-extraction archived:false"
    assert by_family(build(), QueryFamily.TOPIC).q == expected


def test_topic_with_single_domain_entry():
    assert by_family(build(domain=["pdf"]), QueryFamily.TOPIC).q == "topic:pdf archived:false"


def test_topic_values_are_slugified():
    query_set = build(domain=["Data Extraction", "PDF"])
    assert by_family(query_set, QueryFamily.TOPIC).q.startswith("topic:data-extraction topic:pdf")


def test_topic_skipped_when_domain_empty():
    query_set = build(domain=[])
    assert QueryFamily.TOPIC not in families(query_set)
    assert len(query_set.queries) >= MIN_QUERIES


def test_topic_skipped_when_domain_has_no_latin_characters():
    """Домен по-русски топиком GitHub быть не может — семейство просто отпадает."""
    assert QueryFamily.TOPIC not in families(build(domain=["извлечение"]))


# --------------------------------------------------------------------------
# Семейство library
# --------------------------------------------------------------------------


def test_library_joins_hypotheses_with_or():
    assert by_family(build(), QueryFamily.LIBRARY).q == (
        "camelot OR tabula OR pdfplumber OR unstructured in:name,description"
    )


def test_library_skipped_when_no_hypotheses():
    query_set = build(known_libraries=[])
    assert QueryFamily.LIBRARY not in families(query_set)
    assert len(query_set.queries) >= MIN_QUERIES


def test_library_names_with_spaces_are_quoted():
    query_set = build(known_libraries=["pdf tools", "camelot"])
    assert by_family(query_set, QueryFamily.LIBRARY).q.startswith('"pdf tools" OR camelot')


def test_library_is_trimmed_to_fit_query_length_limit():
    long_names = [f"library-name-number-{i:02d}-with-padding" for i in range(8)]
    query = by_family(build(known_libraries=long_names), QueryFamily.LIBRARY)
    assert len(query.q) <= MAX_QUERY_LENGTH
    assert query.q.startswith(long_names[0])
    assert long_names[7] not in query.q


# --------------------------------------------------------------------------
# Границы и вырожденные случаи
# --------------------------------------------------------------------------


def test_every_query_fits_the_length_limit():
    assert all(len(q.q) <= MAX_QUERY_LENGTH for q in build().queries)


def test_overlong_text_query_is_dropped_not_truncated():
    """Обрезать фразу посередине — молча исказить запрос; лучше не отправлять его."""
    query_set = build(synonyms=["pdf table extraction", "x" * 300, "pdf table parser"])
    assert QueryFamily.SYNONYM not in families(query_set)
    assert all(len(q.q) <= MAX_QUERY_LENGTH for q in query_set.queries)


def test_duplicate_synonyms_do_not_produce_duplicate_queries():
    query_set = build(synonyms=["pdf table extraction", "pdf table extraction", "pdf table parser"])
    strings = [q.q for q in query_set.queries]
    assert len(strings) == len(set(strings))
    assert QueryFamily.SYNONYM not in families(query_set)


def test_never_exceeds_the_search_budget():
    """CLAUDE.md, запрет 5: не больше 10 поисковых запросов на скан."""
    assert len(build().queries) <= 10


def test_too_poor_intent_raises():
    """Меньше пяти различимых запросов — SearchQuerySet невалиден по схеме."""
    with pytest.raises(QueryGenerationError):
        build(synonyms=["x", "x", "x"], domain=[], known_libraries=[], languages=[])


def test_result_survives_schema_revalidation():
    payload = json.loads(build().model_dump_json())
    assert SearchQuerySet.model_validate(payload).queries[0].id == "q1"


def test_library_family_stays_within_the_boolean_operator_limit():
    """Search API отвечает 422 на запрос с более чем пятью AND/OR/NOT.

    Найдено пилотом golden-set 2026-09-05: интент с семью гипотезами давал шесть
    `OR`, GitHub отклонял запрос, и всё семейство `library` пропадало из выдачи
    молча — единственное семейство, которое ищет нишевые проекты по имени.
    """
    intent = make_intent(
        known_libraries=["one", "two", "three", "four", "five", "six", "seven", "eight"]
    )

    library = next(q for q in build_query_set(intent).queries if q.family.value == "library")

    assert library.q.count(" OR ") <= MAX_BOOLEAN_OPERATORS


def test_library_family_keeps_the_most_confident_hypotheses():
    """Гипотезы идут по убыванию уверенности модели — режется хвост, а не голова."""
    intent = make_intent(
        known_libraries=["camelot", "tabula", "pdfplumber", "fitz", "borb", "pypdf", "slate"]
    )

    library = next(q for q in build_query_set(intent).queries if q.family.value == "library")

    assert "camelot" in library.q
    assert "slate" not in library.q
