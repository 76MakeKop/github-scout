"""Отчёт и детектор регургитации (день 13).

Сборка отчёта — детерминированный код без единого вызова модели, поэтому
проверяется целиком: порядок пятёрки, рекомендация, provenance, обязательства
лицензии. Детектор проверяется на границе 120 символов — числе из
`ARCHITECTURE.md`, которое повторяется в `CHECKLIST.md`.
"""

from uuid import UUID

import pytest

from scout.regurgitation import (
    MIN_VERBATIM_CHARS,
    RegurgitationDetected,
    assert_clean,
    find_verbatim,
    normalize,
)
from scout.report import build_report, render_markdown
from scout.schemas import Verdict
from test_cache import SHA_A, audit_result
from test_evaluate import audit_for

REQUEST = UUID("6f1f1b9c-0000-4000-8000-0000000000aa")


def report_of(audits, **overrides):
    return build_report(
        request_id=REQUEST,
        query_text="нужен парсер PDF-таблиц на Python",
        audits=list(audits),
        urls={audit.full_name: f"https://github.com/{audit.full_name}" for audit in audits},
        **overrides,
    )


# --------------------------------------------------------------------------
# Порядок и отбор пятёрки
# --------------------------------------------------------------------------


def test_candidates_are_ordered_by_total_descending():
    report = report_of(
        [audit_for(1, total=0.60), audit_for(2, total=0.90), audit_for(3, total=0.75)]
    )

    assert [item.full_name for item in report.candidates] == [
        "owner2/repo2",
        "owner3/repo3",
        "owner1/repo1",
    ]


def test_ties_are_broken_by_repo_id_so_two_runs_line_up():
    """При равном `total` порядок обязан быть воспроизводимым: иначе сравнение
    «до и после» из `CHECKLIST.md` меряло бы порядок словаря."""
    report = report_of([audit_for(9, total=0.8), audit_for(2, total=0.8), audit_for(5, total=0.8)])

    assert [item.rank for item in report.candidates] == [1, 2, 3]
    assert [item.full_name for item in report.candidates] == [
        "owner2/repo2",
        "owner5/repo5",
        "owner9/repo9",
    ]


def test_report_never_holds_more_than_five():
    """`CLAUDE.md`, запрет 5, и `maxItems` в `SCHEMAS.md` §8."""
    report = report_of([audit_for(n, total=0.9 - n / 100) for n in range(1, 9)])

    assert len(report.candidates) == 5


def test_report_limit_can_be_narrowed_for_a_cheap_run():
    report = report_of([audit_for(n) for n in range(1, 5)], limit=2)

    assert len(report.candidates) == 2


# --------------------------------------------------------------------------
# Рекомендация
# --------------------------------------------------------------------------


def test_recommendation_is_the_verdict_of_the_first_candidate():
    """Второго суждения не заводим: `recommendation` и `candidates[0].verdict`
    разошлись бы при первой же правке порогов."""
    report = report_of([audit_for(1, verdict="FORK", total=0.9), audit_for(2, total=0.5)])

    assert report.recommendation is Verdict.FORK
    assert report.recommendation_target == "owner1/repo1"


def test_empty_report_recommends_build():
    """Подходящего решения не нашлось — это содержательный ответ, а не отказ."""
    report = report_of([])

    assert report.recommendation is Verdict.BUILD
    assert report.recommendation_target is None
    assert report.candidates == []


def test_build_verdict_has_no_target():
    """BUILD означает «писать самим», и указывать на репозиторий тут нечем."""
    report = report_of([audit_for(1, verdict="BUILD", total=0.4)])

    assert report.recommendation is Verdict.BUILD
    assert report.recommendation_target is None


# --------------------------------------------------------------------------
# Содержание кандидата
# --------------------------------------------------------------------------


def test_candidate_carries_passport_effort_and_provenance():
    """`CHECKLIST.md`: у каждого кандидата заполнен паспорт и непустой provenance."""
    report = report_of([audit_result()])
    item = report.candidates[0]

    assert item.license_passport.spdx_id == "MIT"
    assert item.integration_effort_days.likely == 2.0
    assert item.provenance
    assert item.head_sha == SHA_A


def test_weaknesses_put_gaps_before_risks():
    """Для читателя это одна колонка «на что смотреть», но порядок закреплён:
    сначала чего не хватает для задачи, потом чем это грозит потом."""
    report = report_of([audit_result()])

    assert report.candidates[0].weaknesses[0] == "нет OCR"
    assert "single-maintainer" in report.candidates[0].weaknesses[1]


def test_strengths_are_never_empty():
    """Контракт требует минимум одну строку, а модель может не найти покрытий.

    Отчёт не выдумывает достоинство: он честно говорит, что кандидат держится
    на оценке, — и это проверяемое утверждение, в отличие от придуманной фразы.
    """
    bare = audit_result()
    bare.fit.covers = []

    strengths = report_of([bare]).candidates[0].strengths

    assert len(strengths) == 1
    assert "Оценка соответствия" in strengths[0]


def test_url_comes_from_the_candidate_not_from_the_name():
    """GitHub переносит репозитории: собранный из имени адрес указывал бы
    на редирект вместо канонического."""
    audit = audit_for(1)
    report = build_report(
        request_id=REQUEST,
        query_text="задача",
        audits=[audit],
        urls={"owner1/repo1": "https://github.com/moved-org/repo1"},
    )

    assert str(report.candidates[0].html_url) == "https://github.com/moved-org/repo1"


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def test_markdown_shows_verdict_licence_and_provenance():
    text = render_markdown(report_of([audit_result()]))

    assert "**Рекомендация: USE**" in text
    assert "owner1/repo1" in text
    assert "MIT" in text
    assert "код брать можно" in text
    assert "**Откуда сведения:**" in text


def test_markdown_warns_about_partial_result():
    text = render_markdown(report_of([audit_result()], partial=True))

    assert "Результат неполный" in text


def test_markdown_of_an_empty_report_still_advises_build():
    text = render_markdown(report_of([]))

    assert "**Рекомендация: BUILD**" in text
    assert "писать самим" in text


def test_markdown_shows_licence_obligations():
    """`CHECKLIST.md`: отчёт без лицензионного паспорта не выдаётся."""
    text = render_markdown(report_of([audit_result()]))

    assert "**Обязательства лицензии:**" in text
    assert "сохранять текст лицензии" in text


def test_copyleft_candidate_is_marked_as_unusable_code():
    audit = audit_result()
    audit.license_passport.code_reuse_allowed = False
    audit.license_passport.spdx_id = "AGPL-3.0"

    assert "код брать нельзя" in render_markdown(report_of([audit]))


# --------------------------------------------------------------------------
# Детектор регургитации
# --------------------------------------------------------------------------

SOURCE = (
    "Этот проект извлекает таблицы из PDF-файлов, опираясь на положение слов "
    "на странице и линии разметки. Установка через pip, интерфейс командной "
    "строки прилагается, поддерживаются многостраничные документы."
)


def test_short_quote_is_allowed():
    """Имя функции и путь короче порога — детектор их не ловит."""
    assert find_verbatim("извлекает таблицы из PDF", SOURCE) is None


def test_long_verbatim_run_is_caught():
    borrowed = SOURCE[: MIN_VERBATIM_CHARS + 20]

    assert find_verbatim(borrowed, SOURCE) is not None


def test_detector_ignores_reflowed_whitespace():
    """Модель переносит строки; побайтовое сравнение пропускало бы ровно те
    случаи, ради которых детектор заводится."""
    borrowed = SOURCE[: MIN_VERBATIM_CHARS + 10].replace(" ", "\n  ")

    assert find_verbatim(borrowed, SOURCE) is not None


def test_detector_ignores_letter_case():
    assert find_verbatim(SOURCE[: MIN_VERBATIM_CHARS + 5].upper(), SOURCE) is not None


@pytest.mark.parametrize("length", [MIN_VERBATIM_CHARS - 1, MIN_VERBATIM_CHARS])
def test_the_border_is_exactly_120_characters(length):
    """Граница проверяется на ней самой: 119 — цитата, 120 — перенос."""
    borrowed = normalize(SOURCE)[:length]
    caught = find_verbatim(borrowed, SOURCE) is not None

    assert caught is (length >= MIN_VERBATIM_CHARS)


def test_assert_clean_names_the_offending_field():
    payload = {"fit": {"covers": ["своими словами"], "gaps": [SOURCE[:150]]}}

    with pytest.raises(RegurgitationDetected) as caught:
        assert_clean(payload, SOURCE)

    assert caught.value.field == "fit.gaps[0]"


def test_assert_clean_passes_a_paraphrase():
    payload = {
        "fit": {"covers": ["разбирает таблицы"], "gaps": ["не умеет OCR"]},
        "verdict_rationale": "Зрелая библиотека, решает задачу целиком.",
    }

    assert_clean(payload, SOURCE)


def test_detector_checks_prose_fields_too():
    """Закон не различает «процитировал код» и «процитировал документацию»."""
    payload = {"verdict_rationale": SOURCE[:140]}

    with pytest.raises(RegurgitationDetected):
        assert_clean(payload, SOURCE)


def test_long_risk_note_does_not_break_the_contract():
    """Живой прогон 2026-09-10: `Risk.note` в §7 допускает 300 символов, элемент
    `weaknesses` в §8 — только 200, и склейка «тип: заметка» роняла весь замер
    на первой же задаче. Слой отчёта обязан разрешать это рассогласование."""
    audit = audit_result()
    audit.risks[0].note = "з" * 300

    weakness = report_of([audit]).candidates[0].weaknesses[-1]

    assert len(weakness) <= 200
    assert weakness.endswith("…")


def test_long_gap_note_is_also_trimmed():
    audit = audit_result()
    audit.fit.gaps[0].note = "п" * 200

    assert all(len(text) <= 200 for text in report_of([audit]).candidates[0].weaknesses)


def test_short_notes_are_left_alone():
    """Обрезка не должна трогать то, что и так укладывается."""
    audit = audit_result()

    assert report_of([audit]).candidates[0].weaknesses[0] == "нет OCR"
