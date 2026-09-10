"""Живучесть конвейера: ни один сбой не роняет скан целиком.

Файл поперечный, а не по модулю: он проверяет не работу клиента GitHub
и не работу Слоя 1 по отдельности — это `test_github.py` и `test_screen.py`, —
а то, что отказ **внутри** одного шага остаётся отказом одного шага. Пул потоков
переживает поломку кандидата, CLI переживает поломку пула, пользователь получает
код возврата и внятную строку вместо traceback.

Соответствие строкам таблицы «Обработка ошибок» из `ARCHITECTURE.md` — в колонке
«Проверено» самой таблицы; здесь живут только сквозные случаи.
"""

import pytest

from conftest import screening_run
from scout import cli, pipeline
from scout.config import MissingCredential
from scout.deepseek import DeepSeekAuth
from scout.github import GitHubAuth, GitHubUnavailable
from scout.schemas import DropStage
from scout.search import SearchOutcome
from test_cli import QUERY, events
from test_screen import FakeDeepSeek, FakeGitHub, Recorder, answer, candidate, run

# --------------------------------------------------------------------------
# Пул потоков Слоя 1 переживает поломку кандидата
# --------------------------------------------------------------------------


class ExplodingGitHub(FakeGitHub):
    """README отдаётся всем, кроме одного репозитория: на нём — неожиданный сбой.

    Неожиданный намеренно: `RuntimeError` не значится ни в одной строке таблицы
    ошибок. Предвиденное поймает свой `except`, а живучесть проверяется как раз
    на том, чего никто не предусмотрел.
    """

    def __init__(self, victim: str, error: Exception | None = None):
        super().__init__()
        self.victim = victim
        self.error = error or RuntimeError("boom")

    def get_file(self, full_name, path, ref=None):
        if full_name == self.victim:
            raise self.error
        return super().get_file(full_name, path, ref=ref)


def test_crash_on_one_candidate_does_not_kill_the_pool():
    """Четверо разобраны, пятый потерян — а не «потеряны все пятеро»."""
    candidates = [candidate(n) for n in range(1, 6)]
    github = ExplodingGitHub(victim="owner3/repo3")

    outcome = run(candidates, github=github)

    assert len(outcome.result.results) == 4
    assert outcome.failed == ["owner3/repo3"]
    assert 3 not in outcome.result.passed


def test_crash_is_logged_with_the_repo_and_the_error_type():
    log = Recorder()
    run([candidate(1), candidate(2)], github=ExplodingGitHub(victim="owner1/repo1"), logger=log)

    crash = next(fields for event, fields in log.events if event == "screening_crash")
    assert crash["full_name"] == "owner1/repo1"
    assert crash["error_type"] == "RuntimeError"
    assert "boom" in crash["error"]


def test_tokens_spent_before_the_crash_are_still_counted():
    """Поломка не отменяет счёт: первый вызов состоялся, токены за него оплачены.

    Поэтому словарь счётчиков и живёт снаружи разбора: иначе стоимость скана
    занижалась бы ровно на тех кандидатах, которые сломались.
    """

    class CrashOnRetry(FakeDeepSeek):
        def chat_json(self, **kwargs):
            if self.calls:
                raise RuntimeError("оборвалось на повторе")
            return super().chat_json(**kwargs)

    # relevance вне диапазона 0..1 — схема не пропустит, будет повтор, а на нём слом.
    deepseek = CrashOnRetry(responses=[answer(relevance=5.0)])
    outcome = run([candidate(1)], deepseek=deepseek)

    assert outcome.failed == ["owner1/repo1"]
    assert outcome.result.token_usage.input_tokens == 1500


def test_transport_failure_costs_one_candidate_not_the_scan():
    """Строка таблицы «обрыв связи»: GitHub сдался, но только по этому репо."""
    github = ExplodingGitHub(
        victim="owner2/repo2", error=GitHubUnavailable("GitHub недоступен после 3 повторов")
    )

    outcome = run([candidate(n) for n in range(1, 4)], github=github)

    # README не прочитан — кандидат разбирается по метаданным и остаётся в игре.
    assert len(outcome.result.results) == 3
    assert outcome.failed == []


@pytest.mark.parametrize(
    "error",
    [
        MissingCredential("нет DEEPSEEK_API_KEY"),
        DeepSeekAuth("DeepSeek отклонил ключ (401)"),
        GitHubAuth("GitHub отклонил токен (401)"),
    ],
)
def test_rejected_key_is_not_swallowed_by_the_pool(error):
    """Ключ одинаков для всех пятидесяти кандидатов.

    Проглотить его как «модель не справилась» значит выдать пустой скан там,
    где чинить надо `.env` одной строкой.
    """
    github = ExplodingGitHub(victim="owner1/repo1", error=error)

    with pytest.raises(type(error)):
        run([candidate(1), candidate(2)], github=github)


# --------------------------------------------------------------------------
# CLI: частичный результат — это результат
# --------------------------------------------------------------------------


def partial_search(**outcome_fields):
    """Подмена поиска, отдающая заранее заданный `SearchOutcome`."""

    def collect(query_set, *, intent, github, limit, logger=None, **kwargs):
        return SearchOutcome(
            candidates=[candidate(1), candidate(2)],
            queries_used=[query.q for query in query_set.queries],
            **outcome_fields,
        )

    return collect


def test_partial_scan_exits_zero_and_says_so(ok_intent, offline, monkeypatch, capsys):
    """Потеря части данных не отменяет ответ, но и не скрывается."""
    monkeypatch.setattr(pipeline, "collect_candidates", partial_search(partial=True))

    assert cli.main(["scan", QUERY]) == 0

    assert "Результат неполный" in capsys.readouterr().out


def test_partial_flag_reaches_the_log(ok_intent, offline, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "collect_candidates", partial_search(partial=True))

    cli.main(["scan", QUERY])

    assert events(capsys)["scan_finished"]["partial"] is True


def test_clean_run_is_not_marked_partial(ok_intent, offline, capsys):
    assert cli.main(["scan", QUERY]) == 0

    captured = capsys.readouterr()
    assert "Результат неполный" not in captured.out


def test_failed_screening_makes_the_run_partial(ok_intent, offline, monkeypatch, capsys):
    """Кандидат, чей ответ дважды не прошёл схему, — потеря по нашей вине."""

    def with_failure(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id, failed=["owner9/repo9"])

    monkeypatch.setattr(pipeline, "screen", with_failure)

    assert cli.main(["scan", QUERY]) == 0

    finished = events(capsys)["scan_finished"]
    assert finished["partial"] is True
    assert finished["dropped"][0]["stage"] == DropStage.SCREENING.value
    assert finished["dropped"][0]["full_name"] == "owner9/repo9"


def test_dropped_candidates_are_shown_to_the_user(ok_intent, offline, monkeypatch, capsys):
    def with_failure(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id, failed=["owner9/repo9"])

    monkeypatch.setattr(pipeline, "screen", with_failure)
    cli.main(["scan", QUERY])

    out = capsys.readouterr().out
    assert "## Выбыли" in out
    assert "owner9/repo9" in out


def test_rejected_token_is_a_configuration_error(ok_intent, offline, monkeypatch, capsys):
    """401 от GitHub — не «сервис недоступен», а «почините .env»: код 3."""

    def explode(query_set, *, intent, github, limit, logger=None, **kwargs):
        raise GitHubAuth("GitHub отклонил токен (401): проверьте .env")

    monkeypatch.setattr(pipeline, "collect_candidates", explode)

    assert cli.main(["scan", QUERY]) == 3

    # Лог и сообщение человеку идут в один stderr, поэтому вычитываем его разом.
    captured = capsys.readouterr().err
    assert "Ошибка конфигурации" in captured
    assert "credential_rejected" in captured


def test_unexpected_crash_becomes_exit_one_not_a_traceback(ok_intent, offline, monkeypatch, capsys):
    """Последняя застава: наружу выходит код, а не стек вызовов."""

    def explode(query_set, *, intent, github, limit, logger=None, **kwargs):
        raise ZeroDivisionError("деление на ноль в чужом коде")

    monkeypatch.setattr(pipeline, "collect_candidates", explode)

    assert cli.main(["scan", QUERY]) == 1

    captured = capsys.readouterr()
    assert "Скан прерван: ZeroDivisionError" in captured.err
    assert "Traceback" not in captured.err


def test_build_recommendation_lists_the_queries_it_checked(ok_intent, offline, monkeypatch, capsys):
    """Строка таблицы: ни один кандидат не прошёл → BUILD и список запросов."""

    def none_passed(candidates, intent, *, request_id, github, logger=None, limit=10, **kwargs):
        return screening_run(request_id, passed=())

    monkeypatch.setattr(pipeline, "screen", none_passed)

    assert cli.main(["scan", QUERY]) == 0

    out = capsys.readouterr().out
    assert "BUILD" in out
    assert "Проверено запросов: 7." in out
    assert "language:python" in out


# --------------------------------------------------------------------------
# Идемпотентность
# --------------------------------------------------------------------------


def test_two_identical_runs_print_the_same_candidates(ok_intent, offline, capsys):
    """Критерий приёмки дня 8: два прогона подряд дают одинаковый список.

    Различаться обязан только `run_id`: он на то и есть, чтобы отличать прогоны
    друг от друга, — а вот кандидаты при неизменных данных обязаны совпасть.
    """
    cli.main(["scan", QUERY])
    first = capsys.readouterr().out

    cli.main(["scan", QUERY])
    second = capsys.readouterr().out

    assert first == second
