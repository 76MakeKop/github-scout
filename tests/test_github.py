"""Клиент GitHub: таблица обработки ошибок из ARCHITECTURE.md.

Сети здесь нет — транспорт подменён. Пауз тоже нет: sleep пишет в список,
поэтому тест на 429 идёт за миллисекунды, а не за минуту.
"""

import json

import pytest

from scout.github import (
    MAX_TREE_PATHS,
    RATE_LIMIT_BUFFER_SECONDS,
    SEARCH_THROTTLE_SECONDS,
    GitHubAuth,
    GitHubClient,
    GitHubError,
    GitHubUnavailable,
    RateLimitExhausted,
    SearchQuotaExceeded,
)
from scout.http import HttpError, HttpResponse


class FakeTransport:
    """Отдаёт заранее заготовленные ответы по очереди и помнит все вызовы."""

    def __init__(self, *responses: HttpResponse) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, method, url, headers, body, timeout) -> HttpResponse:
        self.calls.append(url)
        self.headers = headers
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def ok(payload) -> HttpResponse:
    return HttpResponse(status=200, headers={}, body=json.dumps(payload).encode())


def make_client(transport, **kwargs):
    slept: list[float] = []
    client = GitHubClient(
        token="ghp_testtoken",
        transport=transport,
        sleep=slept.append,
        now=lambda: 1_000_000.0,
        **kwargs,
    )
    return client, slept


# --- успешный путь --------------------------------------------------------


def test_search_returns_items_and_counts_calls():
    transport = FakeTransport(ok({"items": [{"id": 1, "full_name": "a/b"}]}))
    client, _ = make_client(transport)

    items = client.search_repositories("pdf table extraction language:python")

    assert items == [{"id": 1, "full_name": "a/b"}]
    assert client.search_calls == 1
    assert "q=pdf+table+extraction+language%3Apython" in transport.calls[0]


def test_best_match_omits_sort_parameter():
    """GitHub сортирует по релевантности, когда sort не передан."""
    transport = FakeTransport(ok({"items": []}))
    client, _ = make_client(transport)

    client.search_repositories("pdf", sort="best-match")

    assert "sort=" not in transport.calls[0]


def test_stars_sort_is_passed_through():
    transport = FakeTransport(ok({"items": []}))
    client, _ = make_client(transport)

    client.search_repositories("topic:pdf", sort="stars")

    assert "sort=stars" in transport.calls[0]
    assert "order=desc" in transport.calls[0]


def test_token_goes_into_authorization_header():
    transport = FakeTransport(ok({"items": []}))
    client, _ = make_client(transport)

    client.search_repositories("pdf")

    assert transport.headers["Authorization"] == "Bearer ghp_testtoken"


# --- 429 с Retry-After (критерий приёмки дня 2) ---------------------------


def test_429_with_retry_after_waits_and_retries():
    transport = FakeTransport(
        HttpResponse(status=429, headers={"Retry-After": "7"}, body=b"slow down"),
        ok({"items": [{"id": 42}]}),
    )
    client, slept = make_client(transport)

    items = client.search_repositories("pdf")

    assert items == [{"id": 42}]
    assert slept == [7.0 + RATE_LIMIT_BUFFER_SECONDS]
    assert len(transport.calls) == 2


def test_403_with_ratelimit_reset_waits_until_reset():
    """Без Retry-After пауза считается по X-RateLimit-Reset минус текущее время."""
    transport = FakeTransport(
        HttpResponse(
            status=403,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1000045"},
            body=b"rate limit exceeded",
        ),
        ok({"items": []}),
    )
    client, slept = make_client(transport)

    client.search_repositories("pdf")

    assert slept == [45.0 + RATE_LIMIT_BUFFER_SECONDS]


def test_rate_limit_gives_up_after_three_attempts():
    transport = FakeTransport(
        HttpResponse(status=429, headers={"Retry-After": "1"}, body=b"slow down")
    )
    client, slept = make_client(transport)

    with pytest.raises(RateLimitExhausted):
        client.search_repositories("pdf")

    assert len(slept) == 3


# --- вторичный лимит ------------------------------------------------------


def test_secondary_rate_limit_backs_off_with_jitter():
    transport = FakeTransport(
        HttpResponse(status=403, headers={}, body=b"You have exceeded a secondary rate limit"),
        ok({"items": []}),
    )
    client, slept = make_client(transport)

    client.search_repositories("pdf")

    assert len(slept) == 1
    assert 0.8 <= slept[0] <= 1.2  # 1 с с джиттером +-20%


def test_secondary_rate_limit_gives_up_after_three_delays():
    transport = FakeTransport(HttpResponse(status=403, headers={}, body=b"secondary rate limit"))
    client, slept = make_client(transport)

    with pytest.raises(RateLimitExhausted):
        client.search_repositories("pdf")

    assert len(slept) == 3
    assert slept[0] < slept[1] < slept[2]


def test_plain_403_is_not_retried():
    """403 без признаков лимита — это отказ в доступе, повтор не поможет."""
    transport = FakeTransport(HttpResponse(status=403, headers={}, body=b"resource is restricted"))
    client, slept = make_client(transport)

    with pytest.raises(GitHubError):
        client.search_repositories("pdf")

    assert slept == []


# --- отклонённый токен ----------------------------------------------------


def test_401_is_a_credential_error_not_a_retry():
    transport = FakeTransport(HttpResponse(status=401, headers={}, body=b"Bad credentials"))
    client, slept = make_client(transport)

    with pytest.raises(GitHubAuth):
        client.get_repo("a/b")

    assert slept == []
    assert len(transport.calls) == 1


def test_403_bad_credentials_is_told_apart_from_a_rate_limit():
    """У «Bad credentials» и у лимита один статус 403, а лечение противоположное."""
    transport = FakeTransport(HttpResponse(status=403, headers={}, body=b"Bad credentials"))
    client, slept = make_client(transport)

    with pytest.raises(GitHubAuth):
        client.search_repositories("pdf")

    assert slept == []


# --- лимиты и троттлинг ---------------------------------------------------


def test_search_quota_is_enforced():
    """CLAUDE.md, запрет 5: не более 10 поисковых запросов на скан."""
    transport = FakeTransport(ok({"items": []}))
    client, _ = make_client(transport, max_search_queries=3)

    for _ in range(3):
        client.search_repositories("pdf")

    with pytest.raises(SearchQuotaExceeded):
        client.search_repositories("pdf")


def test_second_search_is_throttled():
    """Между поисковыми вызовами выдерживается пауза: now заморожен, значит ждём целиком."""
    transport = FakeTransport(ok({"items": []}))
    client, slept = make_client(transport)

    client.search_repositories("pdf")
    client.search_repositories("pdf tables")

    assert slept == [SEARCH_THROTTLE_SECONDS]


def test_per_page_is_capped_at_thirty():
    transport = FakeTransport(ok({"items": []}))
    client, _ = make_client(transport)

    client.search_repositories("pdf", per_page=100)

    assert "per_page=30" in transport.calls[0]


# --- прочие ручки ---------------------------------------------------------


def test_missing_repo_returns_none():
    """Репозиторий удалён или стал приватным между поиском и аудитом."""
    transport = FakeTransport(HttpResponse(status=404, headers={}, body=b"{}"))
    client, slept = make_client(transport)

    assert client.get_repo("gone/repo") is None
    assert slept == []  # повторять нечего: репозитория для нас больше нет
    assert len(transport.calls) == 1


def test_451_is_treated_like_a_missing_repo():
    """Repository unavailable for legal reasons: смысл тот же, что у 404."""
    transport = FakeTransport(HttpResponse(status=451, headers={}, body=b"unavailable"))
    client, slept = make_client(transport)

    assert client.get_file("dmca/repo", "README.md") is None
    assert slept == []


def test_tree_is_truncated_to_context_budget():
    payload = {"tree": [{"path": f"f{i}.py"} for i in range(500)], "truncated": True}
    transport = FakeTransport(ok(payload))
    client, _ = make_client(transport)

    tree = client.get_tree("a/b", "a1b2c3d")

    assert len(tree) == MAX_TREE_PATHS


def test_get_file_decodes_base64():
    body = {"type": "file", "encoding": "base64", "content": "cHJpdmV0"}
    transport = FakeTransport(ok(body))
    client, _ = make_client(transport)

    assert client.get_file("a/b", "README.md", ref="main") == "privet"


def test_get_head_sha():
    transport = FakeTransport(ok({"sha": "a1b2c3d4e5f6"}))
    client, _ = make_client(transport)

    assert client.get_head_sha("a/b", "main") == "a1b2c3d4e5f6"


def test_server_error_is_retried_then_raised():
    transport = FakeTransport(HttpResponse(status=503, headers={}, body=b"unavailable"))
    client, slept = make_client(transport)

    with pytest.raises(GitHubError):
        client.get_repo("a/b")

    assert len(slept) == 3


# --- обрыв связи и таймаут ------------------------------------------------


class FlakyTransport:
    """Первые `failures` вызовов обрываются на транспорте, дальше — обычный ответ."""

    def __init__(self, failures: int, response: HttpResponse | None = None) -> None:
        self.failures = failures
        self.response = response or ok({"sha": "a1b2c3d4e5f6"})
        self.calls = 0

    def __call__(self, method, url, headers, body, timeout) -> HttpResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise HttpError(f"сеть недоступна: обрыв на {url}")
        return self.response


def test_transport_error_is_retried_like_a_server_error():
    """Обрыв связи стоит в одной строке таблицы с 5xx: три повтора, затем отказ.

    До дня 8 `HttpError` пролетал мимо `_request` голым исключением: один обрыв
    из сотни вызовов `core` ронял весь скан, хотя следующая попытка прошла бы.
    """
    transport = FlakyTransport(failures=2)
    client, slept = make_client(transport)

    assert client.get_head_sha("a/b", "main") == "a1b2c3d4e5f6"
    assert len(slept) == 2
    assert slept[0] < slept[1]


def test_permanent_transport_error_becomes_github_error():
    transport = FlakyTransport(failures=99)
    client, slept = make_client(transport)

    with pytest.raises(GitHubUnavailable):
        client.get_head_sha("a/b", "main")

    assert len(slept) == 3
    assert transport.calls == 4  # первая попытка плюс три повтора


def test_transport_retry_does_not_eat_the_rate_limit_budget():
    """Обрыв и лимит считаются раздельно: у каждого своя причина и свой счётчик."""
    transport = FlakyTransport(
        failures=1,
        response=HttpResponse(status=429, headers={"Retry-After": "5"}, body=b"rate limited"),
    )
    client, slept = make_client(transport)

    with pytest.raises(RateLimitExhausted):
        client.get_repo("a/b")

    assert slept[0] < 2.0  # первый — джиттер обрыва
    waited = 5.0 + RATE_LIMIT_BUFFER_SECONDS
    assert slept[1:] == [waited, waited, waited]  # дальше — три попытки по Retry-After
