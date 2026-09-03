"""Клиент GitHub REST API.

Реализует таблицу обработки ошибок из ARCHITECTURE.md:
403/429 — ждать до сброса лимита (до 3 попыток), вторичный лимит — backoff
1/4/16 с с джиттером, поиск — не глубже одной страницы, не более 10 запросов
на скан, троттлинг 2 с между поисковыми вызовами.

Клонирования здесь нет и не будет: CLAUDE.md, запрет 1.
"""

import base64
import random
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

from scout import config
from scout.http import DEFAULT_TIMEOUT, HttpResponse, Transport, urllib_transport
from scout.log import RunLogger

API_ROOT = "https://api.github.com"
ACCEPT = "application/vnd.github+json"
API_VERSION = "2022-11-28"

MAX_TREE_PATHS = 300
SEARCH_THROTTLE_SECONDS = 2.0
SECONDARY_BACKOFF_SECONDS = (1.0, 4.0, 16.0)
MAX_RATE_LIMIT_ATTEMPTS = 3


class GitHubError(RuntimeError):
    """Ответ GitHub, который не лечится повтором."""


class RateLimitExhausted(GitHubError):
    """Лимит не сбросился за отведённые попытки."""


class SearchQuotaExceeded(GitHubError):
    """Превышен лимит поисковых запросов на один скан (CLAUDE.md, запрет 5)."""


def _jitter(base: float) -> float:
    return base * random.uniform(0.8, 1.2)


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        transport: Transport = urllib_transport,
        logger: RunLogger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        timeout: float = DEFAULT_TIMEOUT,
        max_search_queries: int = config.MAX_SEARCH_QUERIES,
    ) -> None:
        self._token = token
        self._transport = transport
        self._log = logger
        self._sleep = sleep
        self._now = now
        self._timeout = timeout
        self._max_search_queries = max_search_queries
        self.search_calls = 0
        self._last_search_at: float | None = None

    # -- внутреннее ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": ACCEPT,
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "github-scout/0.1",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _event(self, event: str, **fields: Any) -> None:
        if self._log is not None:
            self._log.info(event, **fields)

    def _retry_after_seconds(self, response: HttpResponse) -> float | None:
        """Пауза по заголовкам GitHub. None — заголовков нет, лимит ни при чём."""
        retry_after = response.header("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                return None

        remaining = response.header("X-RateLimit-Remaining")
        reset = response.header("X-RateLimit-Reset")
        if remaining == "0" and reset:
            try:
                return max(0.0, float(reset) - self._now())
            except ValueError:
                return None
        return None

    def _request(self, path: str, *, allow_404: bool = False) -> Any:
        url = path if path.startswith("http") else f"{API_ROOT}{path}"
        secondary_attempt = 0
        rate_limit_attempt = 0

        while True:
            response = self._transport("GET", url, self._headers(), None, self._timeout)

            if response.status == 200:
                return response.json()

            if response.status == 404 and allow_404:
                return None

            if response.status in (403, 429):
                delay = self._retry_after_seconds(response)
                if delay is not None:
                    rate_limit_attempt += 1
                    if rate_limit_attempt > MAX_RATE_LIMIT_ATTEMPTS:
                        raise RateLimitExhausted(
                            f"лимит GitHub не сбросился за {MAX_RATE_LIMIT_ATTEMPTS} попытки: {url}"
                        )
                    self._event(
                        "github_rate_limit",
                        url=url,
                        wait_seconds=round(delay, 1),
                        attempt=rate_limit_attempt,
                    )
                    self._sleep(delay)
                    continue

                if "secondary rate limit" in response.text().lower():
                    if secondary_attempt >= len(SECONDARY_BACKOFF_SECONDS):
                        raise RateLimitExhausted(f"вторичный лимит GitHub не отпустил: {url}")
                    delay = _jitter(SECONDARY_BACKOFF_SECONDS[secondary_attempt])
                    secondary_attempt += 1
                    self._event(
                        "github_secondary_limit",
                        url=url,
                        wait_seconds=round(delay, 1),
                        attempt=secondary_attempt,
                    )
                    self._sleep(delay)
                    continue

                raise GitHubError(f"{response.status} от GitHub: {response.text()[:200]}")

            if 500 <= response.status < 600:
                if secondary_attempt >= len(SECONDARY_BACKOFF_SECONDS):
                    raise GitHubError(f"GitHub отдаёт {response.status}: {url}")
                delay = _jitter(SECONDARY_BACKOFF_SECONDS[secondary_attempt])
                secondary_attempt += 1
                self._event("github_server_error", url=url, status=response.status)
                self._sleep(delay)
                continue

            raise GitHubError(f"{response.status} от GitHub: {response.text()[:200]}")

    # -- публичные методы ---------------------------------------------------

    def search_repositories(
        self,
        q: str,
        *,
        sort: str | None = None,
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        """Одна страница выдачи. Глубже не ходим: потолок Search API — 1000 результатов.

        `sort=None` соответствует `best-match` — GitHub сортирует по релевантности,
        когда параметр не передан.
        """
        if self.search_calls >= self._max_search_queries:
            raise SearchQuotaExceeded(
                f"исчерпан лимит поисковых запросов на скан: {self._max_search_queries}"
            )

        if self._last_search_at is not None:
            elapsed = self._now() - self._last_search_at
            if elapsed < SEARCH_THROTTLE_SECONDS:
                self._sleep(SEARCH_THROTTLE_SECONDS - elapsed)

        params: dict[str, str] = {"q": q, "per_page": str(min(per_page, 30))}
        if sort and sort != "best-match":
            params["sort"] = sort
            params["order"] = "desc"

        payload = self._request(f"/search/repositories?{urllib.parse.urlencode(params)}")
        self.search_calls += 1
        self._last_search_at = self._now()

        items = payload.get("items", []) if isinstance(payload, dict) else []
        self._event("github_search", q=q, sort=sort or "best-match", found=len(items))
        return items

    def get_repo(self, full_name: str) -> dict[str, Any] | None:
        """None — репозиторий удалён или стал приватным между поиском и аудитом."""
        return self._request(f"/repos/{full_name}", allow_404=True)

    def get_tree(self, full_name: str, sha: str) -> list[dict[str, Any]]:
        """Дерево файлов, обрезанное до MAX_TREE_PATHS — бюджет контекста Слоя 2."""
        payload = self._request(f"/repos/{full_name}/git/trees/{sha}?recursive=1", allow_404=True)
        if not payload:
            return []
        tree = payload.get("tree", [])
        if payload.get("truncated"):
            self._event("github_tree_truncated", full_name=full_name, returned=len(tree))
        return tree[:MAX_TREE_PATHS]

    def get_file(self, full_name: str, path: str, ref: str | None = None) -> str | None:
        """Содержимое текстового файла. None — файла нет или он не текстовый."""
        query = f"?ref={urllib.parse.quote(ref)}" if ref else ""
        payload = self._request(
            f"/repos/{full_name}/contents/{urllib.parse.quote(path)}{query}", allow_404=True
        )
        if not isinstance(payload, dict) or payload.get("type") != "file":
            return None

        encoding = payload.get("encoding")
        if encoding != "base64":
            return payload.get("content")

        try:
            return base64.b64decode(payload["content"]).decode("utf-8")
        except (KeyError, ValueError, UnicodeDecodeError):
            return None

    def get_head_sha(self, full_name: str, branch: str) -> str | None:
        """SHA головы ветки — ключ инвалидации кэша аудитов (ARCHITECTURE.md → «Кэш»)."""
        payload = self._request(
            f"/repos/{full_name}/commits/{urllib.parse.quote(branch)}", allow_404=True
        )
        return payload.get("sha") if isinstance(payload, dict) else None
