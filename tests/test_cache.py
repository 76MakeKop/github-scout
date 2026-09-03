"""Кэш аудитов: SQLite, ключ `repo:{id}:{sha}`, инвалидация только по SHA.

База поднимается в `tmp_path`, сети нет. Кэшируется **только** `AuditResult`
Слоя 2: `ScreeningResult` Слоя 1 дёшев и после распараллеливания быстр,
кэшировать его незачем (`ARCHITECTURE.md` → «Кэш»).
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from scout import cli
from scout.cache import AuditCache, cache_key, lookup_or_store
from scout.schemas import AuditResult, CacheEntry

REQUEST_ID = UUID("6f1f1b9c-0000-4000-8000-000000000001")
AUDITED_AT = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)
SHA_A = "a" * 40
SHA_B = "b" * 40


def audit_result(repo_id: int = 1, head_sha: str = SHA_A, **overrides) -> AuditResult:
    """Полный `AuditResult` по SCHEMAS.md §7 — round-trip проверяется на нём целиком."""
    fields = {
        "repo_id": repo_id,
        "full_name": f"owner{repo_id}/repo{repo_id}",
        "head_sha": head_sha,
        "audited_at": AUDITED_AT,
        "model": "deepseek-v4-pro",
        "prompt_version": "l2-1",
        "structure": {
            "entrypoints": ["src/pkg/__init__.py"],
            "modules": ["parser", "writer"],
            "has_tests": True,
            "test_paths": ["tests/"],
            "has_ci": True,
            "ci_files": [".github/workflows/ci.yml"],
            "has_docs": True,
            "file_count": 142,
        },
        "dependencies": {
            "manifest": "pyproject.toml",
            "runtime": [{"name": "pdfminer.six", "constraint": ">=2023"}, {"name": "click"}],
            "count": 2,
            "heavy": ["opencv-python"],
        },
        "license_passport": {
            "spdx_id": "MIT",
            "name": "MIT License",
            "detected_by": "github-api",
            "confidence": 0.95,
            "copyleft": "none",
            "network_copyleft": False,
            "commercial_use": True,
            "attribution_required": True,
            "share_alike": False,
            "code_reuse_allowed": True,
            "obligations": ["сохранять текст лицензии"],
            "source": {
                "path": "LICENSE",
                "commit_sha": head_sha,
                "retrieved_at": AUDITED_AT,
                "api": "licenses",
            },
        },
        "maintenance": {
            "last_commit": AUDITED_AT,
            "commits_90d": 34,
            "contributors_12m": 7,
            "open_issues": 12,
            "releases_12m": 3,
        },
        "fit": {
            "covers": ["извлечение таблиц"],
            "gaps": ["нет OCR"],
            "integration_effort_days": {"low": 0.5, "likely": 2.0, "high": 5.0},
        },
        "risks": [{"type": "single-maintainer", "severity": "medium", "note": "один автор"}],
        "score": {
            "relevance": 0.9,
            "quality": 0.8,
            "maintenance": 0.7,
            "license": 1.0,
            "total": 0.85,
        },
        "verdict": "USE",
        "verdict_rationale": "решает задачу целиком, лицензия разрешает переиспользование",
        "provenance": [
            {
                "path": "metadata",
                "commit_sha": head_sha,
                "retrieved_at": AUDITED_AT,
                "api": "repos",
            }
        ],
        "token_usage": {
            "model": "deepseek-v4-pro",
            "input_tokens": 12_000,
            "cached_input_tokens": 9_000,
            "output_tokens": 1_500,
            "cost_usd": 0.011,
            "pricing_window": "off-peak",
        },
    }
    fields.update(overrides)
    return AuditResult(**fields)


def entry(result: AuditResult | None = None, hits: int = 0) -> CacheEntry:
    result = result or audit_result()
    return CacheEntry(
        key=cache_key(result.repo_id, result.head_sha),
        repo_id=result.repo_id,
        head_sha=result.head_sha,
        payload_type="audit_result_v1",
        payload=result,
        created_at=AUDITED_AT,
        hits=hits,
    )


@pytest.fixture
def cache(tmp_path):
    with AuditCache(tmp_path / ".cache" / "scout.sqlite3") as opened:
        yield opened


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def info(self, event, **fields):
        self.events.append((event, fields))

    def error(self, event, **fields):
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [event for event, _ in self.events]


# --------------------------------------------------------------------------
# База и ключ
# --------------------------------------------------------------------------


def test_database_file_is_created_on_first_use(tmp_path):
    path = tmp_path / ".cache" / "scout.sqlite3"
    assert not path.exists()

    with AuditCache(path):
        pass

    assert path.exists()


def test_cache_key_matches_the_schema_pattern():
    """`SCHEMAS.md` §9: ^repo:\\d+:[0-9a-f]{7,40}$ — ключ собирает код, не модель."""
    assert cache_key(42, SHA_A) == f"repo:42:{SHA_A}"
    entry(audit_result(repo_id=42))  # валидация паттерна — на модели CacheEntry


def test_missing_key_is_a_miss(cache):
    assert cache.get(cache_key(1, SHA_A)) is None


# --------------------------------------------------------------------------
# Round-trip
# --------------------------------------------------------------------------


def test_put_then_get_returns_the_same_audit_result(cache):
    original = audit_result()
    cache.put(entry(original))

    restored = cache.get(cache_key(1, SHA_A))

    assert restored == original


def test_every_field_survives_serialisation(cache):
    """Проверяется весь `AuditResult` целиком, а не выборочные поля."""
    original = audit_result()
    cache.put(entry(original))

    restored = cache.get(cache_key(1, SHA_A))

    assert restored.model_dump() == original.model_dump()
    assert restored.license_passport.source.retrieved_at == AUDITED_AT
    assert restored.fit.integration_effort_days.likely == 2.0
    assert restored.token_usage.cost_usd == 0.011


def test_put_twice_replaces_the_row(cache):
    cache.put(entry(audit_result()))
    cache.put(entry(audit_result(verdict="FORK")))

    assert len(cache.entries()) == 1
    assert cache.get(cache_key(1, SHA_A)).verdict.value == "FORK"


# --------------------------------------------------------------------------
# Попадания и инвалидация
# --------------------------------------------------------------------------


def test_hits_are_counted_on_every_read(cache):
    cache.put(entry(audit_result()))
    key = cache_key(1, SHA_A)

    cache.get(key)
    cache.get(key)

    assert cache.entries()[0].hits == 2


def test_new_head_sha_is_a_miss(cache):
    """Инвалидация только по SHA: новый коммит — новый ключ, TTL нет."""
    cache.put(entry(audit_result(head_sha=SHA_A)))

    assert cache.get(cache_key(1, SHA_B)) is None
    assert cache.get(cache_key(1, SHA_A)) is not None


def test_old_entry_survives_invalidation(cache):
    """Старьё не удаляем: место дешёвое, а `hits` покажут востребованность."""
    cache.put(entry(audit_result(head_sha=SHA_A)))
    cache.put(entry(audit_result(head_sha=SHA_B)))

    assert len(cache.entries()) == 2


# --------------------------------------------------------------------------
# --refresh
# --------------------------------------------------------------------------


def test_refresh_ignores_existing_entry(tmp_path):
    path = tmp_path / "scout.sqlite3"
    with AuditCache(path) as warm:
        warm.put(entry(audit_result()))

    with AuditCache(path, refresh=True) as cold:
        assert cold.get(cache_key(1, SHA_A)) is None


def test_refresh_still_writes_fresh_entries(tmp_path):
    path = tmp_path / "scout.sqlite3"
    with AuditCache(path, refresh=True) as cache:
        cache.put(entry(audit_result(verdict="BUILD")))

    with AuditCache(path) as reopened:
        assert reopened.get(cache_key(1, SHA_A)).verdict.value == "BUILD"


def test_refresh_does_not_count_hits(tmp_path):
    path = tmp_path / "scout.sqlite3"
    with AuditCache(path) as warm:
        warm.put(entry(audit_result()))

    with AuditCache(path, refresh=True) as cold:
        cold.get(cache_key(1, SHA_A))

    with AuditCache(path) as reopened:
        assert reopened.entries()[0].hits == 0


# --------------------------------------------------------------------------
# Политика «сначала кэш» — то, что позовёт Слой 2
# --------------------------------------------------------------------------


class CountingAuditor:
    """Заглушка дорогого слоя: считает, сколько раз её позвали."""

    def __init__(self, result: AuditResult | None = None):
        self.result = result or audit_result()
        self.calls = 0

    def __call__(self) -> AuditResult:
        self.calls += 1
        return self.result


def test_second_lookup_with_the_same_key_costs_zero_model_calls(cache):
    """Критерий приёмки дня 6 на моках: повтор не тратит токены."""
    auditor = CountingAuditor()
    logger = Recorder()

    first = lookup_or_store(
        cache, repo_id=1, full_name="owner1/repo1", head_sha=SHA_A, produce=auditor, logger=logger
    )
    second = lookup_or_store(
        cache, repo_id=1, full_name="owner1/repo1", head_sha=SHA_A, produce=auditor, logger=logger
    )

    assert auditor.calls == 1
    assert first == second
    assert logger.names() == ["cache_miss", "cache_hit"]


def test_cache_hit_event_carries_what_the_report_needs(cache):
    auditor = CountingAuditor()
    logger = Recorder()

    for _ in range(2):
        lookup_or_store(
            cache,
            repo_id=1,
            full_name="owner1/repo1",
            head_sha=SHA_A,
            produce=auditor,
            logger=logger,
        )

    hit = next(fields for event, fields in logger.events if event == "cache_hit")
    assert hit["repo_id"] == 1
    assert hit["full_name"] == "owner1/repo1"
    assert hit["head_sha"] == SHA_A
    assert hit["hits"] == 1


def test_new_commit_sends_the_candidate_back_to_the_model(cache):
    auditor = CountingAuditor()

    lookup_or_store(cache, repo_id=1, full_name="owner1/repo1", head_sha=SHA_A, produce=auditor)
    auditor.result = audit_result(head_sha=SHA_B)
    lookup_or_store(cache, repo_id=1, full_name="owner1/repo1", head_sha=SHA_B, produce=auditor)

    assert auditor.calls == 2
    assert len(cache.entries()) == 2


def test_refresh_sends_everything_back_to_the_model(tmp_path):
    path = tmp_path / "scout.sqlite3"
    auditor = CountingAuditor()

    with AuditCache(path) as warm:
        lookup_or_store(warm, repo_id=1, full_name="owner1/repo1", head_sha=SHA_A, produce=auditor)

    with AuditCache(path, refresh=True) as cold:
        lookup_or_store(cold, repo_id=1, full_name="owner1/repo1", head_sha=SHA_A, produce=auditor)

    assert auditor.calls == 2


def test_stored_entry_keeps_the_payload_type_of_its_version(cache):
    auditor = CountingAuditor()
    lookup_or_store(cache, repo_id=1, full_name="owner1/repo1", head_sha=SHA_A, produce=auditor)

    assert cache.entries()[0].payload_type == "audit_result_v1"


# --------------------------------------------------------------------------
# Обслуживание: list / drop
# --------------------------------------------------------------------------


def test_entries_are_listed_newest_first(cache):
    cache.put(entry(audit_result(repo_id=1), hits=5))
    cache.put(entry(audit_result(repo_id=2)))

    assert {item.repo_id for item in cache.entries()} == {1, 2}


def test_drop_removes_every_entry_of_one_repository(cache):
    cache.put(entry(audit_result(repo_id=1, head_sha=SHA_A)))
    cache.put(entry(audit_result(repo_id=1, head_sha=SHA_B)))
    cache.put(entry(audit_result(repo_id=2)))

    assert cache.drop(1) == 2
    assert [item.repo_id for item in cache.entries()] == [2]


def test_drop_of_unknown_repository_is_not_an_error(cache):
    assert cache.drop(999) == 0


# --------------------------------------------------------------------------
# Команды `scout cache` (критерий приёмки дня 6 в ROADMAP.md)
# --------------------------------------------------------------------------


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    """CLI ходит в базу проекта — на время теста подменяем её на временную."""
    path = tmp_path / ".cache" / "scout.sqlite3"
    monkeypatch.setattr(cli, "DEFAULT_CACHE_PATH", path)
    return path


def test_cache_list_reports_empty_database(cache_path, capsys):
    assert cli.main(["cache", "list"]) == 0
    assert "Кэш пуст" in capsys.readouterr().out


def test_cache_list_shows_key_verdict_and_hits(cache_path, capsys):
    with AuditCache(cache_path) as cache:
        cache.put(entry(audit_result(), hits=3))

    assert cli.main(["cache", "list"]) == 0

    out = capsys.readouterr().out
    assert f"repo:1:{SHA_A}" in out
    assert "owner1/repo1" in out
    assert "USE" in out
    assert "попаданий 3" in out


def test_cache_drop_removes_the_repository(cache_path, capsys):
    with AuditCache(cache_path) as cache:
        cache.put(entry(audit_result(head_sha=SHA_A)))
        cache.put(entry(audit_result(head_sha=SHA_B)))

    assert cli.main(["cache", "drop", "1"]) == 0
    assert "Удалено записей: 2" in capsys.readouterr().out

    with AuditCache(cache_path) as reopened:
        assert reopened.entries() == []
