"""По одному валидному и одному невалидному объекту на каждый тип из SCHEMAS.md §1–9."""

import pytest
from pydantic import ValidationError

from scout.schemas import (
    AuditResult,
    CacheEntry,
    Candidate,
    Intent,
    LicensePassport,
    Report,
    ScanRequest,
    ScreeningResult,
    SearchQuerySet,
)

UUID_OK = "0f6d1a2e-8c4b-4f1a-9d3e-1b2c3d4e5f60"
SHA = "a1b2c3d"
TS = "2026-09-03T12:00:00Z"

PROVENANCE = {"path": "README.md", "commit_sha": SHA, "retrieved_at": TS, "api": "contents"}

TOKEN_USAGE = {
    "model": "deepseek-v4-flash",
    "input_tokens": 75_000,
    "output_tokens": 5_000,
    "cost_usd": 0.0198,
    "pricing_window": "off-peak",
}

LICENSE_MIT = {
    "spdx_id": "MIT",
    "name": "MIT License",
    "detected_by": "license-file",
    "confidence": 0.99,
    "copyleft": "none",
    "network_copyleft": False,
    "commercial_use": True,
    "attribution_required": True,
    "share_alike": False,
    "code_reuse_allowed": True,
    "obligations": ["сохранить текст лицензии и copyright"],
    "source": PROVENANCE,
}

AUDIT_OK = {
    "repo_id": 12345,
    "full_name": "camelot-dev/camelot",
    "head_sha": SHA,
    "audited_at": TS,
    "model": "deepseek-v4-pro",
    "prompt_version": "l2-1",
    "structure": {
        "entrypoints": ["camelot/__init__.py"],
        "modules": ["camelot.io", "camelot.parsers"],
        "has_tests": True,
        "test_paths": ["tests/"],
        "has_ci": True,
        "ci_files": [".github/workflows/ci.yml"],
        "has_docs": True,
        "file_count": 180,
    },
    "dependencies": {
        "manifest": "pyproject.toml",
        "runtime": [{"name": "pdfminer.six", "constraint": ">=20221105"}],
        "count": 12,
        "heavy": ["opencv-python"],
    },
    "license_passport": LICENSE_MIT,
    "maintenance": {
        "last_commit": TS,
        "commits_90d": 34,
        "contributors_12m": 7,
        "open_issues": 120,
    },
    "fit": {
        "covers": ["извлечение таблиц из текстовых PDF"],
        "gaps": ["нет OCR для сканов"],
        "integration_effort_days": {"low": 2, "likely": 5, "high": 10},
    },
    "risks": [{"type": "dependency", "severity": "medium", "note": "тянет opencv-python"}],
    "score": {
        "relevance": 0.9,
        "quality": 0.8,
        "maintenance": 0.7,
        "license": 1.0,
        "total": 0.81,
    },
    "verdict": "USE",
    "provenance": [PROVENANCE],
}


# --- §1. ScanRequest ------------------------------------------------------


def test_scan_request_valid():
    request = ScanRequest(
        request_id=UUID_OK,
        query_text="нужен парсер PDF-таблиц на Python",
        created_at=TS,
        options={},
    )
    assert request.options.max_candidates == 50
    assert request.options.off_peak is False


def test_scan_request_invalid_short_query():
    with pytest.raises(ValidationError):
        ScanRequest(request_id=UUID_OK, query_text="pdf", created_at=TS, options={})


# --- §2. Intent -----------------------------------------------------------


INTENT_OK = {
    "request_id": UUID_OK,
    "task": "извлечение таблиц из PDF в структурированный вид",
    "domain": ["pdf", "data-extraction"],
    "languages": ["python"],
    "must_have": ["сохранение строк и столбцов"],
    "exclude": ["платные SaaS-обёртки"],
    "synonyms": ["pdf table extraction", "extract tables from pdf", "pdf table parser"],
    "known_libraries": ["camelot", "tabula-py"],
    "model": "deepseek-v4-flash",
    "prompt_version": "intent-1",
}


def test_intent_valid():
    intent = Intent(**INTENT_OK)
    assert intent.prompt_version == "intent-1"
    assert intent.nice_to_have == []


def test_intent_invalid_too_few_synonyms():
    with pytest.raises(ValidationError):
        Intent(**{**INTENT_OK, "synonyms": ["pdf table extraction"]})


# --- §3. SearchQuerySet ---------------------------------------------------


def _queries(count: int = 5) -> list[dict]:
    return [
        {
            "id": f"q{i}",
            "family": "exact",
            "q": "pdf table extraction language:python archived:false",
            "sort": "best-match",
            "per_page": 30,
        }
        for i in range(1, count + 1)
    ]


def test_search_query_set_valid():
    qs = SearchQuerySet(
        request_id=UUID_OK, generated_at=TS, generator_version="qg-1", queries=_queries(7)
    )
    assert len(qs.queries) == 7


def test_search_query_set_invalid_too_few_queries():
    with pytest.raises(ValidationError):
        SearchQuerySet(
            request_id=UUID_OK, generated_at=TS, generator_version="qg-1", queries=_queries(4)
        )


# --- §4. Candidate --------------------------------------------------------


CANDIDATE_OK = {
    "repo_id": 12345,
    "full_name": "camelot-dev/camelot",
    "html_url": "https://github.com/camelot-dev/camelot",
    "stars": 3400,
    "archived": False,
    "is_fork": False,
    "pushed_at": TS,
    "default_branch": "main",
    "head_sha": SHA,
    "found_by": ["q1", "q3"],
    "rrf_score": 0.031,
    "rank": 1,
    "retrieved_at": TS,
}


def test_candidate_valid():
    candidate = Candidate(**CANDIDATE_OK)
    assert candidate.topics == []
    assert candidate.license_spdx is None


def test_candidate_invalid_full_name_without_owner():
    with pytest.raises(ValidationError):
        Candidate(**{**CANDIDATE_OK, "full_name": "camelot"})


# --- §5. ScreeningResult --------------------------------------------------


SCREENING_OK = {
    "request_id": UUID_OK,
    "layer": 1,
    "model": "deepseek-v4-flash",
    "prompt_version": "l1-1",
    "results": [
        {
            "repo_id": 12345,
            "full_name": "camelot-dev/camelot",
            "relevance": 0.92,
            "verdict": "pass",
            "reasons": ["решает ровно задачу извлечения таблиц"],
            "red_flags": [],
            "evidence": [PROVENANCE],
        }
    ],
    "passed": [12345],
    "token_usage": TOKEN_USAGE,
}


def test_screening_result_valid():
    result = ScreeningResult(**SCREENING_OK)
    assert result.layer == 1
    assert result.results[0].verdict == "pass"


def test_screening_result_invalid_layer():
    with pytest.raises(ValidationError):
        ScreeningResult(**{**SCREENING_OK, "layer": 2})


# --- §6. LicensePassport --------------------------------------------------


def test_license_passport_valid():
    passport = LicensePassport(**LICENSE_MIT)
    assert passport.code_reuse_allowed is True


def test_license_passport_invalid_reuse_without_license():
    """SCHEMAS.md §6: spdx_id=null → code_reuse_allowed обязан быть false."""
    with pytest.raises(ValidationError):
        LicensePassport(
            **{
                **LICENSE_MIT,
                "spdx_id": None,
                "copyleft": "unknown",
                "detected_by": "none",
                "code_reuse_allowed": True,
            }
        )


# --- §7. AuditResult ------------------------------------------------------


def test_audit_result_valid():
    audit = AuditResult(**AUDIT_OK)
    assert audit.verdict == "USE"
    assert audit.dependencies.count == 12


def test_audit_result_invalid_model_for_layer2():
    """Слой 2 не работает на Flash — только Pro или Qwen-fallback."""
    with pytest.raises(ValidationError):
        AuditResult(**{**AUDIT_OK, "model": "deepseek-v4-flash"})


# --- §8. Report -----------------------------------------------------------


REPORT_OK = {
    "request_id": UUID_OK,
    "query_text": "нужен парсер PDF-таблиц на Python",
    "generated_at": TS,
    "mode": "sync",
    "recommendation": "USE",
    "recommendation_target": "camelot-dev/camelot",
    "candidates": [
        {
            "rank": 1,
            "full_name": "camelot-dev/camelot",
            "html_url": "https://github.com/camelot-dev/camelot",
            "verdict": "USE",
            "score": 0.81,
            "strengths": ["два режима разбора"],
            "weaknesses": ["тянет opencv-python"],
            "license_passport": LICENSE_MIT,
            "integration_effort_days": {"low": 2, "likely": 5, "high": 10},
            "provenance": [PROVENANCE],
        }
    ],
    "cost_usd": 0.13,
    "duration_sec": 210,
    "cache": {"audits_hit": 6, "audits_miss": 4},
}


def test_report_valid():
    report = Report(**REPORT_OK)
    assert report.candidates[0].rank == 1
    assert report.dropped == []


def test_report_invalid_rank_out_of_range():
    broken = {**REPORT_OK}
    broken["candidates"] = [{**REPORT_OK["candidates"][0], "rank": 6}]
    with pytest.raises(ValidationError):
        Report(**broken)


# --- §9. CacheEntry -------------------------------------------------------


CACHE_OK = {
    "key": f"repo:12345:{SHA}",
    "repo_id": 12345,
    "head_sha": SHA,
    "payload_type": "audit_result_v1",
    "payload": AUDIT_OK,
    "created_at": TS,
}


def test_cache_entry_valid():
    entry = CacheEntry(**CACHE_OK)
    assert entry.hits == 0
    assert entry.payload.full_name == "camelot-dev/camelot"


def test_cache_entry_invalid_key_format():
    with pytest.raises(ValidationError):
        CacheEntry(**{**CACHE_OK, "key": "12345"})


# --- общие инварианты -----------------------------------------------------


def test_extra_fields_are_forbidden():
    """additionalProperties: false во всех контрактах."""
    with pytest.raises(ValidationError):
        Candidate(**{**CANDIDATE_OK, "unexpected": 1})


def test_naive_timestamp_is_rejected():
    """SCHEMAS.md: метки времени — ISO-8601 UTC с суффиксом Z."""
    with pytest.raises(ValidationError):
        Candidate(**{**CANDIDATE_OK, "pushed_at": "2026-09-03T12:00:00"})
