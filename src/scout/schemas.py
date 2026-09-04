"""Pydantic-модели контрактов между слоями.

Источник истины — SCHEMAS.md, §1–9. Ничего сверх описанного там здесь нет.
Скоринг (`total`) и правила вердикта USE/FORK/BUILD намеренно не реализованы:
по ROADMAP.md это день 13, и считает их код конвейера, а не схема.
"""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)

# --------------------------------------------------------------------------
# Базовые типы
# --------------------------------------------------------------------------


def _require_utc(value: datetime) -> datetime:
    """SCHEMAS.md: все метки времени — ISO-8601 UTC с суффиксом Z."""
    if value.utcoffset() != timedelta(0):
        raise ValueError("метка времени должна быть в UTC (суффикс Z)")
    return value


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_require_utc)]

CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{7,40}$")]
FullName = Annotated[str, StringConstraints(pattern=r"^[^/]+/[^/]+$")]
QueryId = Annotated[str, StringConstraints(pattern=r"^q\d+$")]

Text200 = Annotated[str, StringConstraints(max_length=200)]
Text300 = Annotated[str, StringConstraints(max_length=300)]
Text500 = Annotated[str, StringConstraints(max_length=500)]
Text600 = Annotated[str, StringConstraints(max_length=600)]
Text1000 = Annotated[str, StringConstraints(max_length=1000)]

Unit = Annotated[float, Field(ge=0, le=1)]
NonNegInt = Annotated[int, Field(ge=0)]
NonNegFloat = Annotated[float, Field(ge=0)]


class SchemaModel(BaseModel):
    """`additionalProperties: false` для всех контрактов."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Перечисления
# --------------------------------------------------------------------------


class ModelName(StrEnum):
    FLASH = "deepseek-v4-flash"
    PRO = "deepseek-v4-pro"
    QWEN = "qwen3.8-max"


class PricingWindow(StrEnum):
    OFF_PEAK = "off-peak"
    PEAK = "peak"


class ProvenanceApi(StrEnum):
    SEARCH = "search"
    REPOS = "repos"
    CONTENTS = "contents"
    TREES = "trees"
    LICENSES = "licenses"


class QueryFamily(StrEnum):
    EXACT = "exact"
    SYNONYM = "synonym"
    TOPIC = "topic"
    LIBRARY = "library"
    BROAD = "broad"
    README = "readme"
    RECENT = "recent"


class SortOrder(StrEnum):
    STARS = "stars"
    UPDATED = "updated"
    BEST_MATCH = "best-match"


class ScreeningVerdict(StrEnum):
    PASS = "pass"
    REJECT = "reject"


class RedFlag(StrEnum):
    DEMO_OR_TUTORIAL = "demo-or-tutorial"
    WRONG_DOMAIN = "wrong-domain"
    WRONG_LANGUAGE = "wrong-language"
    SAAS_WRAPPER = "saas-wrapper"
    ABANDONED = "abandoned"
    NO_CODE = "no-code"
    DUPLICATE_OF_KNOWN = "duplicate-of-known"


class LicenseDetectedBy(StrEnum):
    GITHUB_API = "github-api"
    LICENSE_FILE = "license-file"
    README_MENTION = "readme-mention"
    NONE = "none"


class Copyleft(StrEnum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"
    UNKNOWN = "unknown"


class RiskType(StrEnum):
    LICENSE = "license"
    MAINTENANCE = "maintenance"
    DEPENDENCY = "dependency"
    API_STABILITY = "api-stability"
    SCOPE_MISMATCH = "scope-mismatch"
    SINGLE_MAINTAINER = "single-maintainer"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Verdict(StrEnum):
    USE = "USE"
    FORK = "FORK"
    BUILD = "BUILD"


class ReportMode(StrEnum):
    SYNC = "sync"
    OFF_PEAK = "off-peak"


class DropStage(StrEnum):
    SEARCH = "search"
    SCREENING = "screening"
    AUDIT = "audit"


# --------------------------------------------------------------------------
# Общие определения (SCHEMAS.md → «Общие определения»)
# --------------------------------------------------------------------------


class Provenance(SchemaModel):
    path: str = Field(description="путь в репозитории или 'metadata'")
    commit_sha: CommitSha
    retrieved_at: UtcDatetime
    api: ProvenanceApi


class TokenUsage(SchemaModel):
    model: ModelName
    input_tokens: NonNegInt
    cached_input_tokens: NonNegInt = 0
    output_tokens: NonNegInt
    cost_usd: NonNegFloat
    pricing_window: PricingWindow


class EffortDays(SchemaModel):
    """Трёхточечная оценка трудозатрат на интеграцию."""

    low: NonNegFloat
    likely: NonNegFloat
    high: NonNegFloat


# --------------------------------------------------------------------------
# §1. ScanRequest
# --------------------------------------------------------------------------


class ScanOptions(SchemaModel):
    max_candidates: Annotated[int, Field(ge=5, le=50)] = 50
    audit_limit: Annotated[int, Field(ge=1, le=10)] = 10
    report_limit: Annotated[int, Field(ge=1, le=5)] = 5
    off_peak: bool = False
    refresh: bool = False


class ScanRequest(SchemaModel):
    request_id: UUID
    query_text: Annotated[str, StringConstraints(min_length=8, max_length=500)]
    created_at: UtcDatetime
    options: ScanOptions


# --------------------------------------------------------------------------
# §2. Intent
# --------------------------------------------------------------------------


class Intent(SchemaModel):
    request_id: UUID
    task: Text200 = Field(description="переформулировка задачи одним предложением")
    domain: Annotated[list[str], Field(max_length=5)]
    languages: Annotated[list[str], Field(max_length=3)] = Field(
        description="пусто = язык не важен"
    )
    must_have: Annotated[list[str], Field(max_length=6)]
    nice_to_have: Annotated[list[str], Field(max_length=6)] = Field(default_factory=list)
    exclude: Annotated[list[str], Field(max_length=6)]
    synonyms: Annotated[list[str], Field(min_length=3, max_length=10)] = Field(
        description="англоязычные формулировки задачи"
    )
    known_libraries: Annotated[list[str], Field(max_length=8)] = Field(
        description="гипотезы об именах известных библиотек"
    )
    model: Literal[ModelName.FLASH]
    prompt_version: Annotated[str, StringConstraints(pattern=r"^intent-\d+$")]


# --------------------------------------------------------------------------
# §3. SearchQuerySet
# --------------------------------------------------------------------------


class SearchQuery(SchemaModel):
    id: QueryId
    family: QueryFamily
    q: Annotated[str, StringConstraints(max_length=256)] = Field(
        description="строка запроса GitHub Search с квалификаторами"
    )
    sort: SortOrder
    per_page: Annotated[int, Field(ge=10, le=30)]


class SearchQuerySet(SchemaModel):
    request_id: UUID
    generated_at: UtcDatetime
    generator_version: Annotated[str, StringConstraints(pattern=r"^qg-\d+$")]
    queries: Annotated[list[SearchQuery], Field(min_length=5, max_length=10)]


# --------------------------------------------------------------------------
# §4. Candidate
# --------------------------------------------------------------------------


class Candidate(SchemaModel):
    repo_id: int
    full_name: FullName
    html_url: HttpUrl
    description: Text500 | None = None
    language: str | None = None
    topics: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    stars: NonNegInt
    forks: NonNegInt | None = None
    open_issues: NonNegInt | None = None
    archived: bool
    is_fork: bool
    created_at: UtcDatetime | None = None
    pushed_at: UtcDatetime
    default_branch: str
    head_sha: CommitSha
    license_spdx: str | None = Field(
        default=None, description="из GitHub API; уточняется на Слое 2"
    )
    found_by: Annotated[list[QueryId], Field(min_length=1)]
    rrf_score: NonNegFloat
    prior_score: Unit | None = None
    rank: Annotated[int, Field(ge=1)]
    retrieved_at: UtcDatetime


# --------------------------------------------------------------------------
# §5. ScreeningResult
# --------------------------------------------------------------------------


class ScreeningItem(SchemaModel):
    repo_id: int
    full_name: str
    relevance: Unit
    verdict: ScreeningVerdict
    reasons: Annotated[list[Text200], Field(min_length=1, max_length=3)]
    red_flags: list[RedFlag] = Field(default_factory=list)
    evidence: list[Provenance] = Field(default_factory=list)


class ScreeningResult(SchemaModel):
    request_id: UUID
    layer: Literal[1]
    model: Literal[ModelName.FLASH]
    prompt_version: Annotated[str, StringConstraints(pattern=r"^l1-\d+$")]
    results: list[ScreeningItem]
    passed: Annotated[list[int], Field(max_length=10)] = Field(
        description="repo_id, отсортированные по relevance убыв."
    )
    token_usage: TokenUsage


# --------------------------------------------------------------------------
# §6. LicensePassport
# --------------------------------------------------------------------------


class LicensePassport(SchemaModel):
    spdx_id: str | None = Field(description="null = лицензия не определена")
    name: str | None = None
    detected_by: LicenseDetectedBy
    confidence: Unit
    copyleft: Copyleft
    network_copyleft: bool = Field(default=False, description="true для AGPL")
    commercial_use: bool | None = None
    attribution_required: bool | None = None
    share_alike: bool | None = None
    code_reuse_allowed: bool = Field(
        description="false при copyleft strong/unknown — тогда только реинжиниринг"
    )
    obligations: list[Text200]
    source: Provenance

    @model_validator(mode="after")
    def _unknown_license_forbids_reuse(self) -> "LicensePassport":
        """SCHEMAS.md §6: при неопределённой лицензии переиспользование запрещено."""
        if (self.spdx_id is None or self.copyleft is Copyleft.UNKNOWN) and self.code_reuse_allowed:
            raise ValueError(
                "code_reuse_allowed должен быть false при spdx_id=null или copyleft=unknown"
            )
        return self


# --------------------------------------------------------------------------
# §7. AuditResult
# --------------------------------------------------------------------------


class RepoStructure(SchemaModel):
    entrypoints: Annotated[list[str], Field(max_length=10)]
    modules: Annotated[list[str], Field(max_length=30)]
    has_tests: bool
    test_paths: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    has_ci: bool
    ci_files: Annotated[list[str], Field(max_length=10)] = Field(default_factory=list)
    has_docs: bool
    file_count: NonNegInt | None = None


class Dependency(SchemaModel):
    name: str
    constraint: str | None = None


class Dependencies(SchemaModel):
    manifest: str | None
    runtime: Annotated[list[Dependency], Field(max_length=40)]
    count: NonNegInt
    heavy: Annotated[list[str], Field(max_length=10)] = Field(
        default_factory=list,
        description="зависимости с тяжёлой установкой или нативными сборками",
    )


class Maintenance(SchemaModel):
    last_commit: UtcDatetime
    commits_90d: NonNegInt | None = None
    contributors_12m: NonNegInt | None = None
    open_issues: NonNegInt
    releases_12m: NonNegInt | None = None


class Fit(SchemaModel):
    covers: list[Text200]
    gaps: list[Text200]
    integration_effort_days: EffortDays


class Risk(SchemaModel):
    type: RiskType
    severity: Severity
    note: Text300


class Score(SchemaModel):
    relevance: Unit
    quality: Unit
    maintenance: Unit
    license: Unit
    total: Unit


class AuditResult(SchemaModel):
    repo_id: int
    full_name: str
    head_sha: CommitSha
    audited_at: UtcDatetime
    model: Literal[ModelName.PRO, ModelName.QWEN]
    prompt_version: Annotated[str, StringConstraints(pattern=r"^l2-\d+$")]
    structure: RepoStructure
    dependencies: Dependencies
    license_passport: LicensePassport
    maintenance: Maintenance
    fit: Fit
    risks: list[Risk]
    score: Score
    verdict: Verdict
    verdict_rationale: Text600 | None = None
    provenance: Annotated[list[Provenance], Field(min_length=1)]
    token_usage: TokenUsage | None = None


# --------------------------------------------------------------------------
# §8. Report
# --------------------------------------------------------------------------


class ReportCandidate(SchemaModel):
    rank: Annotated[int, Field(ge=1, le=5)]
    full_name: str
    html_url: HttpUrl
    head_sha: str | None = None
    verdict: Verdict
    score: Unit
    strengths: Annotated[list[Text200], Field(min_length=1, max_length=5)]
    weaknesses: Annotated[list[Text200], Field(max_length=5)]
    license_passport: LicensePassport
    integration_effort_days: EffortDays
    provenance: list[Provenance]


class DroppedCandidate(SchemaModel):
    full_name: str
    stage: DropStage
    reason: Text200


class CacheStats(SchemaModel):
    audits_hit: NonNegInt
    audits_miss: NonNegInt


class Report(SchemaModel):
    request_id: UUID
    query_text: str
    generated_at: UtcDatetime
    mode: ReportMode
    recommendation: Verdict
    recommendation_target: str | None = Field(
        default=None, description="full_name или null для BUILD"
    )
    rationale: Text1000 | None = None
    candidates: Annotated[list[ReportCandidate], Field(max_length=5)]
    partial: bool = Field(
        default=False,
        description="часть данных потеряна из-за сбоя: сеть, лимит, неразобранный ответ",
    )
    dropped: list[DroppedCandidate] = Field(default_factory=list)
    queries_used: list[str] = Field(
        default_factory=list,
        description="строки запросов — для отладки и воспроизводимости",
    )
    cost_usd: NonNegFloat
    duration_sec: NonNegFloat
    cache: CacheStats


# --------------------------------------------------------------------------
# §9. CacheEntry
# --------------------------------------------------------------------------


class CacheEntry(SchemaModel):
    key: Annotated[str, StringConstraints(pattern=r"^repo:\d+:[0-9a-f]{7,40}$")]
    repo_id: int
    head_sha: CommitSha
    payload_type: Literal["audit_result_v1"]
    payload: AuditResult
    created_at: UtcDatetime
    hits: NonNegInt = 0
