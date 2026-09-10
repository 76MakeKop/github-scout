"""Сборка `Report` из `AuditResult[]` (SCHEMAS.md §8). Детерминированный код.

Модель сюда не заглядывает вовсе: отчёт — это перекладка уже полученных суждений
в контракт, плюс два решения, которые обязаны быть воспроизводимыми, — кто попал
в пятёрку и какая рекомендация вышла у скана целиком.

Порядок кандидатов — по `(score.total DESC, repo_id ASC)`. Второй ключ не
украшение: при равном `total` два прогона обязаны дать одну и ту же пятёрку,
иначе сравнение «до и после» из `CHECKLIST.md` меряло бы порядок словаря.

**Рекомендация скана — это вердикт первого кандидата, а не отдельное суждение.**
Иначе их стало бы два: один в `candidates[0].verdict`, другой в
`recommendation`, — и они разошлись бы при первой же правке порогов. Пустой
отчёт даёт BUILD: подходящего решения не нашлось, и это содержательный ответ,
а не отказ (`ARCHITECTURE.md`, последняя строка таблицы).
"""

from datetime import UTC, datetime
from uuid import UUID

from scout.schemas import (
    AuditResult,
    CacheStats,
    DroppedCandidate,
    EffortDays,
    LicensePassport,
    Report,
    ReportCandidate,
    ReportMode,
    Verdict,
)

MAX_REPORT_CANDIDATES = 5
"""`CLAUDE.md`, запрет 5. Совпадает с `maxItems` в `SCHEMAS.md` §8."""

MAX_STRENGTHS = 5
MAX_WEAKNESSES = 5

BUILD_RATIONALE = (
    "Готового решения под задачу не нашлось: ни один кандидат не дошёл до аудита "
    "или все отсеяны по существу. Разумный следующий шаг — писать самим."
)


def _rank(audits: list[AuditResult]) -> list[AuditResult]:
    """Порядок пятёрки. Считает код: `total` уже посчитан кодом же."""
    return sorted(audits, key=lambda audit: (-audit.score.total, audit.repo_id))


def _strengths(audit: AuditResult) -> list[str]:
    """Чем кандидат хорош. Минимум одна строка — этого требует контракт.

    Источник — `fit.covers`: это ответ модели на вопрос «что из задачи он уже
    решает», то есть ровно та формулировка, которой место в отчёте. Когда
    покрытий не нашлось, отчёт не выдумывает достоинство, а честно говорит,
    что кандидат держится на оценке.
    """
    covers = [text for text in audit.fit.covers if text.strip()][:MAX_STRENGTHS]
    if covers:
        return covers
    return [
        f"Оценка соответствия задаче {audit.score.relevance:.2f} при общей {audit.score.total:.2f}"
    ]


def _weaknesses(audit: AuditResult) -> list[str]:
    """Пробелы и риски одним списком, пробелы первыми.

    Риск и пробел — разные вещи (`не умеет X` против `один мейнтейнер`), но для
    читателя отчёта это одна колонка «на что смотреть». Порядок закреплён:
    сначала то, чего не хватает для задачи, потом то, чем это грозит потом.
    """
    gaps = [text for text in audit.fit.gaps if text.strip()]
    risks = [f"{risk.type.value}: {risk.note}" for risk in audit.risks]
    return (gaps + risks)[:MAX_WEAKNESSES]


def _candidate(audit: AuditResult, rank: int, html_url: str) -> ReportCandidate:
    return ReportCandidate(
        rank=rank,
        full_name=audit.full_name,
        html_url=html_url,
        head_sha=audit.head_sha,
        verdict=audit.verdict,
        score=audit.score.total,
        strengths=_strengths(audit),
        weaknesses=_weaknesses(audit),
        license_passport=audit.license_passport,
        integration_effort_days=audit.fit.integration_effort_days,
        provenance=list(audit.provenance),
    )


def _rationale(top: AuditResult | None) -> str:
    if top is None:
        return BUILD_RATIONALE
    if top.verdict_rationale:
        return top.verdict_rationale
    return (
        f"{top.full_name}: оценка {top.score.total:.2f}, лицензия "
        f"{top.license_passport.spdx_id or 'не определена'}."
    )


def build_report(
    *,
    request_id: UUID,
    query_text: str,
    audits: list[AuditResult],
    urls: dict[str, str],
    mode: ReportMode = ReportMode.SYNC,
    cost_usd: float = 0.0,
    duration_sec: float = 0.0,
    audits_hit: int = 0,
    audits_miss: int = 0,
    partial: bool = False,
    dropped: list[DroppedCandidate] | None = None,
    queries_used: list[str] | None = None,
    limit: int = MAX_REPORT_CANDIDATES,
    generated_at: datetime | None = None,
) -> Report:
    """`AuditResult[]` → `Report`. Ни одного вызова модели.

    `urls` приходит снаружи: `html_url` живёт в `Candidate`, а не в `AuditResult`,
    и выдумывать его из `full_name` нельзя — GitHub переносит репозитории, и
    собранная строка указывала бы на редирект вместо канонического адреса.
    """
    ranked = _rank(audits)[:limit]
    top = ranked[0] if ranked else None

    return Report(
        request_id=request_id,
        query_text=query_text,
        generated_at=generated_at or datetime.now(UTC),
        mode=mode,
        recommendation=top.verdict if top else Verdict.BUILD,
        recommendation_target=top.full_name if top and top.verdict is not Verdict.BUILD else None,
        rationale=_rationale(top),
        candidates=[
            _candidate(
                audit, rank, urls.get(audit.full_name, f"https://github.com/{audit.full_name}")
            )
            for rank, audit in enumerate(ranked, start=1)
        ],
        partial=partial,
        dropped=list(dropped or []),
        queries_used=list(queries_used or []),
        cost_usd=cost_usd,
        duration_sec=duration_sec,
        cache=CacheStats(audits_hit=audits_hit, audits_miss=audits_miss),
    )


def render_markdown(report: Report) -> str:
    """Markdown — производная от JSON, а не второй источник истины (`SCHEMAS.md` §8).

    Поэтому здесь нет ни одного числа, которого не было бы в объекте: функция
    только переставляет и подписывает.
    """
    lines = [
        f"# {report.query_text}",
        "",
        f"**Рекомендация: {report.recommendation.value}**"
        + (f" — `{report.recommendation_target}`" if report.recommendation_target else ""),
        "",
        report.rationale or "",
    ]

    if report.partial:
        lines += ["", "> ⚠ Результат неполный: часть данных потеряна из-за сбоев."]

    for candidate in report.candidates:
        passport: LicensePassport = candidate.license_passport
        effort: EffortDays = candidate.integration_effort_days
        reuse = "код брать можно" if passport.code_reuse_allowed else "код брать нельзя"
        lines += [
            "",
            f"## {candidate.rank}. {candidate.full_name} — {candidate.verdict.value}"
            f" ({candidate.score:.2f})",
            "",
            f"{candidate.html_url}",
            "",
            f"**Лицензия:** {passport.spdx_id or 'не определена'} "
            f"({passport.copyleft.value}, {reuse})",
            f"**Интеграция:** {effort.low:g}–{effort.high:g} дней, вероятно {effort.likely:g}",
            "",
            "**Сильные стороны:**",
        ]
        lines += [f"- {text}" for text in candidate.strengths]
        if candidate.weaknesses:
            lines += ["", "**Слабые места:**"]
            lines += [f"- {text}" for text in candidate.weaknesses]
        if passport.obligations:
            lines += ["", "**Обязательства лицензии:**"]
            lines += [f"- {text}" for text in passport.obligations]
        lines += ["", "**Откуда сведения:**"]
        lines += [
            f"- `{item.path}` @ `{item.commit_sha[:7]}` ({item.api.value})"
            for item in candidate.provenance
        ]

    if report.dropped:
        lines += ["", "## Выбыли", ""]
        lines += [
            f"- `{item.full_name}` ({item.stage.value}): {item.reason}" for item in report.dropped
        ]

    lines += [
        "",
        "---",
        "",
        f"Стоимость {report.cost_usd:.4f} $ · {report.duration_sec:.0f} с · "
        f"кэш {report.cache.audits_hit}/{report.cache.audits_hit + report.cache.audits_miss}",
    ]
    return "\n".join(lines)
