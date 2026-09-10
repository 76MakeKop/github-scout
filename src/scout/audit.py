"""Слой 2 — аудит: `Candidate` + материал репозитория → `AuditResult`.

Разделение обязанностей здесь жёсткое и обосновано `SCHEMAS.md` §7:

- **модель** судит о том, что видно только человеческим чтением: структура,
  зависимости, лицензионный паспорт, покрытие задачи, риски и четыре частные
  оценки `relevance` / `quality` / `maintenance` / `license`;
- **код** считает `total` по формуле и выносит вердикт по порогам, а также
  проставляет всё, что является фактом, а не суждением: `repo_id`, `full_name`,
  `head_sha`, `audited_at`, `model`, `prompt_version`, `maintenance` (данные
  GitHub), `provenance`, `token_usage`.

Так сделано не ради чистоты слоёв. Вердикт, произнесённый моделью, невоспроизводим
и неоткатываем: два прогона на одном и том же репозитории дадут разные USE/FORK,
и никакой замер recall уже не скажет, что изменилось — данные или настроение
модели. Формула из четырёх чисел даёт одинаковый ответ на одинаковых числах,
а её пороги можно двигать осознанно и с замером до/после (`CHECKLIST.md`).

`code_reuse_allowed` тоже считает код: `SCHEMAS.md` §6 требует `false` при
`copyleft: strong`/`unknown` или `spdx_id: null`, и это правило слишком дорого,
чтобы доверять его модели — на нём висит юридическая часть отчёта.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from scout import config, repo_reader
from scout.cache import AuditCache, lookup_or_store
from scout.deepseek import DeepSeekAuth, DeepSeekBadResponse, DeepSeekClient
from scout.github import GitHubAuth, GitHubClient
from scout.log import RunLogger
from scout.regurgitation import RegurgitationDetected, assert_clean
from scout.schemas import (
    AuditResult,
    Candidate,
    Copyleft,
    Intent,
    LicensePassport,
    Maintenance,
    ModelName,
    Score,
    Verdict,
)

MAX_ATTEMPTS = 2  # первая попытка + один повтор с текстом ошибки

SCORE_WEIGHTS = {"relevance": 0.40, "quality": 0.25, "maintenance": 0.20, "license": 0.15}
"""`SCHEMAS.md` §7. Менять только с замером recall до/после (`CLAUDE.md`, запрет 7)."""

USE_THRESHOLD = 0.70
FORK_THRESHOLD = 0.50
"""Пороги вердикта. С весами RRF 0,70/0,30 из `QUERIES.md` не связаны никак —
совпадение чисел случайное, и `SCHEMAS.md` §7 отдельно предупреждает, что
«унификация» сломает обе формулы."""

# Эти поля модель не заполняет: они либо факты из GitHub, либо решение кода.
_CODE_OWNED_FIELDS = (
    "repo_id",
    "full_name",
    "head_sha",
    "audited_at",
    "model",
    "prompt_version",
    "maintenance",
    "verdict",
    "provenance",
    "token_usage",
)


class AuditUnparsed(RuntimeError):
    """Модель дважды вернула аудит не по схеме. Кандидат остаётся без аудита."""


def load_system_prompt() -> str:
    return config.prompt_path("l2").read_text(encoding="utf-8")


def _format_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]


def total_score(relevance: float, quality: float, maintenance: float, license_: float) -> float:
    """`total = 0.40*relevance + 0.25*quality + 0.20*maintenance + 0.15*license`."""
    return (
        SCORE_WEIGHTS["relevance"] * relevance
        + SCORE_WEIGHTS["quality"] * quality
        + SCORE_WEIGHTS["maintenance"] * maintenance
        + SCORE_WEIGHTS["license"] * license_
    )


def code_reuse_allowed(passport_fields: dict[str, Any]) -> bool:
    """`SCHEMAS.md` §6: strong-copyleft, unknown и отсутствие лицензии — только
    реинжиниринг. Нет лицензии — нет разрешения; молчание правами не наделяет."""
    spdx = passport_fields.get("spdx_id")
    copyleft = passport_fields.get("copyleft")
    if not spdx:
        return False
    return copyleft not in (Copyleft.STRONG.value, Copyleft.UNKNOWN.value, None)


def decide_verdict(total: float, *, gaps: Sequence[Any], passport: LicensePassport) -> Verdict:
    """Правила `SCHEMAS.md` §7, применяются кодом после получения оценок.

    Проверка лицензии стоит первой и до порогов: `copyleft: strong` и
    `spdx_id: null` дают BUILD независимо от того, насколько репозиторий хорош.
    Иначе отличный AGPL-проект получил бы USE и попал в отчёт как «можно брать»,
    что и есть та ошибка, ради предотвращения которой паспорт заводился.
    """
    if not passport.code_reuse_allowed:
        return Verdict.BUILD

    if total >= USE_THRESHOLD and not _blocking(gaps):
        return Verdict.USE

    if total >= FORK_THRESHOLD or passport.copyleft is Copyleft.WEAK:
        return Verdict.FORK

    return Verdict.BUILD


def _blocking(gaps: Sequence[Any]) -> list[Any]:
    """Пробелы, из-за которых задача остаётся нерешённой.

    Пробел без признака считается блокирующим: это поведение до дня 13, когда
    `gaps` был списком строк и `USE` требовал полного их отсутствия. Умолчание
    выбрано строгим намеренно — ошибка в сторону FORK стоит лишней осторожности,
    ошибка в сторону USE стоит рекомендации взять то, что не решает задачу.
    """
    blocking = []
    for gap in gaps:
        flag = (
            gap.get("blocking", True) if isinstance(gap, dict) else getattr(gap, "blocking", True)
        )
        if flag:
            blocking.append(gap)
    return blocking


def _maintenance(candidate: Candidate) -> Maintenance:
    """Поддерживаемость — факты GitHub, а не суждение модели.

    `commits_90d`, `contributors_12m` и `releases_12m` остаются пустыми: каждое
    стоит отдельного запроса на репозиторий, а контракт разрешает `null`. Ставить
    туда догадку модели нельзя — это число, которое потом читают как факт.
    """
    return Maintenance(
        last_commit=candidate.pushed_at,
        open_issues=candidate.open_issues or 0,
    )


def _user_message(intent: Intent, candidate: Candidate, material: repo_reader.AuditMaterial) -> str:
    task = "; ".join(intent.synonyms[:3]) if intent.synonyms else intent.task
    lines = [
        "ЗАДАЧА ПОЛЬЗОВАТЕЛЯ:",
        intent.task,
        f"КЛЮЧЕВЫЕ ФОРМУЛИРОВКИ: {task}",
    ]
    if intent.languages:
        lines.append(f"ОЖИДАЕМЫЕ ЯЗЫКИ: {', '.join(intent.languages)}")
    if candidate.language:
        lines.append(f"ОСНОВНОЙ ЯЗЫК РЕПОЗИТОРИЯ: {candidate.language}")
    if candidate.license_spdx:
        lines.append(f"ЛИЦЕНЗИЯ ПО ДАННЫМ GITHUB API: {candidate.license_spdx}")
    lines.append("")
    lines.append(material.as_prompt_block())
    return "\n".join(lines)


def _assemble(
    payload: dict[str, Any],
    candidate: Candidate,
    material: repo_reader.AuditMaterial,
    *,
    audited_at: datetime,
) -> AuditResult:
    """Собирает `AuditResult` из ответа модели и того, что принадлежит коду.

    Поля из `_CODE_OWNED_FIELDS` выбрасываются из ответа, что бы модель ни
    прислала: она может выдумать `repo_id` или объявить вердикт, и молча
    принять это значило бы потерять и воспроизводимость, и связь с кандидатом.
    """
    fields = {key: value for key, value in payload.items() if key not in _CODE_OWNED_FIELDS}

    # Детектор регургитации стоит до сборки объекта, а не после: отклонённая
    # запись не должна существовать даже в памяти как валидный `AuditResult`.
    assert_clean(fields, material.as_prompt_block())

    passport_fields = dict(fields.get("license_passport") or {})
    passport_fields.pop("source", None)
    passport_fields.pop("code_reuse_allowed", None)
    passport_fields["code_reuse_allowed"] = code_reuse_allowed(passport_fields)
    passport_fields["source"] = material.provenance[0].model_dump()
    passport = LicensePassport(**passport_fields)

    scores = dict(fields.get("score") or {})
    scores.pop("total", None)
    score = Score(
        **scores,
        total=total_score(
            scores.get("relevance", 0.0),
            scores.get("quality", 0.0),
            scores.get("maintenance", 0.0),
            scores.get("license", 0.0),
        ),
    )

    fit = fields.get("fit") or {}

    return AuditResult(
        **{key: value for key, value in fields.items() if key not in ("license_passport", "score")},
        license_passport=passport,
        score=score,
        repo_id=candidate.repo_id,
        full_name=candidate.full_name,
        head_sha=candidate.head_sha,
        audited_at=audited_at,
        model=ModelName.PRO,
        prompt_version=config.PROMPT_VERSIONS["l2"],
        maintenance=_maintenance(candidate),
        verdict=decide_verdict(score.total, gaps=fit.get("gaps") or [], passport=passport),
        provenance=material.provenance,
    )


def audit_one(
    candidate: Candidate,
    *,
    intent: Intent,
    system: str,
    client: DeepSeekClient,
    github: GitHubClient,
    logger: RunLogger | None = None,
    totals: dict[str, int] | None = None,
) -> AuditResult:
    """Аудит одного кандидата: материал, вызов V4-Pro, один повтор при невалидном.

    Повтор здесь тот же, что на Слое 1, и по той же причине: «пустая строка
    вместо JSON» и «JSON не по схеме» для кандидата — одно и то же, ответа нет.
    Разница в цене: на Слое 2 повтор стоит в тридцать раз дороже, поэтому
    попытки по-прежнему две, а не три.
    """
    counters = totals if totals is not None else {}
    material = repo_reader.collect(candidate, github=github, logger=logger)
    user = _user_message(intent, candidate, material)
    audited_at = datetime.now(UTC)

    if logger:
        logger.info(
            "audit_started",
            full_name=candidate.full_name,
            tree_paths=len(material.tree_paths),
            manifests=list(material.manifests),
            has_readme=bool(material.readme),
            license_file=material.license_path,
            estimated_tokens=material.estimated_tokens,
        )

    problems: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            payload, usage = client.chat_json(
                system=system,
                user=user,
                model=ModelName.PRO.value,
                temperature=0.0,
            )
        except DeepSeekBadResponse as exc:
            problems = [str(exc)]
            if logger:
                logger.info(
                    "audit_bad_response",
                    full_name=candidate.full_name,
                    attempt=attempt,
                    detail=str(exc),
                )
            if attempt == MAX_ATTEMPTS:
                break
            continue

        for key, value in usage.items():
            counters[key] = counters.get(key, 0) + value

        try:
            result = _assemble(payload, candidate, material, audited_at=audited_at)
        except RegurgitationDetected as exc:
            # Отказ принять запись, а не сбой: `ARCHITECTURE.md` → «Разделение
            # контекстов». Повтор с явным требованием пересказать своими словами.
            problems = [str(exc)]
            if logger:
                logger.error(
                    "audit_regurgitation",
                    full_name=candidate.full_name,
                    attempt=attempt,
                    field=exc.field,
                    chars=len(exc.excerpt),
                )
            if attempt == MAX_ATTEMPTS:
                break
            user = (
                f"{user}\n\n"
                f"Поле {exc.field} дословно повторяет материал репозитория "
                f"({len(exc.excerpt)} символов). Перескажи своими словами: "
                "поля содержат факты и твою прозу, а не цитаты. Верни исправленный JSON."
            )
            continue
        except ValidationError as exc:
            problems = _format_errors(exc)
            if logger:
                logger.info(
                    "audit_invalid",
                    full_name=candidate.full_name,
                    attempt=attempt,
                    problems=problems,
                )
            if attempt == MAX_ATTEMPTS:
                break
            user = (
                f"{user}\n\n"
                "Предыдущий ответ не прошёл валидацию схемы:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\nВерни исправленный JSON."
            )
            continue

        if logger:
            logger.info(
                "audit_done",
                full_name=candidate.full_name,
                verdict=result.verdict.value,
                total=round(result.score.total, 4),
                spdx=result.license_passport.spdx_id,
                copyleft=result.license_passport.copyleft.value,
                gaps=len(result.fit.gaps),
            )
        return result

    if logger:
        logger.error("audit_failed", full_name=candidate.full_name, problems=problems)
    raise AuditUnparsed(f"{candidate.full_name}: {'; '.join(problems) or 'ответ не разобран'}")


def audit_candidates(
    candidates: Sequence[Candidate],
    intent: Intent,
    *,
    client: DeepSeekClient,
    github: GitHubClient,
    cache: AuditCache | None = None,
    logger: RunLogger | None = None,
) -> tuple[list[AuditResult], list[str], dict[str, int]]:
    """Аудит топ-N последовательно. Возвращает результаты, имена провалившихся
    и счётчики токенов.

    Последовательно, а не пулом: Слой 1 распараллелен, потому что там полсотни
    дешёвых вызовов, здесь их пять и каждый дорогой. Пул дал бы выигрыш в минуту
    и отнял бы возможность остановиться после первого же отказа ключа.

    Провал одного кандидата не роняет остальных — цена дня 8. Кандидат без
    аудита просто не попадёт в отчёт.
    """
    totals: dict[str, int] = {}
    results: list[AuditResult] = []
    failed: list[str] = []
    system = load_system_prompt()

    for candidate in candidates:
        produce = lambda c=candidate: audit_one(  # noqa: E731 — замыкание на кандидата
            c,
            intent=intent,
            system=system,
            client=client,
            github=github,
            logger=logger,
            totals=totals,
        )

        try:
            if cache is None:
                results.append(produce())
            else:
                results.append(
                    lookup_or_store(
                        cache,
                        repo_id=candidate.repo_id,
                        full_name=candidate.full_name,
                        head_sha=candidate.head_sha,
                        prompt_version=config.PROMPT_VERSIONS["l2"],
                        produce=produce,
                        logger=logger,
                    )
                )
        except (DeepSeekAuth, GitHubAuth):
            # Одинаково для всех кандидатов: продолжать значит заплатить
            # за пять одинаковых отказов.
            raise
        except Exception as exc:
            failed.append(candidate.full_name)
            if logger:
                logger.error(
                    "audit_crashed",
                    full_name=candidate.full_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    return results, failed, totals
