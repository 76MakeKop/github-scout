"""Слой 2: что считает код, что судит модель, и где проходит граница.

V4-Pro не дёргается ни разу: `DeepSeekClient` подменяется фейком с заданным
ответом. Проверяется главное свойство слоя — вердикт и `total` считает код,
поэтому одинаковые оценки обязаны давать одинаковое решение.
"""

import pytest

from scout import audit
from scout.audit import (
    AuditUnparsed,
    audit_candidates,
    audit_one,
    code_reuse_allowed,
    decide_verdict,
    total_score,
)
from scout.cache import AuditCache
from scout.deepseek import DeepSeekBadResponse
from scout.log import RunLogger
from scout.schemas import Copyleft, LicensePassport, Provenance, ProvenanceApi, Verdict
from test_queries import make_intent
from test_repo_reader import FakeGitHub, blob_tree
from test_screen import candidate

MODEL_ANSWER = {
    "structure": {
        "entrypoints": ["src/scout/cli.py"],
        "modules": ["src/scout"],
        "has_tests": True,
        "test_paths": ["tests/"],
        "has_ci": True,
        "ci_files": [".github/workflows/ci.yml"],
        "has_docs": False,
        "file_count": 12,
    },
    "dependencies": {
        "manifest": "pyproject.toml",
        "runtime": [{"name": "pydantic", "constraint": ">=2"}],
        "count": 1,
        "heavy": [],
    },
    "license_passport": {
        "spdx_id": "MIT",
        "name": "MIT License",
        "detected_by": "license-file",
        "confidence": 0.95,
        "copyleft": "none",
        "network_copyleft": False,
        "commercial_use": True,
        "attribution_required": True,
        "share_alike": False,
        "obligations": ["сохранять уведомление об авторстве"],
    },
    "fit": {
        "covers": ["разбор таблиц из PDF"],
        "gaps": [],
        "integration_effort_days": {"low": 0.5, "likely": 1.0, "high": 2.0},
    },
    "risks": [{"type": "single-maintainer", "severity": "low", "note": "один автор"}],
    "score": {"relevance": 0.9, "quality": 0.8, "maintenance": 0.8, "license": 1.0},
    "verdict_rationale": "Зрелая библиотека под ровно эту задачу.",
}

USAGE = {"input_tokens": 9000, "output_tokens": 1200, "cached_input_tokens": 4000}


class FakeDeepSeek:
    """Отдаёт заготовленные ответы по одному на вызов. Считает вызовы:
    лишний вызов V4-Pro — это реальные деньги."""

    def __init__(self, *answers, raises=None):
        self.answers = list(answers)
        self.raises = raises
        self.calls = 0
        self.last_user = ""

    def chat_json(self, *, system, user, model=None, temperature=0.0):
        self.calls += 1
        self.last_user = user
        if self.raises is not None:
            raise self.raises
        answer = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer, dict(USAGE)


def github_with_material():
    return FakeGitHub(
        tree=blob_tree("README.md", "LICENSE", "pyproject.toml", "src/scout/cli.py"),
        files={
            "README.md": "разбор PDF-таблиц",
            "LICENSE": "MIT License",
            "pyproject.toml": "[project]",
        },
    )


def run_audit(answer=None, *, subject=None, client=None, logger=None):
    return audit_one(
        subject or candidate(1),
        intent=make_intent(),
        system="системный промпт",
        client=client or FakeDeepSeek(answer or MODEL_ANSWER),
        github=github_with_material(),
        logger=logger,
    )


def passport(spdx="MIT", copyleft=Copyleft.NONE, reuse=True):
    return LicensePassport(
        spdx_id=spdx,
        detected_by="license-file",
        confidence=0.9,
        copyleft=copyleft,
        code_reuse_allowed=reuse,
        obligations=[],
        source=Provenance(
            path="LICENSE",
            commit_sha="0" * 40,
            retrieved_at="2026-09-10T00:00:00Z",
            api=ProvenanceApi.CONTENTS,
        ),
    )


# --------------------------------------------------------------------------
# Формула и вердикт — территория кода
# --------------------------------------------------------------------------


def test_total_follows_the_weights_from_the_contract():
    """`SCHEMAS.md` §7: 0.40/0.25/0.20/0.15. Число посчитано вручную по формуле,
    а не снято с вывода кода."""
    assert total_score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert total_score(0.9, 0.8, 0.8, 1.0) == pytest.approx(0.87)
    assert total_score(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_model_cannot_declare_the_verdict():
    """Вердикт от модели невоспроизводим: два прогона на одном репозитории дали бы
    разные USE/FORK, и замер recall перестал бы что-либо значить."""
    lying = dict(MODEL_ANSWER) | {"verdict": "BUILD", "repo_id": 999}

    result = run_audit(lying)

    assert result.verdict is Verdict.USE
    assert result.repo_id == candidate(1).repo_id


def test_model_cannot_declare_the_total_either():
    inflated = dict(MODEL_ANSWER)
    inflated["score"] = dict(MODEL_ANSWER["score"]) | {"total": 1.0}

    result = run_audit(inflated)

    assert result.score.total == pytest.approx(0.87)


@pytest.mark.parametrize(
    "total,gaps,expected",
    [
        (0.85, [], Verdict.USE),
        (0.85, ["нет асинхронного API"], Verdict.FORK),
        (0.60, [], Verdict.FORK),
        (0.40, [], Verdict.BUILD),
        (0.70, [], Verdict.USE),
        (0.50, ["пробел"], Verdict.FORK),
    ],
)
def test_verdict_table_by_its_boundaries(total, gaps, expected):
    """Пороги 0.70 и 0.50 из `SCHEMAS.md` §7, проверенные ровно на границах."""
    assert decide_verdict(total, gaps=gaps, passport=passport()) is expected


def test_strong_copyleft_can_never_be_use():
    """Отличный AGPL-проект остаётся BUILD: `code_reuse_allowed` ложен, и порог
    его не перебивает. Ради этого правила паспорт и заводился."""
    agpl = passport(spdx="AGPL-3.0", copyleft=Copyleft.STRONG, reuse=False)

    assert decide_verdict(0.99, gaps=[], passport=agpl) is Verdict.BUILD


def test_unknown_licence_can_never_be_use():
    unknown = passport(spdx=None, copyleft=Copyleft.UNKNOWN, reuse=False)

    assert decide_verdict(0.99, gaps=[], passport=unknown) is Verdict.BUILD


def test_weak_copyleft_reaches_fork_below_the_threshold():
    """`SCHEMAS.md` §7: FORK при `copyleft: weak` — даже когда total ниже 0.50."""
    mpl = passport(spdx="MPL-2.0", copyleft=Copyleft.WEAK)

    assert decide_verdict(0.30, gaps=[], passport=mpl) is Verdict.FORK


@pytest.mark.parametrize(
    "fields,allowed",
    [
        ({"spdx_id": "MIT", "copyleft": "none"}, True),
        ({"spdx_id": "MPL-2.0", "copyleft": "weak"}, True),
        ({"spdx_id": "GPL-3.0", "copyleft": "strong"}, False),
        ({"spdx_id": None, "copyleft": "unknown"}, False),
        ({"spdx_id": "MIT", "copyleft": "unknown"}, False),
        ({"spdx_id": None, "copyleft": "none"}, False),
    ],
)
def test_code_reuse_is_decided_by_code_not_by_the_model(fields, allowed):
    """Молчание правами не наделяет: нет лицензии — нет разрешения."""
    assert code_reuse_allowed(fields) is allowed


def test_model_claim_about_reuse_is_overridden():
    """Модель может объявить `code_reuse_allowed: true` на GPL. На этом поле висит
    юридическая часть отчёта, поэтому его пересчитывает код."""
    answer = dict(MODEL_ANSWER)
    answer["license_passport"] = dict(MODEL_ANSWER["license_passport"]) | {
        "spdx_id": "GPL-3.0",
        "copyleft": "strong",
        "code_reuse_allowed": True,
    }

    result = run_audit(answer)

    assert result.license_passport.code_reuse_allowed is False
    assert result.verdict is Verdict.BUILD


# --------------------------------------------------------------------------
# Сборка результата
# --------------------------------------------------------------------------


def test_audit_result_carries_code_owned_facts():
    subject = candidate(1)

    result = run_audit(subject=subject)

    assert result.full_name == subject.full_name
    assert result.head_sha == subject.head_sha
    assert result.prompt_version == "l2-1"
    assert result.model.value == "deepseek-v4-pro"
    assert result.maintenance.last_commit == subject.pushed_at
    assert result.provenance


def test_intent_and_material_both_reach_the_model():
    client = FakeDeepSeek(MODEL_ANSWER)

    run_audit(client=client)

    assert "ЗАДАЧА ПОЛЬЗОВАТЕЛЯ" in client.last_user
    assert "ДЕРЕВО ФАЙЛОВ" in client.last_user
    assert "МАНИФЕСТ pyproject.toml" in client.last_user


def test_token_counters_are_collected():
    totals: dict[str, int] = {}

    audit_one(
        candidate(1),
        intent=make_intent(),
        system="s",
        client=FakeDeepSeek(MODEL_ANSWER),
        github=github_with_material(),
        totals=totals,
    )

    assert totals["input_tokens"] == USAGE["input_tokens"]


# --------------------------------------------------------------------------
# Живучесть
# --------------------------------------------------------------------------


def test_invalid_answer_is_retried_once_with_the_errors():
    """Тот же приём, что на Слое 1, но повтор здесь дороже в тридцать раз —
    поэтому попыток две, а не три."""
    broken = {"score": {"relevance": 2.0}}
    client = FakeDeepSeek(broken, MODEL_ANSWER)

    result = run_audit(client=client)

    assert client.calls == 2
    assert result.verdict is Verdict.USE
    assert "не прошёл валидацию схемы" in client.last_user


def test_twice_invalid_answer_raises_and_does_not_invent_a_verdict():
    """Без аудита кандидат просто не попадёт в отчёт. Выдумать ему вердикт
    было бы хуже, чем не выдать вовсе."""
    client = FakeDeepSeek({"score": {}}, {"score": {}})

    with pytest.raises(AuditUnparsed):
        run_audit(client=client)

    assert client.calls == 2


def test_unparseable_response_is_retried_like_an_invalid_schema():
    """Пустая строка вместо JSON и JSON не по схеме для кандидата — одно и то же:
    ответа нет. Находка пилота дня 10, перенесённая на Слой 2."""
    client = FakeDeepSeek(DeepSeekBadResponse("пустой ответ"), MODEL_ANSWER)

    result = run_audit(client=client)

    assert client.calls == 2
    assert result.verdict is Verdict.USE


def test_one_broken_candidate_does_not_sink_the_others():
    """Цена дня 8: сбой шага стоит шага. Кандидат без аудита выбывает один."""

    class Flaky(FakeDeepSeek):
        def chat_json(self, *, system, user, model=None, temperature=0.0):
            self.calls += 1
            if "repo2" in user:
                raise RuntimeError("этот кандидат сломался")
            return MODEL_ANSWER, dict(USAGE)

    results, failed, _totals = audit_candidates(
        [candidate(1), candidate(2), candidate(3)],
        make_intent(),
        client=Flaky(),
        github=github_with_material(),
        logger=None,
    )

    assert [r.full_name for r in results] == ["owner1/repo1", "owner3/repo3"]
    assert failed == ["owner2/repo2"]


# --------------------------------------------------------------------------
# Кэш (день 6) и --refresh (техдолг 5)
# --------------------------------------------------------------------------


def test_second_audit_of_the_same_revision_costs_nothing(tmp_path):
    """Ключ `repo:{repo_id}:{head_sha}`: та же ревизия — модель не зовём."""
    client = FakeDeepSeek(MODEL_ANSWER)

    with AuditCache(tmp_path / "cache.sqlite3") as cache:
        audit_candidates(
            [candidate(1)],
            make_intent(),
            client=client,
            github=github_with_material(),
            cache=cache,
        )
        audit_candidates(
            [candidate(1)],
            make_intent(),
            client=client,
            github=github_with_material(),
            cache=cache,
        )

    assert client.calls == 1


def test_refresh_makes_the_audit_pay_again(tmp_path):
    """Техдолг 5: флаг разбирался, доходил до CLI и не имел потребителя, пока
    Слоя 2 не существовало. Теперь `--refresh` обходит кэш."""
    client = FakeDeepSeek(MODEL_ANSWER)
    path = tmp_path / "cache.sqlite3"

    with AuditCache(path) as cache:
        audit_candidates(
            [candidate(1)], make_intent(), client=client, github=github_with_material(), cache=cache
        )

    with AuditCache(path, refresh=True) as cache:
        audit_candidates(
            [candidate(1)], make_intent(), client=client, github=github_with_material(), cache=cache
        )

    assert client.calls == 2


def test_audit_logs_the_verdict_and_the_licence():
    """Без события в логе разбор дорогого слоя пришлось бы вести по stdout."""

    class Recorder(RunLogger):
        def __init__(self):
            super().__init__("audittest00")
            self.events = []

        def info(self, event, **fields):
            self.events.append((event, fields))

        def error(self, event, **fields):
            self.events.append((event, fields))

    log = Recorder()
    run_audit(logger=log)

    done = next(fields for name, fields in log.events if name == "audit_done")
    assert done["verdict"] == "USE"
    assert done["spdx"] == "MIT"


def test_prompt_file_exists_where_the_convention_says():
    """`CLAUDE.md` → «Где живут промпты»: `l2-1` это `prompts/l2/v1.md`."""
    assert audit.load_system_prompt().strip()
