"""Снимок выдачи: вход слоёв, зафиксированный на диске, и прогон по нему.

Замер «до и после» из `CHECKLIST.md` требует, чтобы между плечами A/B менялся
ровно один слой. 2026-09-11 выяснилось, что само собой это не выполняется:
на замере `blocking` состав выдачи совпал побайтово у одной задачи из двенадцати,
а весь прирост recall@5 пришёл из ловушки, где поиск в один прогон вернул ноль
кандидатов, а в другой — одиннадцать. Разброс выше измеряемого слоя оказался
больше измеряемого эффекта, и правило «ухудшение recall@5 — основание откатить»
превратилось в подбрасывание монеты.

Снимок чинит это на входе. Прогон записывает то, что добыл — интент, строки
запросов, `Candidate[]` и результат Слоя 1, — а следующий подаёт записанное
вместо того, чтобы добывать заново. Оба плеча A/B получают один и тот же вход,
и разойтись они могут только тем слоем, который правят.

Усреднение по прогонам вместо фиксации отвергнуто по цене: прогон 12 задач со
Слоем 2 стоит около доллара, и десяток повторов ради устойчивого среднего стоит
дороже, чем весь остальной проект вместе взятый.

**Замораживать можно по-разному, и уровень выбирает тот, кто правит слой:**

- `ReplayStage.SEARCH` — заморожены интент, запросы и `Candidate[]`; Слой 1 идёт
  живьём. Этим меряется правка промпта Слоя 1 (`l1-2`).
- `ReplayStage.SCREENING` — заморожен ещё и результат Слоя 1; живьём идёт только
  Слой 2. Этим меряется правка промпта Слоя 2 (`l2-3`).

Замораживать «до Слоя 1» при правке Слоя 2 недостаточно, и это не осторожность
сверх меры: на том же замере у `k8s-secrets-in-git` Слой 1 пропустил восьмерых
против троих на одних и тех же кандидатах. Слой между заморозкой и правкой
приносит свой разброс, и он ничем не лучше разброса поиска.

Формат — JSONL, по задаче на строку, как у журнала `evaluate.py` и по той же
причине: строка доводится до диска сразу, как задача отработала, и переживает
`kill -9` посреди полуторачасового прогона.
"""

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError

from scout.schemas import (
    Candidate,
    DroppedCandidate,
    DropStage,
    Intent,
    SchemaModel,
    ScreeningResult,
    UtcDatetime,
)

if TYPE_CHECKING:  # только для аннотации: `pipeline` импортирует этот модуль
    from scout.pipeline import ScanOutcome

SNAPSHOT_VERSION = 1
"""Версия формата строки. Читатель обязан отличить свой снимок от чужого:
несовместимую строку лучше пропустить, чем разобрать наполовину."""


class ReplayStage(StrEnum):
    """Докуда прогон берёт записанное вместо того, чтобы добывать заново."""

    SEARCH = "search"
    SCREENING = "screening"


class SnapshotIncomplete(RuntimeError):
    """В снимке нет того, что просят воспроизвести."""


class FrozenCase(SchemaModel):
    """Одна задача снимка: всё, что прогон добыл до Слоя 2.

    `dropped` хранит только потери поиска: потери скрининга и аудита принадлежат
    тем слоям, и при воспроизведении они либо пересчитываются из `screening_failed`,
    либо случаются заново — записывать их значило бы удваивать.
    """

    snapshot_version: Literal[1] = SNAPSHOT_VERSION
    slug: str
    query_text: str
    recorded_at: UtcDatetime
    intent: Intent
    queries_used: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    dropped: list[DroppedCandidate] = Field(default_factory=list)
    screening: ScreeningResult | None = None
    screening_failed: list[str] = Field(default_factory=list)

    def has(self, stage: ReplayStage) -> bool:
        """Хватает ли записанного, чтобы воспроизвести до этого этапа."""
        if stage is ReplayStage.SCREENING:
            return self.screening is not None
        return True


def freeze(slug: str, outcome: "ScanOutcome", *, moment: datetime | None = None) -> FrozenCase:
    """`ScanOutcome` → строка снимка. Знание о том, что заморожено, живёт здесь."""
    screening = outcome.screening
    return FrozenCase(
        slug=slug,
        query_text=outcome.request.query_text,
        recorded_at=moment or datetime.now(UTC),
        intent=outcome.intent,
        queries_used=list(outcome.queries_used),
        candidates=list(outcome.candidates),
        dropped=[item for item in outcome.dropped if item.stage is DropStage.SEARCH],
        screening=screening.result if screening else None,
        screening_failed=list(screening.failed) if screening else [],
    )


def write_case(path: Path, case: FrozenCase) -> None:
    """Дописывает задачу и доводит её до диска — как `append_journal`.

    `fsync` по той же причине: без него строка остаётся в буфере ОС, а прогон,
    который её добыл, уже оплачен и повторно бесплатным не будет.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read(path: Path) -> dict[str, FrozenCase]:
    """Снимок с диска, по slug. Битая строка пропускается молча.

    Пропускается, а не роняет чтение: снимок существует ровно на случай, когда
    процесс убили посреди записи, и терять из-за последней строки все остальные
    значило бы отменять то, ради чего файл заводится. Порядок ключей — порядок
    записи: питоновский словарь его хранит, и повторный прогон идёт задачами
    в том же порядке, что исходный.
    """
    if not path.exists():
        raise SnapshotIncomplete(f"снимок не найден: {path}")

    cases: dict[str, FrozenCase] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            case = FrozenCase(**json.loads(line))
        except (json.JSONDecodeError, TypeError, ValidationError):
            continue
        cases[case.slug] = case

    if not cases:
        raise SnapshotIncomplete(f"в снимке {path} нет ни одной разобранной задачи")
    return cases


def missing(cases: dict[str, FrozenCase], slugs: list[str], stage: ReplayStage) -> list[str]:
    """Задачи, которых в снимке нет или которые записаны не до нужного этапа.

    Проверка идёт до первого платного вызова: прогон, который сорвётся на
    середине из-за отсутствующей задачи, уже потратит деньги на предыдущие.
    """
    return [slug for slug in slugs if slug not in cases or not cases[slug].has(stage)]
