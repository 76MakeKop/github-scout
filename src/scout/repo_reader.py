"""Сбор материала для Слоя 2: дерево, манифесты, README, лицензия.

`ARCHITECTURE.md` → «Что каждый слой видит»: аудит получает дерево файлов
(до 300 путей), файлы-манифесты, README и файл лицензии. **Исходный код модулей
не запрашивается** — ни здесь, ни где-либо ещё в конвейере. Это не экономия
контекста, а «Разделение контекстов»: чем меньше чужого кода прошло через
модель, тем меньше поверхность для дословной регургитации.

Всё читается по `head_sha` кандидата, а не по подвижной ветке. Иначе скрининг,
аудит и запись в кэш смотрели бы на разные ревизии под одним ключом
`repo:{repo_id}:{head_sha}`.

Бюджет контекста — 12k токенов на репозиторий (`ARCHITECTURE.md` → токеномика).
Он считается по оценке «символов на токен», а не точным токенайзером: своего
токенайзера у нас нет, тянуть зависимость ради оценки, которая всё равно
проверяется постфактум по `token_usage`, незачем. Оценка сознательно
пессимистична — лучше отрезать лишнее, чем внезапно заплатить за перерасход.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from scout.github import GitHubAuth, GitHubClient, GitHubError
from scout.log import RunLogger
from scout.schemas import Candidate, Provenance, ProvenanceApi

MAX_TREE_PATHS = 300
"""`ARCHITECTURE.md`. Столько же режет `GitHubClient.get_tree` — здесь предел
повторён затем, чтобы урезание бюджета не молчало о том, сколько путей осталось."""

MAX_README_CHARS = 7000
"""~2000 токенов. Слой 1 видит первые 4000 символов, аудиту нужен полный текст:
установка, ограничения и «что этот проект не делает» живут ниже середины."""

MAX_MANIFEST_CHARS = 4000
MAX_MANIFESTS = 3
MAX_LICENSE_CHARS = 3000
"""Шапки хватает: SPDX приходит из API, а тело GPL — это 35k символов текста,
который модель и так знает наизусть."""

AUDIT_TOKEN_BUDGET = 12_000
CHARS_PER_TOKEN = 3.5
"""Пессимистичнее обычных ~4: в манифестах и деревьях путей много пунктуации,
а она дробит токены."""

MANIFEST_NAMES = (
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "Gemfile",
    "mix.exs",
    "pubspec.yaml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "Package.swift",
)
"""Порядок значим: он же порядок предпочтения, когда манифестов несколько."""

README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")
LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "LICENCE",
    "LICENCE.md",
    "COPYING",
    "COPYING.txt",
)

# Порядок урезания при перерасходе бюджета: от наименее полезного к самому
# ценному. README режется последним — из всего материала он единственный
# объясняет замысел проекта словами автора.
_TRIM_ORDER = ("license", "tree", "manifests", "readme")


@dataclass
class AuditMaterial:
    """Что Слой 2 увидит про один репозиторий.

    `provenance` собирается по ходу чтения: `SCHEMAS.md` §7 требует минимум одну
    запись, и она должна указывать на реально прочитанный путь, а не на общий
    «репозиторий». Без этого в отчёте нечем подтвердить утверждение об аудите.
    """

    full_name: str
    head_sha: str
    tree_paths: list[str] = field(default_factory=list)
    manifests: dict[str, str] = field(default_factory=dict)
    readme: str = ""
    license_path: str | None = None
    license_text: str = ""
    provenance: list[Provenance] = field(default_factory=list)
    trimmed: list[str] = field(default_factory=list)
    tree_available: bool = True

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.as_prompt_block())

    def as_prompt_block(self) -> str:
        """Материал одним текстом — ровно в том виде, в каком его увидит модель."""
        parts = [f"РЕПОЗИТОРИЙ: {self.full_name}", f"РЕВИЗИЯ: {self.head_sha}"]

        if self.tree_paths:
            parts.append(
                f"\nДЕРЕВО ФАЙЛОВ ({len(self.tree_paths)} путей):\n" + "\n".join(self.tree_paths)
            )
        elif not self.tree_available:
            parts.append("\nДЕРЕВО ФАЙЛОВ: недоступно (репозиторий не отдал дерево).")

        for path, text in self.manifests.items():
            parts.append(f"\nМАНИФЕСТ {path}:\n{text}")

        if self.readme:
            parts.append(f"\nREADME:\n{self.readme}")
        else:
            parts.append("\nREADME: отсутствует.")

        if self.license_text:
            parts.append(f"\nФАЙЛ ЛИЦЕНЗИИ {self.license_path} (начало):\n{self.license_text}")
        else:
            parts.append("\nФАЙЛ ЛИЦЕНЗИИ: не найден в дереве.")

        return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Оценка сверху. Точное число приходит потом в `token_usage` от API."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _provenance(path: str, head_sha: str, api: ProvenanceApi, when: datetime) -> Provenance:
    return Provenance(path=path, commit_sha=head_sha, retrieved_at=when, api=api)


def _read(
    github: GitHubClient,
    full_name: str,
    path: str,
    ref: str,
    *,
    logger: RunLogger | None,
) -> str | None:
    """Один файл. Сетевая осечка на файле не должна ронять аудит репозитория.

    Отсутствующий манифест или лицензия — это факт о репозитории, а не сбой:
    именно его модель и должна увидеть. Отклонённый ключ — наоборот, одинаков
    для всех, и глушить его значило бы выдать вердикт по пустому материалу.
    """
    try:
        return github.get_file(full_name, path, ref=ref)
    except GitHubAuth:
        raise
    except GitHubError as exc:
        if logger:
            logger.info("audit_file_failed", full_name=full_name, path=path, detail=str(exc))
        return None


def _pick(tree_paths: list[str], names: tuple[str, ...]) -> list[str]:
    """Файлы из корня репозитория, в порядке предпочтения `names`.

    Только корень: `LICENSE` внутри `vendor/` или `node_modules/` описывает
    чужую зависимость, а не проект, и приняв его за лицензию проекта мы
    выдали бы паспорт на посторонний код.
    """
    root = {path for path in tree_paths if "/" not in path}
    return [name for name in names if name in root]


def collect(
    candidate: Candidate,
    *,
    github: GitHubClient,
    logger: RunLogger | None = None,
    budget_tokens: int = AUDIT_TOKEN_BUDGET,
) -> AuditMaterial:
    """Материал для аудита одного кандидата, уложенный в бюджет контекста."""
    retrieved_at = datetime.now(UTC)
    material = AuditMaterial(full_name=candidate.full_name, head_sha=candidate.head_sha)

    try:
        tree: list[dict[str, Any]] = github.get_tree(
            candidate.full_name, candidate.head_sha, limit=None
        )
    except GitHubAuth:
        raise
    except GitHubError as exc:
        tree = []
        material.tree_available = False
        if logger:
            logger.info("audit_tree_failed", full_name=candidate.full_name, detail=str(exc))

    if not tree:
        material.tree_available = False

    all_paths = [node["path"] for node in tree if node.get("type") == "blob" and node.get("path")]
    material.tree_paths = all_paths[:MAX_TREE_PATHS]

    # Искать README, лицензию и манифесты нужно по **всему** дереву, а не по
    # обрезанным тремстам путям. Обрезка — это бюджет контекста, а не утверждение
    # о том, что в репозитории больше ничего нет: в проекте с тысячей файлов
    # `LICENSE` легко оказывается за границей среза, и аудит выдал бы «лицензия
    # не найдена» там, где она лежит в корне.
    if material.tree_paths:
        material.provenance.append(
            _provenance("git/trees", candidate.head_sha, ProvenanceApi.TREES, retrieved_at)
        )

    for name in _pick(all_paths, MANIFEST_NAMES)[:MAX_MANIFESTS]:
        text = _read(github, candidate.full_name, name, candidate.head_sha, logger=logger)
        if text:
            material.manifests[name] = text[:MAX_MANIFEST_CHARS]
            material.provenance.append(
                _provenance(name, candidate.head_sha, ProvenanceApi.CONTENTS, retrieved_at)
            )

    for name in _pick(all_paths, README_NAMES) or [README_NAMES[0]]:
        text = _read(github, candidate.full_name, name, candidate.head_sha, logger=logger)
        if text:
            material.readme = text[:MAX_README_CHARS]
            material.provenance.append(
                _provenance(name, candidate.head_sha, ProvenanceApi.CONTENTS, retrieved_at)
            )
            break

    for name in _pick(all_paths, LICENSE_NAMES):
        text = _read(github, candidate.full_name, name, candidate.head_sha, logger=logger)
        if text:
            material.license_path = name
            material.license_text = text[:MAX_LICENSE_CHARS]
            material.provenance.append(
                _provenance(name, candidate.head_sha, ProvenanceApi.CONTENTS, retrieved_at)
            )
            break

    if not material.provenance:
        # Контракт требует минимум одну запись, а сказать про репозиторий всё
        # равно есть что: метаданные пришли вместе с кандидатом.
        material.provenance.append(
            _provenance("metadata", candidate.head_sha, ProvenanceApi.REPOS, retrieved_at)
        )

    _fit_budget(material, budget_tokens, logger=logger)
    return material


def _fit_budget(material: AuditMaterial, budget: int, *, logger: RunLogger | None) -> None:
    """Ужимает материал до бюджета, начиная с наименее полезного.

    Режем по частям в фиксированном порядке, а не пропорционально: пропорция
    отняла бы у README столько же, сколько у текста лицензии, хотя README —
    единственное место, где замысел проекта описан словами.
    """
    for part in _TRIM_ORDER:
        if material.estimated_tokens <= budget:
            return

        if part == "license" and material.license_text:
            material.license_text = material.license_text[:600]
            material.trimmed.append("license")
        elif part == "tree" and len(material.tree_paths) > 120:
            material.tree_paths = material.tree_paths[:120]
            material.trimmed.append("tree")
        elif part == "manifests" and material.manifests:
            material.manifests = {path: text[:1500] for path, text in material.manifests.items()}
            material.trimmed.append("manifests")
        elif part == "readme" and material.readme:
            material.readme = material.readme[:3500]
            material.trimmed.append("readme")

    if logger and material.trimmed:
        logger.info(
            "audit_material_trimmed",
            full_name=material.full_name,
            trimmed=material.trimmed,
            estimated_tokens=material.estimated_tokens,
            budget=budget,
        )
