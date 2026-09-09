"""Сбор материала для Слоя 2: что читается, чего не читается, как режется бюджет.

Сети нет: `GitHubClient` подменяется фейком с заданным деревом и файлами.
Проверяется не содержание аудита, а то, какой материал до модели доедет —
именно от него зависит и цена вызова, и то, о чём модель вообще может судить.
"""

import pytest

from scout import repo_reader
from scout.github import GitHubError
from scout.repo_reader import AUDIT_TOKEN_BUDGET, collect, estimate_tokens
from test_screen import candidate


class FakeGitHub:
    """Дерево и файлы заданы наперёд. Считает обращения: лишний запрос — это
    и лишняя трата `core`-лимита, и признак того, что слой читает не то."""

    def __init__(self, tree=(), files=None, tree_error=False):
        self.tree = list(tree)
        self.files = dict(files or {})
        self.tree_error = tree_error
        self.requested: list[str] = []

    def get_tree(self, full_name, sha, *, limit=repo_reader.MAX_TREE_PATHS):
        if self.tree_error:
            raise GitHubError("дерево недоступно")
        self.tree_limit = limit
        nodes = [{"path": path, "type": "blob"} for path in self.tree]
        return nodes if limit is None else nodes[:limit]

    def get_file(self, full_name, path, ref=None):
        self.requested.append(path)
        return self.files.get(path)


def blob_tree(*paths):
    return list(paths)


def test_tree_manifest_readme_and_licence_are_collected():
    github = FakeGitHub(
        tree=blob_tree("README.md", "LICENSE", "pyproject.toml", "src/scout/cli.py"),
        files={
            "README.md": "как пользоваться",
            "LICENSE": "MIT License",
            "pyproject.toml": "[project]\nname='scout'",
        },
    )

    material = collect(candidate(1), github=github)

    assert material.readme == "как пользоваться"
    assert material.license_path == "LICENSE"
    assert material.manifests == {"pyproject.toml": "[project]\nname='scout'"}
    assert "src/scout/cli.py" in material.tree_paths


def test_module_sources_are_never_requested():
    """`ARCHITECTURE.md` → «Разделение контекстов»: исходный код модулей Слой 2
    не запрашивает. Это не экономия, а юридическая гигиена."""
    github = FakeGitHub(
        tree=blob_tree("README.md", "src/app.py", "lib/core.rs", "main.go"),
        files={"README.md": "текст"},
    )

    collect(candidate(1), github=github)

    assert github.requested == ["README.md"]


def test_everything_is_read_at_the_candidate_revision():
    """По `head_sha`, а не по ветке: иначе скрининг, аудит и ключ кэша
    `repo:{repo_id}:{head_sha}` указывали бы на разные ревизии."""
    seen = []

    class Recording(FakeGitHub):
        def get_file(self, full_name, path, ref=None):
            seen.append(ref)
            return super().get_file(full_name, path, ref=ref)

    subject = candidate(1)
    github = Recording(tree=blob_tree("README.md"), files={"README.md": "текст"})

    collect(subject, github=github)

    assert seen == [subject.head_sha]


def test_licence_inside_vendor_is_not_taken_for_the_project_licence():
    """`vendor/LICENSE` описывает чужую зависимость. Приняв его за лицензию
    проекта, мы выдали бы паспорт на посторонний код."""
    github = FakeGitHub(
        tree=blob_tree("vendor/LICENSE", "node_modules/x/LICENSE", "README.md"),
        files={"vendor/LICENSE": "GPL-3.0", "README.md": "текст"},
    )

    material = collect(candidate(1), github=github)

    assert material.license_path is None
    assert material.license_text == ""


def test_missing_licence_and_manifest_are_facts_not_failures():
    """Их отсутствие — это то, что модель обязана увидеть, а не сбой сбора."""
    github = FakeGitHub(tree=blob_tree("README.md"), files={"README.md": "текст"})

    material = collect(candidate(1), github=github)

    assert material.license_path is None
    assert material.manifests == {}
    assert "ФАЙЛ ЛИЦЕНЗИИ: не найден" in material.as_prompt_block()


def test_unreadable_tree_does_not_sink_the_audit():
    """Репозиторий может не отдать дерево. Метаданные кандидата при этом остались,
    и судить по ним всё ещё можно — а `provenance` контракт требует непустым."""
    github = FakeGitHub(tree_error=True)

    material = collect(candidate(1), github=github)

    assert material.tree_available is False
    assert material.tree_paths == []
    assert len(material.provenance) == 1
    assert material.provenance[0].path == "metadata"


def test_manifests_are_capped_so_one_repo_cannot_eat_the_budget():
    github = FakeGitHub(
        tree=blob_tree("pyproject.toml", "package.json", "go.mod", "Cargo.toml", "pom.xml"),
        files={name: "x" for name in ("pyproject.toml", "package.json", "go.mod", "Cargo.toml")},
    )

    material = collect(candidate(1), github=github)

    assert len(material.manifests) <= repo_reader.MAX_MANIFESTS


def deep_paths(count, width):
    """Длинные вложенные пути. Именно они, а не размер файлов, переполняют бюджет
    на реальных монорепозиториях: триста путей по двести символов — это 60k."""
    return [
        f"packages/{'sub/' * (width // 40)}module_{i:04d}.py".ljust(width, "x")
        for i in range(count)
    ]


def test_manifest_is_found_when_a_directory_pushes_it_past_the_cap():
    """Живой прогон дня 11 на `ispras/dedoc`: каталог `dedoc/` сортируется раньше
    корневых `pyproject.toml` и `setup.py`, срез дерева в триста записей попадает
    внутрь каталога, и аудит получал «манифеста нет» при трёх манифестах в корне.
    Поиск файлов обязан идти по полному дереву, урезание — дело сборщика промпта."""
    github = FakeGitHub(
        tree=blob_tree(*[f"dedoc/module_{i:04d}.py" for i in range(400)], "pyproject.toml"),
        files={"pyproject.toml": "[project]"},
    )

    material = collect(candidate(1), github=github)

    assert github.tree_limit is None
    assert "pyproject.toml" in material.manifests


def test_conda_environment_counts_as_a_manifest():
    """`microsoft/table-transformer` объявляет зависимости только в `environment.yml`,
    и без него поле `dependencies` уходило к модели пустым."""
    github = FakeGitHub(
        tree=blob_tree("environment.yml", "README.md"),
        files={"environment.yml": "dependencies:\n  - torch", "README.md": "текст"},
    )

    material = collect(candidate(1), github=github)

    assert "environment.yml" in material.manifests


def test_licence_is_found_behind_the_300_path_cap():
    """Обрезка дерева — это бюджет контекста, а не утверждение, что дальше ничего
    нет. Пока README и лицензия искались по срезу, репозиторий с тысячей файлов
    получал «лицензия не найдена» при лежащем в корне `LICENSE`, и паспорт
    выходил пустым по причине, к самому репозиторию отношения не имеющей."""
    github = FakeGitHub(
        tree=blob_tree(*deep_paths(400, 40), "README.md", "LICENSE"),
        files={"README.md": "текст", "LICENSE": "MIT License"},
    )

    material = collect(candidate(1), github=github)

    assert material.license_path == "LICENSE"
    assert material.readme == "текст"
    assert len(material.tree_paths) <= repo_reader.MAX_TREE_PATHS


def test_material_is_trimmed_into_the_token_budget():
    """Бюджет 12k токенов — это цена вызова V4-Pro. Перерасход тут оплачивается
    деньгами, поэтому урезание обязано срабатывать, а не предупреждать."""
    github = FakeGitHub(
        tree=blob_tree(*deep_paths(300, 200), "README.md", "LICENSE"),
        files={
            "README.md": "очень длинный README. " * 2000,
            "LICENSE": "текст лицензии. " * 2000,
        },
    )

    material = collect(candidate(1), github=github)

    assert material.estimated_tokens <= AUDIT_TOKEN_BUDGET
    assert material.trimmed == ["license", "tree"]


def test_readme_is_trimmed_last():
    """Порядок урезания не случаен: README — единственное место, где замысел
    проекта описан словами автора, поэтому он режется после всего остального.

    Здесь перебор небольшой, и обрезки одной лицензии хватает — до README
    очередь не доходит вовсе.
    """
    github = FakeGitHub(
        tree=blob_tree(*deep_paths(300, 110), "README.md", "LICENSE"),
        files={"README.md": "смысл проекта. " * 600, "LICENSE": "текст лицензии. " * 600},
    )

    material = collect(candidate(1), github=github)

    assert material.trimmed == ["license"]
    assert len(material.readme) == repo_reader.MAX_README_CHARS


def test_small_repository_is_not_trimmed_at_all():
    github = FakeGitHub(tree=blob_tree("README.md"), files={"README.md": "коротко"})

    material = collect(candidate(1), github=github)

    assert material.trimmed == []


@pytest.mark.parametrize("text,expected_min", [("", 1), ("a" * 350, 100)])
def test_token_estimate_is_an_upper_bound(text, expected_min):
    assert estimate_tokens(text) >= expected_min
