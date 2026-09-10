"""Детектор дословных заимствований в `AuditResult`.

`ARCHITECTURE.md` → «Разделение контекстов»: аудит читает чужие файлы и обязан
отдавать наружу только структурированные факты — имена, пути, числа, категории
и собственную прозу. Ни одна строка исходного кода не покидает контекст аудита.

Правило проверяется здесь, а не декларируется: перед тем как `AuditResult`
станет объектом, все его текстовые поля прогоняются против материала, который
модель видела. Совпадение длиннее 120 символов — отказ.

**Почему 120.** Это не порог «похожести», а граница между цитатой и переносом.
Имя функции, путь и короткая формулировка из README в сто двадцать символов
не укладываются; абзац документации или тело функции — укладываются легко.
Число взято из `ARCHITECTURE.md` и повторяется в `CHECKLIST.md`; менять его
можно только в обоих местах сразу.

**Почему сравнение идёт по нормализованному тексту.** Модель переносит строки
и схлопывает отступы, и побайтовое сравнение пропускало бы ровно те случаи,
ради которых детектор заводится. Нормализация приводит обе стороны к одному
виду: пробелы схлопнуты, регистр опущен.
"""

import re

MIN_VERBATIM_CHARS = 120
"""`ARCHITECTURE.md` и `CHECKLIST.md`. Менять — только в обоих файлах сразу."""

_WHITESPACE = re.compile(r"\s+")


class RegurgitationDetected(RuntimeError):
    """Поле содержит дословный фрагмент исходника длиннее допустимого.

    Не ошибка выполнения: это отказ принять запись. Кандидат уйдёт на повтор
    с явным требованием пересказать своими словами, а не процитировать.
    """

    def __init__(self, field: str, excerpt: str) -> None:
        self.field = field
        self.excerpt = excerpt
        super().__init__(f"{field}: дословный фрагмент материала длиной {len(excerpt)} символов")


def normalize(text: str) -> str:
    """Схлопывает пробелы и опускает регистр — обе стороны сравнения к одному виду."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def find_verbatim(text: str, source: str, *, min_length: int = MIN_VERBATIM_CHARS) -> str | None:
    """Самый длинный дословный фрагмент `text`, встречающийся в `source`.

    Возвращает найденный фрагмент в нормализованном виде или `None`. Скользящее
    окно по `text`, а не поиск общих подстрок: нас интересует только то, что
    модель вынесла наружу, и текст поля всегда короче материала на порядки.
    """
    haystack = normalize(source)
    needle = normalize(text)
    if len(needle) < min_length or not haystack:
        return None

    for start in range(len(needle) - min_length + 1):
        window = needle[start : start + min_length]
        if window in haystack:
            return window
    return None


def _strings(value: object, path: str = "") -> list[tuple[str, str]]:
    """Все строковые листья структуры вместе с путём до них."""
    if isinstance(value, str):
        return [(path or "<root>", value)]
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for key, item in value.items():
            found += _strings(item, f"{path}.{key}" if path else str(key))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found += _strings(item, f"{path}[{index}]")
        return found
    return []


def assert_clean(payload: object, source: str, *, min_length: int = MIN_VERBATIM_CHARS) -> None:
    """Проверяет всю структуру ответа модели. Первое совпадение — отказ.

    Проверяются все строковые поля без исключений, включая `verdict_rationale`
    и заметки о рисках: закон не делает различия между «процитировал код»
    и «процитировал документацию», а мы обещали отдавать только пересказ.

    Пути и имена файлов при этом не ловятся сами по себе — они короче порога.
    """
    for field, text in _strings(payload):
        excerpt = find_verbatim(text, source, min_length=min_length)
        if excerpt is not None:
            raise RegurgitationDetected(field, excerpt)
