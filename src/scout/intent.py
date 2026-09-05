"""Извлечение интента: NL-задача → структурированный `Intent` (Слой V4-Flash).

Один логический вызов на скан. Если модель вернула объект, не проходящий схему,
делается ровно один повтор с текстом ошибки — затем извлечение считается
провалившимся.

Статус живёт в обёртке `IntentExtraction`, а не внутри `Intent`: у контракта
`Intent` в SCHEMAS.md §2 поля `status` нет, а модель объявлена с extra="forbid".
"""

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from pydantic import ValidationError

from scout import config
from scout.deepseek import DeepSeekBadResponse, DeepSeekClient
from scout.log import RunLogger
from scout.schemas import Intent, ModelName

MAX_ATTEMPTS = 2  # первая попытка + один повтор с текстом ошибки

# Эти три поля модель не заполняет — их проставляет код.
_CODE_OWNED_FIELDS = ("request_id", "model", "prompt_version")


@dataclass
class IntentExtraction:
    """Результат извлечения. `intent` заполнен только при status="ok"."""

    status: Literal["ok", "failed"]
    intent: Intent | None = None
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


def load_system_prompt() -> str:
    return config.prompt_path("intent").read_text(encoding="utf-8")


def _format_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]


def _assemble(payload: dict[str, Any], request_id: UUID) -> Intent:
    """Код владеет тремя полями — модели их доверять незачем."""
    body = {key: value for key, value in payload.items() if key not in _CODE_OWNED_FIELDS}
    return Intent(
        **body,
        request_id=request_id,
        model=ModelName.FLASH,
        prompt_version=config.PROMPT_VERSIONS["intent"],
    )


def extract_intent(
    task_text: str,
    *,
    request_id: UUID,
    client: DeepSeekClient | None = None,
    logger: RunLogger | None = None,
) -> IntentExtraction:
    client = client or DeepSeekClient(logger=logger)
    system = load_system_prompt()
    user = f"Задача: «{task_text}»"

    errors: list[str] = []
    usage: dict[str, int] = {}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            payload, counters = client.chat_json(
                system=system,
                user=user,
                model=ModelName.FLASH.value,
                temperature=0.0,
            )
        except DeepSeekBadResponse as exc:
            # Неразбираемый ответ равносилен ответу не по схеме: и там, и там
            # интента нет, и оба лечатся одним повтором. Живой прогон дня 10
            # поймал такую осечку на скрининге; здесь она стоила бы всего скана.
            errors.append(str(exc))
            if logger:
                logger.info("intent_bad_response", attempt=attempt, detail=str(exc))
            if attempt == MAX_ATTEMPTS:
                break
            continue

        usage = counters

        try:
            intent = _assemble(payload, request_id)
        except ValidationError as exc:
            problems = _format_errors(exc)
            errors.extend(problems)
            if logger:
                logger.info("intent_invalid", attempt=attempt, problems=problems)
            if attempt == MAX_ATTEMPTS:
                break
            user = (
                f"Задача: «{task_text}»\n\n"
                "Предыдущий ответ не прошёл валидацию схемы:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\nВерни исправленный JSON."
            )
            continue

        return IntentExtraction(status="ok", intent=intent, attempts=attempt, usage=usage)

    return IntentExtraction(status="failed", attempts=MAX_ATTEMPTS, errors=errors, usage=usage)
