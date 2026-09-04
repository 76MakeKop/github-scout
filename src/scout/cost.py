"""Стоимость вызовов моделей по таблице цен `ARCHITECTURE.md`.

Кэшированный вход считается отдельно от обычного и это не мелочь: у V4-Flash
попавший в кэш токен дешевле непопавшего в 31 раз, а системный промпт Слоя 1
выносится в начало сообщения именно ради этого. На живом прогоне дня 5 из кэша
пришла половина входа — счёт по полной цене завысил бы результат на четверть,
при требуемой точности ±5%.

Цены живут здесь, а не в `config.py`, потому что это таблица одной темы, у которой
свой срок годности: сверка с прайсом DeepSeek раз в месяц (`ARCHITECTURE.md`).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from scout import config
from scout.schemas import ModelName, PricingWindow, TokenUsage

PRICE_UNIT = 1_000_000
"""Цены в таблице даны за миллион токенов."""

PEAK_MULTIPLIER = 2.0
"""Peak вдвое дороже off-peak — так устроен прайс DeepSeek."""


@dataclass(frozen=True)
class ModelPrice:
    """Цены за 1M токенов в off-peak. Peak получается умножением."""

    cache_hit: float
    input: float
    output: float
    peak_multiplier: float = PEAK_MULTIPLIER


PRICES: dict[ModelName, ModelPrice] = {
    ModelName.FLASH: ModelPrice(cache_hit=0.007, input=0.22, output=0.66),
    ModelName.PRO: ModelPrice(cache_hit=0.022, input=0.66, output=1.98),
    # У Qwen цена «без деления» (`ARCHITECTURE.md`) и кэша нет: множитель пика к ней
    # не применяется, а цена кэша равна обычной, чтобы формула осталась одной.
    ModelName.QWEN: ModelPrice(cache_hit=2.00, input=2.00, output=6.00, peak_multiplier=1.0),
}


def calculate_cost(
    model: ModelName | str,
    *,
    input_tokens: int,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
    is_peak: bool = False,
) -> float:
    """Стоимость одного вызова в долларах.

    `cached_input_tokens` — часть `input_tokens`, а не добавка к ним: провайдер
    отдаёт общий вход и внутри него долю кэша. Если счётчик пришёл испорченным
    и кэша «больше» всего входа, лишнее отбрасывается, а не уходит в минус.
    """
    price = PRICES[ModelName(model)]

    cached = max(0, min(cached_input_tokens, input_tokens))
    uncached = input_tokens - cached

    total = (
        cached * price.cache_hit + uncached * price.input + output_tokens * price.output
    ) / PRICE_UNIT

    return total * price.peak_multiplier if is_peak else total


def token_usage(
    model: ModelName | str,
    counters: Mapping[str, int],
    *,
    moment: datetime | None = None,
) -> TokenUsage:
    """Счётчики вызова → контракт `TokenUsage` с посчитанной стоимостью.

    Момент передаётся явно: от него зависит окно тарификации, и без явного входа
    тест на peak-цену зависел бы от часа, в который его запустили.
    """
    peak = config.is_peak(moment)
    name = ModelName(model)

    return TokenUsage(
        model=name,
        input_tokens=counters.get("input_tokens", 0),
        cached_input_tokens=counters.get("cached_input_tokens", 0),
        output_tokens=counters.get("output_tokens", 0),
        cost_usd=calculate_cost(
            name,
            input_tokens=counters.get("input_tokens", 0),
            cached_input_tokens=counters.get("cached_input_tokens", 0),
            output_tokens=counters.get("output_tokens", 0),
            is_peak=peak,
        ),
        pricing_window=PricingWindow.PEAK if peak else PricingWindow.OFF_PEAK,
    )
