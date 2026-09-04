"""Учёт стоимости: цены из таблицы `ARCHITECTURE.md`, кэш считается отдельно.

Все ожидаемые суммы посчитаны здесь руками по формуле «токены × цена / 1M»,
а не сняты с вывода кода: критерий приёмки дня 7 — сходимость с ручным расчётом,
и тест, снятый с реализации, проверял бы сам себя.
"""

from datetime import UTC, datetime

import pytest

from scout import config
from scout.cost import PRICES, calculate_cost, token_usage
from scout.schemas import ModelName, PricingWindow

# Замер живого прогона дня 5, этап интента: 1374 входных, из них 1280 из кэша.
INTENT_INPUT = 1374
INTENT_CACHED = 1280
INTENT_OUTPUT = 922

OFF_PEAK = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
PEAK = datetime(2026, 9, 4, 2, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# Цены
# --------------------------------------------------------------------------


def test_flash_cache_price_is_thirty_one_times_cheaper():
    """Ради этой разницы системный промпт и выносится в начало сообщения."""
    flash = PRICES[ModelName.FLASH]
    assert flash.cache_hit == 0.007
    assert flash.input == 0.22
    assert round(flash.input / flash.cache_hit) == 31


def test_prices_match_the_architecture_table():
    assert PRICES[ModelName.PRO].cache_hit == 0.022
    assert PRICES[ModelName.PRO].input == 0.66
    assert PRICES[ModelName.PRO].output == 1.98
    assert PRICES[ModelName.FLASH].output == 0.66


# --------------------------------------------------------------------------
# Расчёт
# --------------------------------------------------------------------------


def test_cost_flash_offpeak_with_cache():
    """Ручной расчёт по числам дня 5.

    кэш:  1280 * 0,007  / 1M = 0,00000896
    вход:   94 * 0,22   / 1M = 0,00002068
    выход: 922 * 0,66   / 1M = 0,00060852
    итого                    = 0,00063816
    """
    cost = calculate_cost(
        ModelName.FLASH,
        input_tokens=INTENT_INPUT,
        cached_input_tokens=INTENT_CACHED,
        output_tokens=INTENT_OUTPUT,
        is_peak=False,
    )
    assert cost == pytest.approx(0.00063816, rel=0.05)


def test_cost_flash_peak_is_exactly_double():
    off = calculate_cost(
        ModelName.FLASH,
        input_tokens=INTENT_INPUT,
        cached_input_tokens=INTENT_CACHED,
        output_tokens=INTENT_OUTPUT,
        is_peak=False,
    )
    on = calculate_cost(
        ModelName.FLASH,
        input_tokens=INTENT_INPUT,
        cached_input_tokens=INTENT_CACHED,
        output_tokens=INTENT_OUTPUT,
        is_peak=True,
    )
    assert on == pytest.approx(off * 2)


def test_cost_pro_peak_no_cache():
    """V4-Pro в пик, без кэша: 12000 * 1,32 + 1500 * 3,96, всё делить на 1M.

    вход:  12000 * 1,32 / 1M = 0,01584
    выход:  1500 * 3,96 / 1M = 0,00594
    итого                    = 0,02178
    """
    cost = calculate_cost(
        ModelName.PRO,
        input_tokens=12_000,
        cached_input_tokens=0,
        output_tokens=1_500,
        is_peak=True,
    )
    assert cost == pytest.approx(0.02178, rel=0.05)


def test_counting_cache_at_full_price_would_break_the_five_percent_target():
    """Обоснование правки `ARCHITECTURE.md`: разница выходит далеко за ±5%."""
    honest = calculate_cost(
        ModelName.FLASH,
        input_tokens=24_822,
        cached_input_tokens=12_544,
        output_tokens=13_139,
        is_peak=False,
    )
    naive = calculate_cost(
        ModelName.FLASH,
        input_tokens=24_822,
        cached_input_tokens=0,
        output_tokens=13_139,
        is_peak=False,
    )
    assert naive > honest * 1.05


def test_qwen_has_no_peak_discount():
    """`ARCHITECTURE.md`: у Qwen цена «без деления» — множитель пика к ней не применяется."""
    off = calculate_cost(ModelName.QWEN, input_tokens=1000, output_tokens=100, is_peak=False)
    on = calculate_cost(ModelName.QWEN, input_tokens=1000, output_tokens=100, is_peak=True)
    assert off == on


def test_cached_tokens_never_exceed_input():
    """Провайдер отдаёт кэш частью общего входа; обратное — испорченный счётчик."""
    cost = calculate_cost(
        ModelName.FLASH, input_tokens=100, cached_input_tokens=500, output_tokens=0, is_peak=False
    )
    assert cost == pytest.approx(100 * 0.007 / 1_000_000)


def test_zero_tokens_cost_nothing():
    assert calculate_cost(ModelName.FLASH, input_tokens=0, output_tokens=0, is_peak=False) == 0.0


# --------------------------------------------------------------------------
# Сборка TokenUsage
# --------------------------------------------------------------------------


def test_token_usage_carries_cost_and_window():
    usage = token_usage(
        ModelName.FLASH,
        {
            "input_tokens": INTENT_INPUT,
            "cached_input_tokens": INTENT_CACHED,
            "output_tokens": INTENT_OUTPUT,
        },
        moment=OFF_PEAK,
    )

    assert usage.model is ModelName.FLASH
    assert usage.pricing_window is PricingWindow.OFF_PEAK
    assert usage.cost_usd == pytest.approx(0.00063816, rel=0.05)


def test_token_usage_in_peak_marks_the_window():
    usage = token_usage(ModelName.FLASH, {"input_tokens": 1000}, moment=PEAK)

    assert usage.pricing_window is PricingWindow.PEAK
    assert usage.cost_usd == pytest.approx(1000 * 0.44 / 1_000_000)


def test_token_usage_survives_missing_counters():
    """Провайдер может не прислать `cached_input_tokens` — это не повод падать."""
    usage = token_usage(ModelName.FLASH, {"input_tokens": 10}, moment=OFF_PEAK)
    assert usage.cached_input_tokens == 0


# --------------------------------------------------------------------------
# Peak по UTC
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "peak"),
    [(0, False), (1, True), (3, True), (4, False), (5, False), (6, True), (9, True), (10, False)],
)
def test_peak_detection_by_utc_hour(hour, peak):
    """`CLAUDE.md`: peak — 01:00–04:00 и 06:00–10:00 UTC, границы полуоткрыты."""
    assert config.is_peak(datetime(2026, 9, 4, hour, 0, tzinfo=UTC)) is peak


def test_peak_is_counted_in_utc_not_local_time():
    """Полночь по Актобе (UTC+5) — это 19:00 UTC предыдущего дня, off-peak."""
    local_midnight = datetime(2026, 9, 4, 19, 0, tzinfo=UTC)
    assert config.is_peak(local_midnight) is False


# --------------------------------------------------------------------------
# Peak только по будням (прайс DeepSeek: Monday through Friday)
# --------------------------------------------------------------------------

FRIDAY = datetime(2026, 9, 4, 2, 0, tzinfo=UTC)
SATURDAY = datetime(2026, 9, 5, 2, 0, tzinfo=UTC)
SUNDAY = datetime(2026, 9, 6, 7, 0, tzinfo=UTC)
MONDAY = datetime(2026, 9, 7, 7, 0, tzinfo=UTC)


def test_friday_peak_hour_is_peak():
    assert FRIDAY.weekday() == 4
    assert config.is_peak(FRIDAY) is True


def test_saturday_peak_hour_is_offpeak():
    """Суббота 02:00 UTC попадает в окно по часам, но выходные тарифицируются дёшево."""
    assert SATURDAY.weekday() == 5
    assert config.is_peak(SATURDAY) is False


def test_sunday_peak_hour_is_offpeak():
    assert SUNDAY.weekday() == 6
    assert config.is_peak(SUNDAY) is False


def test_monday_peak_hour_is_peak():
    assert MONDAY.weekday() == 0
    assert config.is_peak(MONDAY) is True


def test_weekend_call_is_priced_at_offpeak():
    """Цена, а не только флаг: без проверки дня недели счёт вырос бы вдвое."""
    weekend = token_usage(ModelName.FLASH, {"input_tokens": 1000}, moment=SATURDAY)
    weekday = token_usage(ModelName.FLASH, {"input_tokens": 1000}, moment=FRIDAY)

    assert weekend.pricing_window is PricingWindow.OFF_PEAK
    assert weekday.pricing_window is PricingWindow.PEAK
    assert weekday.cost_usd == pytest.approx(weekend.cost_usd * 2)
