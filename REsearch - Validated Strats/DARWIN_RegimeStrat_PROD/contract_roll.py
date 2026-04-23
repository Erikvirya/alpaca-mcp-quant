"""
Quarterly NQ futures contract roll logic for Darwinex Zero MT5.

CRITICAL: Darwinex Zero force-closes positions at expiry.
          A "close-only" date ~3 days before expiry blocks new opens.
          Roll must happen BEFORE the close-only date.

Symbol format on Darwinex Zero: NQ_H (Mar), NQ_M (Jun), NQ_U (Sep), NQ_Z (Dec)
"""
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# NQ quarterly cycle: (month, letter)
NQ_CYCLE = [
    (3, "H"),   # March
    (6, "M"),   # June
    (9, "U"),   # September
    (12, "Z"),  # December
]

# Roll this many business days BEFORE the close-only date
ROLL_BUFFER_BDAYS = 5
# Close-only date is approximately this many calendar days before 3rd Friday
CLOSE_ONLY_OFFSET_DAYS = 3


def _third_friday(year: int, month: int) -> date:
    """Calculate the 3rd Friday of a given month."""
    first_day = date(year, month, 1)
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    return first_friday + timedelta(weeks=2)


def _close_only_date(year: int, month: int) -> date:
    """Estimate the close-only date for a given expiry month."""
    expiry_friday = _third_friday(year, month)
    return expiry_friday - timedelta(days=CLOSE_ONLY_OFFSET_DAYS)


def _subtract_bdays(d: date, n: int) -> date:
    """Subtract n business days from a date."""
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def get_expiry_schedule(today: date = None) -> list:
    """
    Get the next 4 quarterly expiry dates with close-only and roll dates.

    Returns:
        List of dicts: {month, letter, symbol, expiry, close_only, roll_by}
    """
    if today is None:
        today = date.today()

    schedule = []
    year = today.year

    for offset_year in range(0, 2):
        for month, letter in NQ_CYCLE:
            y = year + offset_year
            expiry = _third_friday(y, month)
            close_only = _close_only_date(y, month)
            roll_by = _subtract_bdays(close_only, ROLL_BUFFER_BDAYS)

            if expiry > today:
                schedule.append({
                    "month": month,
                    "letter": letter,
                    "symbol": f"NQ_{letter}",
                    "expiry": expiry,
                    "close_only": close_only,
                    "roll_by": roll_by,
                })
            if len(schedule) >= 4:
                return schedule

    return schedule


def get_front_symbol(today: date = None) -> str:
    """
    Get the current front-month NQ symbol.
    Switches to next contract when today >= roll_by date.
    """
    schedule = get_expiry_schedule(today)
    if not schedule:
        raise RuntimeError("No future expiry dates found")

    if today is None:
        today = date.today()

    # If we're past the roll_by date of the first contract, use the second
    if today >= schedule[0]["roll_by"] and len(schedule) > 1:
        return schedule[1]["symbol"]
    return schedule[0]["symbol"]


def needs_roll(current_symbol: str, today: date = None) -> tuple:
    """
    Check if the current contract needs to be rolled.

    Args:
        current_symbol: Currently held symbol (e.g. "NQ_H")
        today: Date to check (default: today)

    Returns:
        (needs_roll: bool, new_symbol: str or None, days_to_close_only: int)
    """
    if today is None:
        today = date.today()

    front = get_front_symbol(today)
    if front != current_symbol:
        schedule = get_expiry_schedule(today)
        # Find the current symbol's close-only date
        for entry in schedule:
            if entry["symbol"] == current_symbol:
                days_left = (entry["close_only"] - today).days
                logger.warning(
                    f"roll_needed=True old_symbol={current_symbol} "
                    f"new_symbol={front} days_to_close_only={days_left}"
                )
                return True, front, days_left
        # Current symbol not in schedule (already expired?)
        logger.warning(
            f"roll_needed=True old_symbol={current_symbol} "
            f"new_symbol={front} days_to_close_only=UNKNOWN"
        )
        return True, front, 0

    # Check how close we are to roll
    schedule = get_expiry_schedule(today)
    if schedule:
        days_left = (schedule[0]["close_only"] - today).days
        if days_left <= ROLL_BUFFER_BDAYS + 2:
            logger.info(
                f"roll_needed=False symbol={current_symbol} "
                f"days_to_close_only={days_left} approaching_roll=True"
            )
        return False, None, days_left

    return False, None, 999


def verify_symbol_available(symbol: str) -> bool:
    """Check if a symbol is available and has market data on Darwinex MT5."""
    try:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"symbol_not_found symbol={symbol}")
            return False
        if not info.visible:
            # Try to make it visible in MarketWatch
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid == 0:
            logger.warning(f"symbol_no_quote symbol={symbol}")
            return False
        logger.info(f"symbol_available symbol={symbol} bid={tick.bid:.2f}")
        return True
    except Exception as e:
        logger.error(f"symbol_check_failed symbol={symbol} error={e}")
        return False


def roll_position(old_symbol: str, new_symbol: str, quantity: int, mt5_data_module) -> dict:
    """
    Execute a contract roll: close old position, open new position.
    Uses smart limit logic with market order fallback.

    Args:
        old_symbol: Symbol to close (e.g. "NQ_H")
        new_symbol: Symbol to open (e.g. "NQ_M")
        quantity: Number of contracts (positive = long)
        mt5_data_module: The mt5_data module (for place_market_order)

    Returns:
        dict with close_fill and open_fill details

    Raises:
        RuntimeError: if either leg fails.
    """
    logger.info(
        f"roll_start old_symbol={old_symbol} new_symbol={new_symbol} quantity={quantity}"
    )

    # Close old position using smart limit logic
    close_direction = "SELL" if quantity > 0 else "BUY"
    try:
        close_fill = mt5_data_module.place_market_order(
            old_symbol, abs(quantity), close_direction, risk_reducing=True
        )
        logger.info(
            f"roll_close_done symbol={old_symbol} fill_price={close_fill['fill_price']:.2f} "
            f"type={close_fill.get('order_type', 'UNKNOWN')}"
        )
    except Exception as e:
        logger.error(f"roll_close_failed symbol={old_symbol} error={e}")
        raise RuntimeError(f"Roll close leg failed: {e}")

    # Open new position using smart limit logic
    open_direction = "BUY" if quantity > 0 else "SELL"
    try:
        open_fill = mt5_data_module.place_market_order(
            new_symbol, abs(quantity), open_direction, risk_reducing=True
        )
        logger.info(
            f"roll_open_done symbol={new_symbol} fill_price={open_fill['fill_price']:.2f} "
            f"type={open_fill.get('order_type', 'UNKNOWN')}"
        )
    except Exception as e:
        logger.error(f"roll_open_failed symbol={new_symbol} error={e}")
        raise RuntimeError(f"Roll open leg failed: {e}")

    logger.info(
        f"roll_complete old_symbol={old_symbol} new_symbol={new_symbol} "
        f"quantity={quantity} close_price={close_fill['fill_price']:.2f} "
        f"open_price={open_fill['fill_price']:.2f}"
    )

    return {"close_fill": close_fill, "open_fill": open_fill}


def emergency_roll_position(old_symbol: str, new_symbol: str, quantity: int, mt5_data_module) -> dict:
    """
    Emergency roll using direct market orders (failsafe).
    Used only when regular roll fails or on last possible day.

    Args:
        old_symbol: Symbol to close (e.g. "NQ_H")
        new_symbol: Symbol to open (e.g. "NQ_M")
        quantity: Number of contracts (positive = long)
        mt5_data_module: The mt5_data module

    Returns:
        dict with close_fill and open_fill details
    """
    logger.warning(
        f"emergency_roll_start old_symbol={old_symbol} new_symbol={new_symbol} quantity={quantity}"
    )

    # Close old position - direct market order
    close_direction = "SELL" if quantity > 0 else "BUY"
    try:
        # Use emergency order (uncapped market order)
        close_fill = mt5_data_module.place_emergency_order(
            old_symbol, abs(quantity), close_direction
        )
        logger.info(
            f"emergency_roll_close_done symbol={old_symbol} fill_price={close_fill['fill_price']:.2f}"
        )
    except Exception as e:
        logger.error(f"emergency_roll_close_failed symbol={old_symbol} error={e}")
        raise RuntimeError(f"Emergency roll close leg failed: {e}")

    # Open new position - direct market order
    open_direction = "BUY" if quantity > 0 else "SELL"
    try:
        open_fill = mt5_data_module.place_emergency_order(
            new_symbol, abs(quantity), open_direction
        )
        logger.info(
            f"emergency_roll_open_done symbol={new_symbol} fill_price={open_fill['fill_price']:.2f}"
        )
    except Exception as e:
        logger.error(f"emergency_roll_open_failed symbol={new_symbol} error={e}")
        raise RuntimeError(f"Emergency roll open leg failed: {e}")

    logger.warning(
        f"emergency_roll_complete old_symbol={old_symbol} new_symbol={new_symbol} "
        f"quantity={quantity} close_price={close_fill['fill_price']:.2f} "
        f"open_price={open_fill['fill_price']:.2f}"
    )

    return {"close_fill": close_fill, "open_fill": open_fill}


if __name__ == "__main__":
    # Quick sanity check
    today = date(2026, 2, 19)
    print(f"Today: {today}")
    print(f"Front symbol: {get_front_symbol(today)}")
    print()
    print("Expiry schedule:")
    for entry in get_expiry_schedule(today):
        print(
            f"  {entry['symbol']}: expiry={entry['expiry']} "
            f"close_only={entry['close_only']} roll_by={entry['roll_by']}"
        )
    print()
    needs, new_sym, days = needs_roll("NQ_H", today)
    print(f"NQ_H needs roll? {needs} -> {new_sym} (days to close-only: {days})")
