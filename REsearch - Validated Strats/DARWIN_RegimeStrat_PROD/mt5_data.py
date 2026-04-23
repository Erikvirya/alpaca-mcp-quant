"""
MT5 data adapter for NQ futures on Darwinex Zero.
Fetches daily OHLCV from the MT5 terminal's CME data feed.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mt5_initialize(config: dict) -> None:
    """
    Initialize MT5 connection. Raises RuntimeError on failure.
    """
    import MetaTrader5 as mt5

    kwargs = {
        "login": int(config["MT5_LOGIN"]),
        "password": config["MT5_PASSWORD"],
        "server": config["MT5_SERVER"],
    }
    if config.get("MT5_PATH"):
        kwargs["path"] = config["MT5_PATH"]

    if not mt5.initialize(**kwargs):
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: {error}")

    info = mt5.account_info()
    logger.info(
        f"mt5_connected=True login={info.login} server={info.server} "
        f"balance={info.balance:.0f} equity={info.equity:.0f}"
    )


def mt5_shutdown() -> None:
    """Shutdown MT5 connection."""
    import MetaTrader5 as mt5
    mt5.shutdown()
    logger.info("mt5_connected=False")


def fetch_daily(symbol: str, bars: int = 400) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from MT5.

    Args:
        symbol: MT5 symbol name (e.g. "NQ_H")
        bars: Number of daily bars to fetch (default 400 for indicator warm-up)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        DatetimeIndex (timezone-naive, daily)

    Raises:
        RuntimeError: if data fetch fails or insufficient bars.
    """
    import MetaTrader5 as mt5

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, bars)
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        raise RuntimeError(f"MT5 copy_rates failed for {symbol}: {error}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    if len(df) < 200:
        raise RuntimeError(
            f"Insufficient bars for {symbol}: got {len(df)}, need >=200 for indicator warm-up"
        )

    latest = df.index[-1].date()
    logger.info(
        f"source=MT5 symbol={symbol} bars={len(df)} latest_date={latest} "
        f"close={df['Close'].iloc[-1]:.2f}"
    )
    return df


def get_position(symbol: str) -> int:
    """
    Get current position size for a symbol from MT5.
    Returns number of contracts (positive=long, negative=short, 0=flat).
    """
    import MetaTrader5 as mt5

    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return 0

    total = 0
    for pos in positions:
        if pos.type == mt5.ORDER_TYPE_BUY:
            total += int(pos.volume)
        else:
            total -= int(pos.volume)
    return total


def get_position_info() -> tuple:
    """
    Get current position info from MT5.
    Returns (symbol, quantity) for the first position found, or (None, 0) if flat.
    """
    import MetaTrader5 as mt5

    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        return None, 0

    # Return first position found (should only be one NQ position)
    pos = positions[0]
    quantity = int(pos.volume) if pos.type == mt5.POSITION_TYPE_BUY else -int(pos.volume)
    return pos.symbol, quantity


def get_account_equity() -> float:
    """Get current account equity from MT5."""
    import MetaTrader5 as mt5
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("MT5 account_info() returned None")
    return float(info.equity)


def get_margin_free() -> float:
    """Get current free margin from MT5."""
    import MetaTrader5 as mt5
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("MT5 account_info() returned None")
    return float(info.margin_free)


def get_tick(symbol: str) -> dict:
    """
    Get current bid/ask/spread for a symbol.

    Returns:
        dict with bid, ask, spread, mid, last, time
    """
    import MetaTrader5 as mt5

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick returned None for {symbol}")

    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": tick.ask - tick.bid,
        "mid": (tick.bid + tick.ask) / 2.0,
        "last": tick.last,
        "time": tick.time,
    }


# Smart order defaults
# Strategy: resting GTC limit starting at mid-price. Wait 3s passive, then
# cancel + step 0.25pt toward aggressor every 1s until filled.
# Expected fill: ~50% at mid (0 half-spread cost), ~40% at ask (normal), ~10% worse.
# Much better than IOC-at-ask which always pays the full spread.
MAX_SPREAD_PTS    = 0.75   # Reject if spread > 0.75pts. Normal NQ: 0.25-0.50
SPREAD_WAIT_S     = 5      # Wait between spread re-checks (seconds)
SPREAD_RETRIES    = 6      # Max spread re-checks (6 x 5s = 30s total)
PASSIVE_WAIT_S    = 3.0    # Initial wait at mid-price resting limit
WALKBACK_STEP_PTS = 0.25   # Step toward aggressor per retry (1 NQ tick)
WALKBACK_WAIT_S   = 1.0    # Wait after each re-price
WALKBACK_RETRIES  = 8      # Max retries: 3 + 7x1 = 10s worst-case total


def _poll_order_fill(order_id: int, placed_at, wait_s: float):
    """
    Poll MT5 for up to wait_s seconds to see if a pending order filled.
    Returns fill_price (float) if filled, None if still pending at timeout.
    """
    import MetaTrader5 as mt5
    import time as _time
    from datetime import timedelta

    deadline = _time.time() + wait_s
    while _time.time() < deadline:
        _time.sleep(0.25)
        pending = mt5.orders_get(ticket=order_id)
        if pending is not None and len(pending) > 0:
            continue  # still in pending queue
        # Order left pending queue — check deal history for a fill.
        # Use a wide backward window (1 hour) to tolerate broker timezone offset
        # (Darwinex server = EET = UTC+2/3). Filter by order_id for correctness.
        deals = mt5.history_deals_get(
            placed_at - timedelta(hours=1),
            placed_at + timedelta(minutes=30),
        )
        if deals:
            for d in deals:
                if d.order == order_id:
                    logger.info(
                        f"fill_detected order={order_id} deal={d.ticket} "
                        f"price={d.price:.2f} type={d.type}"
                    )
                    return d.price
        return None  # order gone but no matching deal (cancelled/rejected)
    return None  # timed out — order still pending


def _cancel_pending(order_id: int) -> None:
    """Cancel a pending limit order by ticket."""
    import MetaTrader5 as mt5
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": order_id,
    })
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        logger.warning(f"cancel_pending order={order_id} retcode={retcode}")


def place_smart_order(symbol: str, volume: int, direction: str) -> dict:
    """
    Passive resting limit order, walking from mid toward aggressor until filled.

    Algorithm:
    1. Spread gate: reject if spread > MAX_SPREAD_PTS (retry up to 30s)
    2. Record initial_mid as slippage reference
    3. Place GTC BUY_LIMIT/SELL_LIMIT at current mid
    4. Wait PASSIVE_WAIT_S (3s) — cheapest fill if market ticks to us
    5. Not filled: cancel, step limit 0.25pt toward aggressor, wait 1s, repeat
    6. Each step: BUY limit += 0.25pt  |  SELL limit -= 0.25pt
    7. By attempt 2 (mid+0.25 = ask): fills immediately like a market order
    8. After WALKBACK_RETRIES exhausted: raise RuntimeError

    Fill price sequence (BUY, spread=0.25):
      attempt 1: mid        (-0.125pt vs ask)  wait 3s  ~50% fill
      attempt 2: mid+0.25   (= ask)            wait 1s  fills immediately
      attempt 3: ask+0.125                     wait 1s
      ...

    Returns:
        dict with: order_id, fill_price, volume, retcode,
                   initial_mid, spread_at_order, limit_price, attempts
    Raises:
        RuntimeError: if spread stays too wide or fill fails after all retries.
    """
    import MetaTrader5 as mt5
    import time as _time
    from datetime import datetime as _dt, timedelta

    # ── Step 1: Spread gate ──────────────────────────────────────────────────
    tick = None
    for i in range(SPREAD_RETRIES):
        tick = get_tick(symbol)
        if tick["spread"] <= MAX_SPREAD_PTS:
            break
        logger.warning(
            f"spread_wide spread={tick['spread']:.2f} max={MAX_SPREAD_PTS} "
            f"bid={tick['bid']:.2f} ask={tick['ask']:.2f} wait={SPREAD_WAIT_S}s "
            f"attempt={i+1}/{SPREAD_RETRIES}"
        )
        _time.sleep(SPREAD_WAIT_S)
    else:
        raise RuntimeError(
            f"Spread too wide after {SPREAD_RETRIES} checks: "
            f"spread={tick['spread']:.2f} bid={tick['bid']:.2f} ask={tick['ask']:.2f}"
        )

    # ── Step 2: Record initial mid (slippage benchmark) ──────────────────────
    initial_mid = tick["mid"]
    order_type = (
        mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
    )
    sign = 1 if direction == "BUY" else -1

    # ── Steps 3-8: Passive resting limit + walkback ──────────────────────────
    for attempt in range(1, WALKBACK_RETRIES + 1):
        tick = get_tick(symbol)
        mid = tick["mid"]
        step = (attempt - 1) * WALKBACK_STEP_PTS
        limit_price = round((mid + sign * step) * 4) / 4.0
        wait_s = PASSIVE_WAIT_S if attempt == 1 else WALKBACK_WAIT_S

        logger.info(
            f"smart_order attempt={attempt}/{WALKBACK_RETRIES} direction={direction} "
            f"volume={volume} limit={limit_price:.2f} mid={mid:.2f} "
            f"bid={tick['bid']:.2f} ask={tick['ask']:.2f} "
            f"step={step:+.2f}pt wait={wait_s}s initial_mid={initial_mid:.2f}"
        )

        placed_at = _dt.utcnow()
        result = mt5.order_send({
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       symbol,
            "volume":       float(volume),
            "type":         order_type,
            "price":        limit_price,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time":    mt5.ORDER_TIME_GTC,
            "comment":      "DarwinRegime",
        })

        if result is None:
            logger.error(f"order_send None attempt={attempt} error={mt5.last_error()}")
            _time.sleep(1)
            continue

        # Both DONE (10009) and PLACED (10008) mean the limit order was successfully added.
        # We REMOVE the check that returns immediately on TRADE_RETCODE_DONE.
        if result.retcode not in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED]:
            logger.warning(
                f"order_rejected retcode={result.retcode} "
                f"comment={result.comment} attempt={attempt}"
            )
            _time.sleep(1)
            continue

        order_id = result.order

        # Poll for fill during wait window (this will now catch immediate fills correctly)
        fill_price = _poll_order_fill(order_id, placed_at, wait_s)
        if fill_price is not None:
            logger.info(
                f"smart_fill fill={fill_price:.2f} limit={limit_price:.2f} "
                f"initial_mid={initial_mid:.2f} spread={tick['spread']:.2f} "
                f"attempt={attempt}"
            )
            return {
                "order_id":       order_id,
                "fill_price":     fill_price,
                "volume":         volume,
                "retcode":        mt5.TRADE_RETCODE_DONE,
                "initial_mid":    initial_mid,
                "spread_at_order": tick["spread"],
                "limit_price":    limit_price,
                "attempts":       attempt,
            }

        # Not filled — cancel and step toward aggressor
        _cancel_pending(order_id)
        logger.warning(
            f"no_fill attempt={attempt}/{WALKBACK_RETRIES} "
            f"limit={limit_price:.2f} stepping_toward_aggressor"
        )

    raise RuntimeError(
        f"Smart order failed after {WALKBACK_RETRIES} walkback attempts."
    )


def place_emergency_order(symbol: str, volume: int, direction: str) -> dict:
    """
    Uncapped market order for risk-off / emergency exits.
    No spread check, no deviation cap — WILL fill at any price.
    Use only when getting out is more important than fill quality.

    Returns:
        dict with: order_id, fill_price, volume, retcode,
                   initial_mid, spread_at_order, limit_price, attempts
    """
    import MetaTrader5 as mt5

    tick = get_tick(symbol)
    initial_mid = tick["mid"]

    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick["ask"] if direction == "BUY" else tick["bid"]

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "deviation": 50,  # 50 pts = effectively uncapped for NQ
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "DarwinRegime_EMERGENCY",
    }

    logger.warning(
        f"EMERGENCY_ORDER direction={direction} symbol={symbol} volume={volume} "
        f"price={price:.2f} bid={tick['bid']:.2f} ask={tick['ask']:.2f} "
        f"spread={tick['spread']:.2f} deviation=50"
    )

    result = mt5.order_send(request)
    if result is None:
        error = mt5.last_error()
        raise RuntimeError(f"Emergency order_send returned None: {error}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(
            f"Emergency order failed: retcode={result.retcode} comment={result.comment}"
        )

    fill = {
        "order_id": result.order,
        "fill_price": result.price,
        "volume": result.volume,
        "retcode": result.retcode,
        "initial_mid": initial_mid,
        "spread_at_order": tick["spread"],
        "limit_price": price,
        "attempts": 1,
    }
    logger.warning(
        f"EMERGENCY_FILL direction={direction} symbol={symbol} volume={volume} "
        f"fill_price={result.price:.2f} initial_mid={initial_mid:.2f} "
        f"spread={tick['spread']:.2f} order_id={result.order}"
    )
    return fill


def place_market_order(symbol: str, volume: int, direction: str,
                       risk_reducing: bool = False,
                       crisis_mode: bool = False) -> dict:
    """
    Main order entry point.

    Args:
        risk_reducing: If True, falls back to emergency order when smart fails.
                       Use when NOT trading is riskier than bad slippage.
        crisis_mode:   If True, bypass smart order entirely — go direct to
                       emergency market order. Use in CRISIS regime (VXN > 35)
                       where passive resting limits are inappropriate.
    """
    if crisis_mode:
        logger.warning(
            f"crisis_mode=True skipping_smart -> emergency "
            f"direction={direction} volume={volume}"
        )
        return place_emergency_order(symbol, volume, direction)
    try:
        return place_smart_order(symbol, volume, direction)
    except RuntimeError as e:
        if risk_reducing:
            logger.warning(
                f"smart_failed falling_back_to_emergency error={e} "
                f"direction={direction} volume={volume}"
            )
            return place_emergency_order(symbol, volume, direction)
        raise
