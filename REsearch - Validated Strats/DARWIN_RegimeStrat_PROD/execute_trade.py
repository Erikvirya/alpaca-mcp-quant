"""
Daily execution script for the Darwinex Zero NQ Futures pipeline.
Called by Windows Task Scheduler at 21:02 UTC Mon-Fri.

DST-adaptive: CME NQ daily bar closes at 16:00 Chicago Time.
  Summer (CDT, Mar-Nov): 16:00 CT = 21:00 UTC → bar ready ~21:01, ~1 min wait
  Winter (CST, Nov-Mar): 16:00 CT = 22:00 UTC → bar ready ~22:01, ~59 min wait
The bar freshness poll handles both automatically (65 min timeout).

Flow:
1. Load config, connect MT5
2. Check contract roll — execute if needed
3. Poll for today's D1 bar (DST-adaptive, up to 65 min)
4. Fetch VXN
5. Generate strategy signal
6. Size position (notional-based)
7. Reconcile with MT5 position (MT5 = truth)
8. Place order if delta != 0
9. Log fill + slippage
10. Email notification
"""
import csv
import logging
import os
import sys
import time as _time
from datetime import datetime, date, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from logging_config import setup_logging
from strategy_logic import get_strategy_signals
from futures_config import NQ, COMMISSION_PER_SIDE
import mt5_data
import vxn_data
import contract_roll
import notify

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(SCRIPT_DIR, "trade_log.csv")
MAX_RETRIES = 3
RETRY_DELAY_S = 2   # Minimal delay between retries to fit within the 1-minute 15:59 window
BAR_POLL_INTERVAL_S = 30    # Seconds between D1 bar freshness checks
BAR_POLL_MAX_WAIT_S = 1800  # Max wait for today's D1 bar (30 min — generous buffer)

# CME execution time in Chicago (15:59 CT, 1 min before 16:00 close)
CME_EXEC_TIME = dt_time(15, 59)  # 3:59 PM Chicago Time


def load_config() -> dict:
    config = {}
    config_path = os.path.join(SCRIPT_DIR, "config.env")
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


def notional_to_contracts(alloc: float, equity: float, price: float) -> int:
    """Notional-based sizing matching backtest logic exactly."""
    if equity <= 0 or price <= 0 or alloc == 0:
        return 0
    target_notional = equity * abs(alloc)
    contract_notional = price * NQ.point_value
    n = round(target_notional / contract_notional)
    n = max(0, n)
    return n if alloc > 0 else -n


def max_contracts_for_alloc(equity: float, price: float, max_alloc: float = 2.0) -> int:
    """Max contracts at maximum allocation (position cap)."""
    return abs(notional_to_contracts(max_alloc, equity, price))


def already_traded_today() -> bool:
    """Check if trade_log.csv has an entry for today."""
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.exists(TRADE_LOG):
        return False
    try:
        with open(TRADE_LOG, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date", "").startswith(today_str):
                    return True
    except Exception:
        pass
    return False


def detect_regime(close, high, low, vxn_series) -> str:
    """Detect the current regime for logging purposes."""
    last_close = close.iloc[-1]
    last_vxn = vxn_series.iloc[-1]
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi = (100 - (100 / (1 + gain / loss))).iloc[-1]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14).mean()
    up = high - high.shift(1)
    dn = low.shift(1) - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=close.index).ewm(alpha=1/14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).ewm(alpha=1/14, min_periods=14).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]

    if last_vxn > 35:
        return "CRISIS"
    elif last_close > ema_50:
        if (last_close > ema_20) and (rsi > 50) and (last_vxn < 30):
            return "RECOVERY"
        return "TREND"
    elif adx < 25 and last_vxn < 25:
        return "RANGE"
    else:
        return "DEFAULT"


def log_trade(row: dict) -> None:
    """Append a row to trade_log.csv."""
    fieldnames = [
        "date", "regime", "target_alloc", "target_contracts", "prev_contracts",
        "delta_contracts", "signal_close_price", "initial_mid", "fill_price",
        "slippage_total_pts", "slippage_total_bps",
        "slippage_spread_pts", "slippage_walkback_pts",
        "spread_at_order", "fill_attempts",
        "equity", "vxn_level", "commission", "contract_symbol", "order_type",
    ]
    file_exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run() -> None:
    """Main execution flow."""
    setup_logging("execution")
    logger.info("=" * 60)
    logger.info("execute_trade START")

    # Skip weekends
    today = datetime.utcnow()
    if today.weekday() >= 5:
        logger.info(f"reason=weekend day={today.strftime('%A')} skipping")
        return

    # Double-execution guard
    if already_traded_today():
        logger.info("reason=already_traded skipping")
        return

    config = load_config()
    nq_symbol = contract_roll.get_front_symbol(today.date())  # Use existing rollover logic
    point_value = float(config.get("POINT_VALUE", "20"))

    # Connect MT5
    try:
        mt5_data.mt5_initialize(config)
    except Exception as e:
        logger.error(f"mt5_connect_failed error={e}")
        subj, body = notify.format_alert_msg("MT5 CONNECTION FAILED", str(e))
        notify.send_email(subj, body, config)
        return

    try:
        # ── Contract Roll Check ──
        # Check what symbol we currently have position in
        current_pos_info = mt5_data.get_position_info()  # Returns (symbol, quantity)
        current_symbol = current_pos_info[0] if current_pos_info else None
        current_quantity = current_pos_info[1] if current_pos_info else 0
        
        if current_symbol and current_quantity != 0:
            roll_needed, new_symbol, days_left = contract_roll.needs_roll(current_symbol, today.date())
            
            # Check if this is emergency roll day (day before close-only)
            is_emergency_day = days_left <= 1
            
            if roll_needed and new_symbol:
                # Verify new contract is available on Darwinex before rolling
                new_symbol_available = contract_roll.verify_symbol_available(new_symbol)
                if not new_symbol_available:
                    logger.warning(
                        f"roll_deferred new_symbol={new_symbol} reason=symbol_not_available_on_darwinex "
                        f"days_left={days_left}"
                    )
                    if days_left <= 2:
                        # Getting critical — alert so user can contact Darwinex
                        subj, body = notify.format_alert_msg(
                            f"ROLL DEFERRED — {new_symbol} NOT AVAILABLE",
                            f"Roll from {current_symbol} to {new_symbol} is due but {new_symbol} "
                            f"is not yet available on Darwinex.\n"
                            f"Days to close-only: {days_left}\n"
                            f"ACTION REQUIRED: Contact Darwinex support to enable {new_symbol}."
                        )
                        notify.send_email(subj, body, config)
                else:
                    logger.info(f"roll_check old={current_symbol} new={new_symbol} qty={current_quantity} "
                               f"days_left={days_left} emergency={is_emergency_day}")
                    
                    try:
                        if is_emergency_day:
                            logger.warning("emergency_roll_triggered last_chance=True")
                            roll_result = contract_roll.emergency_roll_position(
                                current_symbol, new_symbol, current_quantity, mt5_data
                            )
                            roll_type = "EMERGENCY"
                        else:
                            logger.info("standard_roll_attempt smart_logic=True")
                            roll_result = contract_roll.roll_position(
                                current_symbol, new_symbol, current_quantity, mt5_data
                            )
                            roll_type = "SMART"
                        
                        # Update tracking after successful roll
                        old_symbol = current_symbol
                        current_symbol = new_symbol
                        logger.info(f"roll_tracking_updated old={old_symbol} now_holding={current_symbol}")
                        
                        subj, body = notify.format_alert_msg(
                            f"ROLL EXECUTED ({roll_type}): {old_symbol} -> {current_symbol}",
                            f"Type: {roll_type}\n"
                            f"Quantity: {current_quantity}\n"
                            f"Close price: {roll_result['close_fill']['fill_price']:.2f}\n"
                            f"Open price: {roll_result['open_fill']['fill_price']:.2f}\n"
                            f"Days to close-only: {days_left}"
                        )
                        notify.send_email(subj, body, config)
                        
                    except Exception as e:
                        logger.error(f"roll_failed error={e}")
                        if is_emergency_day:
                            # Last day and even emergency order failed — critical alert
                            subj, body = notify.format_alert_msg(
                                "CRITICAL: EMERGENCY ROLL FAILED",
                                f"Emergency roll failed on last possible day\n"
                                f"Symbol: {current_symbol} -> {new_symbol}\n"
                                f"Quantity: {current_quantity}\n"
                                f"Error: {e}\n"
                                f"DAYS TO CLOSE-ONLY: {days_left} - IMMEDIATE ACTION REQUIRED!"
                            )
                            notify.send_email(subj, body, config)
                        else:
                            # Smart roll failed on a normal day — will retry tomorrow
                            subj, body = notify.format_alert_msg(
                                f"ROLL FAILED — will retry tomorrow: {current_symbol} -> {new_symbol}",
                                f"Smart limit roll failed. Will retry on next daily run.\n"
                                f"Symbol: {current_symbol} -> {new_symbol}\n"
                                f"Quantity: {current_quantity}\n"
                                f"Days to close-only: {days_left}\n"
                                f"Error: {e}"
                            )
                            notify.send_email(subj, body, config)
        else:
            logger.info(f"no_position_to_check current_symbol={current_symbol} quantity={current_quantity}")

        # ── Guard: don't trade front-month until roll is complete ──
        # If we still hold the old contract (roll deferred/failed), trade against
        # what we actually hold — not the front-month — to avoid holding both.
        if current_symbol and current_symbol != nq_symbol:
            logger.warning(
                f"roll_incomplete holding={current_symbol} front={nq_symbol} "
                f"pinning_execution_to={current_symbol}"
            )
            nq_symbol = current_symbol

        # ── Live Equity from MT5 ──
        equity = mt5_data.get_account_equity()
        logger.info(f"live_equity={equity:.0f}")

        # ── Wait for execution time (DST-aware) ──
        df = None
        waited = 0
        
        # We target 15:59 CT (1 min before the 16:00 CME close halt).
        # This perfectly matches the backtest EOD price while avoiding the daily halt
        # and flawlessly handles Friday (which has no 17:00 reopen).
        chicago_now = datetime.now(ZoneInfo("America/Chicago"))
        target_start_chicago = datetime.combine(chicago_now.date(), CME_EXEC_TIME, tzinfo=ZoneInfo("America/Chicago"))
        
        logger.info(
            f"cme_exec_chicago={target_start_chicago.strftime('%Y-%m-%d %H:%M %Z')} "
            f"chicago_now={chicago_now.strftime('%Y-%m-%d %H:%M %Z')}"
        )
        
        # Sleep until target start time (robust loop to handle early wakeups)
        while True:
            current_chicago = datetime.now(ZoneInfo("America/Chicago"))
            if current_chicago >= target_start_chicago:
                break
            wait_seconds = (target_start_chicago - current_chicago).total_seconds()
            logger.info(f"waiting_for_cme_exec seconds={wait_seconds:.0f}")
            _time.sleep(min(wait_seconds, 60))
        
        # Bar freshness threshold: The MT5 D1 bar for today's session has today's calendar date.
        target_bar_date = target_start_chicago.date()

        while waited <= BAR_POLL_MAX_WAIT_S:
            df = mt5_data.fetch_daily(nq_symbol, bars=400)
            latest_bar_ts = df.index[-1]  # EET broker time, timezone-naive
            latest_bar_date = latest_bar_ts.date()
            if latest_bar_date >= target_bar_date:
                logger.info(
                    f"d1_bar_ready=True latest_bar_ts={latest_bar_ts} "
                    f"target_bar_date={target_bar_date} waited={waited}s"
                )
                break
            logger.info(
                f"d1_bar_ready=False latest_bar_ts={latest_bar_ts} "
                f"target_bar_date={target_bar_date} waited={waited}s "
                f"retrying_in={BAR_POLL_INTERVAL_S}s"
            )
            _time.sleep(BAR_POLL_INTERVAL_S)
            waited += BAR_POLL_INTERVAL_S
        else:
            msg = (
                f"D1 bar not available after {BAR_POLL_MAX_WAIT_S}s. "
                f"Latest bar: {df.index[-1]}, target_bar_date: {target_bar_date}"
            )
            logger.error(msg)
            subj, body = notify.format_alert_msg("STALE D1 BAR", msg)
            notify.send_email(subj, body, config)
            mt5_data.mt5_shutdown()
            return

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        vxn = vxn_data.fetch_vxn(lookback_days=400)
        # Align VXN to NQ dates
        vxn_aligned = vxn.reindex(close.index).ffill()

        if vxn_aligned.isna().all():
            raise RuntimeError("VXN alignment failed — no overlapping dates")

        last_close = close.iloc[-1]
        last_vxn = vxn_aligned.iloc[-1]

        # ── Strategy Signal ──
        alloc = get_strategy_signals(close, high, low, vxn_aligned)
        target_alloc = alloc.iloc[-1]

        # NaN guard
        if np.isnan(target_alloc):
            logger.error("reason=nan_allocation holding_current_position=True")
            subj, body = notify.format_alert_msg(
                "NaN ALLOCATION", "Strategy returned NaN. Holding current position."
            )
            notify.send_email(subj, body, config)
            mt5_data.mt5_shutdown()
            return

        # Detect regime for logging
        regime = detect_regime(close, high, low, vxn_aligned)

        # ── Position Sizing ──
        target_contracts = notional_to_contracts(target_alloc, equity, last_close)

        # Max position cap
        max_cts = max_contracts_for_alloc(equity, last_close)
        target_contracts = max(-max_cts, min(max_cts, target_contracts))

        # ── Position Reconciliation (MT5 = truth) ──
        actual_position = mt5_data.get_position(nq_symbol)
        delta = target_contracts - actual_position

        logger.info(
            f"regime={regime} alloc={target_alloc:.2f} target_cts={target_contracts} "
            f"actual_cts={actual_position} delta={delta:+d} equity={equity:.0f} "
            f"vxn={last_vxn:.2f} close={last_close:.2f}"
        )

        # ── Margin Check (only when increasing exposure) ──
        increasing_exposure = abs(target_contracts) > abs(actual_position)
        if delta != 0 and increasing_exposure:
            margin_free = mt5_data.get_margin_free()
            # Only the additional contracts need margin
            additional_cts = abs(target_contracts) - abs(actual_position)
            margin_needed = additional_cts * NQ.margin
            if margin_free < margin_needed:
                logger.warning(
                    f"margin_free={margin_free:.0f} margin_required={margin_needed:.0f} "
                    f"additional_cts={additional_cts} sufficient=False"
                )
                subj, body = notify.format_alert_msg(
                    "INSUFFICIENT MARGIN",
                    f"Need ${margin_needed:,.0f} for {additional_cts} additional contracts "
                    f"but only ${margin_free:,.0f} free.\n"
                    f"Target: {target_contracts}, Current: {actual_position}, Delta: {delta}"
                )
                notify.send_email(subj, body, config)
                mt5_data.mt5_shutdown()
                return

        # ── Execute Order ──
        fill_price = last_close  # default if no trade
        commission = 0.0

        if delta == 0:
            logger.info("delta=0 no_trade")
            # Send heartbeat
            subj, body = notify.format_heartbeat_msg(
                today.strftime("%Y-%m-%d"), actual_position, nq_symbol, equity, last_vxn
            )
            notify.send_email(subj, body, config)
        else:
            direction = "BUY" if delta > 0 else "SELL"
            volume = abs(delta)

            crisis_mode = (regime == "CRISIS")
            risk_reducing = crisis_mode or (abs(target_contracts) < abs(actual_position))

            order_mode = "CRISIS_EMERGENCY" if crisis_mode else ("RISK_OFF" if risk_reducing else "RISK_ON")
            logger.info(
                f"order_mode={order_mode} target={target_contracts} actual={actual_position} delta={delta:+d}"
            )

            retry_count = 0
            for retry_count in range(MAX_RETRIES):
                try:
                    fill = mt5_data.place_market_order(
                        nq_symbol, volume, direction,
                        risk_reducing=risk_reducing,
                        crisis_mode=crisis_mode,
                    )
                    fill_price = fill["fill_price"]
                    initial_mid = fill.get("initial_mid", fill_price)
                    spread_at_order = fill.get("spread_at_order", 0.0)
                    fill_attempts = fill.get("attempts", 1)
                    commission = volume * COMMISSION_PER_SIDE
                    break
                except Exception as e:
                    logger.error(f"order_failed error={e} retry_count={retry_count+1}/{MAX_RETRIES}")
                    if retry_count < MAX_RETRIES - 1:
                        _time.sleep(RETRY_DELAY_S)
                        continue
                    else:
                        raise
            else:
                subj, body = notify.format_alert_msg(
                    "ORDER FAILED AFTER RETRIES",
                    f"Symbol: {nq_symbol}\nDirection: {direction}\n"
                    f"Volume: {volume}\nError: {e}"
                )
                notify.send_email(subj, body, config)
                mt5_data.mt5_shutdown()
                return

            # Slippage decomposition:
            #   total     = fill vs signal close  (what matters for P&L)
            #   spread    = fill vs initial mid   (cost of crossing spread)
            #   walkback  = initial mid vs signal close (market drift during retries)
            sign = 1 if direction == "BUY" else -1
            slippage_total_pts = (fill_price - last_close) * sign
            slippage_spread_pts = (fill_price - initial_mid) * sign
            slippage_walkback_pts = (initial_mid - last_close) * sign
            slippage_total_bps = slippage_total_pts / last_close * 10000 if last_close > 0 else 0

            logger.info(
                f"fill_price={fill_price:.2f} initial_mid={initial_mid:.2f} "
                f"signal_close={last_close:.2f} spread={spread_at_order:.2f} "
                f"slip_total={slippage_total_pts:+.2f}pts "
                f"slip_spread={slippage_spread_pts:+.2f}pts "
                f"slip_walkback={slippage_walkback_pts:+.2f}pts "
                f"attempts={fill_attempts} commission={commission:.2f}"
            )

            # Log trade
            log_trade({
                "date": today.strftime("%Y-%m-%d %H:%M:%S"),
                "regime": regime,
                "target_alloc": f"{target_alloc:.4f}",
                "target_contracts": target_contracts,
                "prev_contracts": actual_position,
                "delta_contracts": delta,
                "signal_close_price": f"{last_close:.2f}",
                "initial_mid": f"{initial_mid:.2f}",
                "fill_price": f"{fill_price:.2f}",
                "slippage_total_pts": f"{slippage_total_pts:.2f}",
                "slippage_total_bps": f"{slippage_total_bps:.1f}",
                "slippage_spread_pts": f"{slippage_spread_pts:.2f}",
                "slippage_walkback_pts": f"{slippage_walkback_pts:.2f}",
                "spread_at_order": f"{spread_at_order:.2f}",
                "fill_attempts": fill_attempts,
                "equity": f"{equity:.0f}",
                "vxn_level": f"{last_vxn:.2f}",
                "commission": f"{commission:.2f}",
                "contract_symbol": nq_symbol,
                "order_type": fill.get("order_type", "SMART"),
            })

            # Email notification
            subj, body = notify.format_trade_msg(
                date=today.strftime("%Y-%m-%d"),
                regime=regime,
                alloc=target_alloc,
                target_cts=target_contracts,
                prev_cts=actual_position,
                delta=delta,
                fill_price=fill_price,
                signal_close=last_close,
                slippage_pts=slippage_total_pts,
                slippage_bps=slippage_total_bps,
                equity=equity,
                vxn=last_vxn,
                contract_symbol=nq_symbol,
            )
            notify.send_email(subj, body, config)

    except Exception as e:
        logger.error(f"execution_error error={e}", exc_info=True)
        try:
            subj, body = notify.format_alert_msg("EXECUTION ERROR", str(e))
            notify.send_email(subj, body, config)
        except Exception:
            pass
    finally:
        mt5_data.mt5_shutdown()
        logger.info("execute_trade END")
        logger.info("=" * 60)


if __name__ == "__main__":
    run()
