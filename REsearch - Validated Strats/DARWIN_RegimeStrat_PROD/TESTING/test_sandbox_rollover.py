#!/usr/bin/env python3
"""
Sandbox test for rollover logic around the December 2025 rollover (NQ_Z -> NQ_H).
Tests the complete flow:
  1. needs_roll with held symbol vs front symbol
  2. verify_symbol_available guard
  3. Smart roll on normal days / emergency on last day
  4. pin-nq_symbol guard (don't buy front month while holding old)
  5. Tracking update prevents duplicate rolls
"""
import csv
import logging
import os
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from logging_config import setup_logging
from strategy_logic import get_strategy_signals
from futures_config import NQ, COMMISSION_PER_SIDE
import vxn_data
import contract_roll

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLLOVER_SANDBOX_LOG = os.path.join(SCRIPT_DIR, "rollover_sandbox_log.csv")

# Dec 2025 rollover dates (calculated by contract_roll)
# Expiry: Dec 19 2025  |  Close-only: Dec 16 2025  |  Roll-by: Dec 9 2025
TEST_DATES = [
    date(2025, 12, 5),   # Pre-roll: roll not yet due
    date(2025, 12, 9),   # Roll-by date: roll triggered, symbol AVAILABLE
    date(2025, 12, 10),  # Day after roll: must NOT roll again
    date(2025, 12, 11),  # Still post-roll: no duplicate rolls
    date(2025, 12, 15),  # Emergency eve: days_left = 1
    date(2025, 12, 16),  # Close-only day: emergency roll
]


# ── Simulated MT5 state ──────────────────────────────────────────────────────

def simulate_market_order(symbol: str, volume: int, direction: str,
                          risk_reducing: bool = False, force_fail: bool = False) -> dict:
    if force_fail:
        raise RuntimeError("Simulated smart order failure (spread too wide)")
    slippage = random.uniform(0.1, 0.5)
    fill_price = 21000.0 + (slippage if direction == "BUY" else -slippage)
    return {
        "fill_price": fill_price,
        "volume": volume,
        "initial_mid": 21000.0,
        "spread_at_order": random.uniform(0.25, 0.75),
        "attempts": 1,
        "order_type": "EMERGENCY" if risk_reducing else "SMART",
    }


def simulate_emergency_order(symbol: str, volume: int, direction: str) -> dict:
    slippage = random.uniform(0.5, 2.0)
    fill_price = 21000.0 + (slippage if direction == "BUY" else -slippage)
    return {"fill_price": fill_price, "volume": volume, "order_type": "EMERGENCY"}


def fetch_historical_data(test_date: date) -> tuple:
    """Fetch real NQ + VXN data up to test_date."""
    start = test_date - timedelta(days=400)
    df = yf.download("NQ=F", start=start.strftime("%Y-%m-%d"),
                     end=(test_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                     progress=False)
    if df.empty or len(df) < 200:
        raise RuntimeError(f"Insufficient NQ data for {test_date}")
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vxn = vxn_data.fetch_vxn(lookback_days=400)
    vxn_aligned = vxn.reindex(close.index).ffill()
    return close, high, low, vxn_aligned


def notional_to_contracts(alloc, equity, price):
    if equity <= 0 or price <= 0 or alloc == 0:
        return 0
    n = round(equity * abs(alloc) / (price * NQ.point_value))
    return max(0, n) if alloc > 0 else -max(0, n)


def detect_regime(close, high, low, vxn):
    lc, lv = close.iloc[-1], vxn.iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    d = close.diff()
    gain = d.where(d > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-d.where(d < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).iloc[-1]
    if lv > 35:   return "CRISIS"
    if lc > ema50: return "TREND"
    if lc > ema20 and rsi > 50 and lv < 30: return "RECOVERY"
    return "DEFAULT"


def log_result(row: dict):
    fields = [
        "test_date", "scenario", "held_symbol", "front_symbol",
        "days_to_close_only", "roll_needed", "new_symbol_available",
        "is_emergency_day", "roll_executed", "roll_type",
        "nq_symbol_after_guard", "regime", "alloc",
        "target_cts", "actual_cts", "delta",
        "order_placed", "fill_price", "test_result", "notes",
    ]
    exists = os.path.exists(ROLLOVER_SANDBOX_LOG)
    with open(ROLLOVER_SANDBOX_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ── Core simulation ──────────────────────────────────────────────────────────

def run_one_day(test_date: date,
                scenario: str,
                held_symbol: str,      # what we currently hold
                held_qty: int,         # contracts held
                symbol_available: bool,# simulate Darwinex availability
                force_smart_fail: bool = False,
                close=None, high=None, low=None, vxn_aligned=None) -> dict:
    """Simulate one daily execution cycle with rollover logic."""

    result = {
        "test_date": str(test_date),
        "scenario": scenario,
        "held_symbol": held_symbol,
        "held_qty_in": held_qty,
        "test_result": "SUCCESS",
        "notes": "",
    }

    equity = 1_000_000.0
    nq_symbol = contract_roll.get_front_symbol(test_date)

    # ── 1. Contract roll check ────────────────────────────────────────────────
    current_symbol  = held_symbol
    current_quantity = held_qty

    roll_needed, new_symbol, days_left = (
        contract_roll.needs_roll(current_symbol, test_date)
        if (current_symbol and current_quantity != 0)
        else (False, None, 999)
    )
    is_emergency_day  = days_left <= 1
    roll_executed     = False
    roll_type         = "N/A"

    result.update({
        "front_symbol":      nq_symbol,
        "days_to_close_only": days_left,
        "roll_needed":        roll_needed,
        "new_symbol_available": symbol_available,
        "is_emergency_day":   is_emergency_day,
    })

    if roll_needed and new_symbol:
        if not symbol_available:
            logger.warning(f"roll_deferred new_symbol={new_symbol} "
                           f"reason=symbol_not_available days_left={days_left}")
            result["notes"] += f"Roll deferred: {new_symbol} not on Darwinex. "
            if days_left <= 2:
                result["notes"] += "CRITICAL: contact Darwinex! "
        else:
            # Execute roll
            try:
                if is_emergency_day:
                    close_dir = "SELL" if current_quantity > 0 else "BUY"
                    open_dir  = "BUY"  if current_quantity > 0 else "SELL"
                    simulate_emergency_order(current_symbol, abs(current_quantity), close_dir)
                    simulate_emergency_order(new_symbol,     abs(current_quantity), open_dir)
                    roll_type = "EMERGENCY"
                else:
                    close_dir = "SELL" if current_quantity > 0 else "BUY"
                    open_dir  = "BUY"  if current_quantity > 0 else "SELL"
                    simulate_market_order(current_symbol, abs(current_quantity), close_dir,
                                          risk_reducing=True, force_fail=force_smart_fail)
                    simulate_market_order(new_symbol,     abs(current_quantity), open_dir,
                                          risk_reducing=True, force_fail=force_smart_fail)
                    roll_type = "SMART"

                old_sym        = current_symbol
                current_symbol = new_symbol          # tracking update
                roll_executed  = True
                logger.info(f"roll_complete old={old_sym} new={current_symbol} type={roll_type}")
                result["notes"] += f"Roll {old_sym}->{current_symbol} ({roll_type}). "

            except Exception as e:
                logger.error(f"roll_failed error={e}")
                if is_emergency_day:
                    result["test_result"] = "ROLL_EMERGENCY_FAILED"
                    result["notes"] += f"EMERGENCY ROLL FAILED: {e}. "
                else:
                    result["notes"] += f"Smart roll failed, will retry tomorrow. "

    result.update({"roll_executed": roll_executed, "roll_type": roll_type,
                   "new_symbol": new_symbol or "N/A"})

    # ── 2. Pin nq_symbol guard ────────────────────────────────────────────────
    if current_symbol and current_symbol != nq_symbol:
        logger.warning(f"roll_incomplete pinning_execution_to={current_symbol}")
        nq_symbol = current_symbol
        result["notes"] += f"Pinned to {nq_symbol} (roll incomplete). "

    result["nq_symbol_after_guard"] = nq_symbol

    # ── 3. Strategy signal + order ────────────────────────────────────────────
    alloc_series = get_strategy_signals(close, high, low, vxn_aligned)
    target_alloc  = alloc_series.iloc[-1]
    last_close    = float(close.iloc[-1])
    regime        = detect_regime(close, high, low, vxn_aligned)

    target_cts    = notional_to_contracts(target_alloc, equity, last_close)
    target_cts    = max(-22, min(22, target_cts))
    actual_pos    = current_quantity if current_symbol == nq_symbol else 0
    delta         = target_cts - actual_pos

    result.update({
        "regime": regime, "alloc": f"{target_alloc:.2f}",
        "target_cts": target_cts, "actual_cts": actual_pos, "delta": delta,
    })

    if delta == 0:
        logger.info("delta=0 no_trade")
        result["order_placed"] = False
        result["fill_price"]   = last_close
    else:
        direction     = "BUY" if delta > 0 else "SELL"
        risk_reducing = (abs(target_cts) < abs(actual_pos)) or regime == "CRISIS"
        try:
            fill = simulate_market_order(nq_symbol, abs(delta), direction, risk_reducing)
            result["order_placed"] = True
            result["fill_price"]   = f"{fill['fill_price']:.2f}"
            logger.info(f"order_filled symbol={nq_symbol} dir={direction} "
                        f"vol={abs(delta)} fill={fill['fill_price']:.2f}")
        except Exception as e:
            logger.error(f"order_failed error={e}")
            result["order_placed"] = False
            result["test_result"]  = "ORDER_FAILED"

    return result, current_symbol   # return updated held symbol for next day


# ── Main ─────────────────────────────────────────────────────────────────────

def run_rollover_sandbox():
    setup_logging("rollover_sandbox")
    logger.info("=" * 60)
    logger.info("ROLLOVER SANDBOX TEST - Dec 2025 (NQ_Z -> NQ_H)")
    logger.info("=" * 60)

    # Pre-fetch all data once (speed)
    logger.info("Fetching market data for Dec 2025...")
    close, high, low, vxn_aligned = fetch_historical_data(date(2025, 12, 17))
    logger.info(f"Data ready: {len(close)} bars, last={close.index[-1].date()}")

    # Slice to each test date
    def slice_to(d: date):
        mask = close.index.date <= d
        return close[mask], high[mask], low[mask], vxn_aligned[mask]

    scenarios = [
        # (date, label, held_symbol, held_qty, symbol_available, force_smart_fail)
        (date(2025,12, 5), "Pre-roll (no action)",          "NQ_Z", 5, True,  False),
        (date(2025,12, 9), "Roll-by (symbol available)",    "NQ_Z", 5, True,  False),
        (date(2025,12,10), "Day after roll (no dup roll)",  "NQ_H", 5, True,  False),
        (date(2025,12,11), "Post-roll (stays NQ_H)",        "NQ_H", 5, True,  False),
        (date(2025,12, 9), "Roll-by (symbol NOT avail)",    "NQ_Z", 5, False, False),
        (date(2025,12, 9), "Roll-by (smart fails)",         "NQ_Z", 5, True,  True ),
        (date(2025,12,15), "Emergency eve (days_left=1)",   "NQ_Z", 5, True,  False),
        (date(2025,12,16), "Close-only (emergency)",        "NQ_Z", 5, True,  False),
    ]

    print(f"\n{'Date':<12} {'Scenario':<38} {'Held':>6} {'Front':>6} "
          f"{'DaysLeft':>9} {'RollNeeded':>11} {'Available':>10} "
          f"{'Emergency':>10} {'Rolled':>7} {'Type':>10} "
          f"{'Pinned?':>8} {'Delta':>6} {'Result'}")
    print("-" * 135)

    for test_date, scenario, held_sym, held_qty, sym_avail, force_fail in scenarios:
        c, h, l, v = slice_to(test_date)
        result, _ = run_one_day(
            test_date, scenario, held_sym, held_qty,
            sym_avail, force_fail, c, h, l, v
        )
        log_result(result)

        pinned = result["nq_symbol_after_guard"] != result["front_symbol"]
        print(
            f"{str(test_date):<12} {scenario:<38} {held_sym:>6} "
            f"{result['front_symbol']:>6} {result['days_to_close_only']:>9} "
            f"{str(result['roll_needed']):>11} {str(sym_avail):>10} "
            f"{str(result['is_emergency_day']):>10} {str(result['roll_executed']):>7} "
            f"{result['roll_type']:>10} {str(pinned):>8} "
            f"{str(result['delta']):>6} {result['test_result']}"
        )
        if result["notes"]:
            print(f"               > {result['notes']}")

    print(f"\nResults logged to: {ROLLOVER_SANDBOX_LOG}")
    logger.info("ROLLOVER SANDBOX COMPLETE")



if __name__ == "__main__":
    run_rollover_sandbox()
