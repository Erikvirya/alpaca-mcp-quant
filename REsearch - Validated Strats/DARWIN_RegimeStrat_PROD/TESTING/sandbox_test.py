#!/usr/bin/env python3
"""
Sandbox test environment for live trading pipeline.
Uses yesterday's market data to simulate today's execution.
Tests the complete pipeline: data fetch, signal generation, order routing, logging.
"""
import csv
import logging
import os
import sys
from datetime import datetime, date, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from logging_config import setup_logging
from strategy_logic import get_strategy_signals
from futures_config import NQ, COMMISSION_PER_SIDE
import vxn_data
import notify

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(SCRIPT_DIR, "trade_log.csv")
SANDBOX_LOG = os.path.join(SCRIPT_DIR, "sandbox_test_log.csv")


def get_front_month_symbol(mt5_config: dict = None) -> str:
    """Get current front month NQ symbol for Darwinex (sandbox version - no validation)."""
    now = datetime.now()
    month = now.month
    
    # CME quarterly contracts: H(Mar), M(Jun), U(Sep), Z(Dec)
    if month <= 3:
        return "NQ_H"  # March contract
    elif month <= 6:
        return "NQ_M"  # June contract  
    elif month <= 9:
        return "NQ_U"  # September contract
    else:
        return "NQ_Z"  # December contract


def load_config() -> dict:
    """Load config.env as a dict."""
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


def simulate_mt5_connection():
    """Simulate MT5 connection and return mock data."""
    logger.info("sandbox_mode=True mt5_connection=SIMULATED")
    return True


def simulate_bar_poll(target_date: date) -> dict:
    """
    Simulate the D1 bar poll from execute_trade.py.

    Darwinex MT5 server runs EET (UTC+2 winter / UTC+3 summer). D1 bar open
    timestamps are broker-local 00:00 EET stored as Unix → when converted to
    UTC pandas Timestamp they land on the *previous* calendar day.

    Tests both the old broken date-check and the new timestamp-based check.
    Returns a dict with results for both.
    """
    chicago_tz = ZoneInfo("America/Chicago")
    eet_tz = ZoneInfo("Europe/Helsinki")  # EET: UTC+2 winter, UTC+3 summer
    utc_tz = timezone.utc

    # Simulate: we are running on target_date at 21:02 UTC
    simulated_run_utc = datetime.combine(target_date, time(21, 2), tzinfo=utc_tz)

    # CME close for target_date session (always 16:00 Chicago)
    cme_close_chicago = datetime.combine(target_date, time(16, 0), tzinfo=chicago_tz)
    cme_close_utc = cme_close_chicago.astimezone(utc_tz)

    # If script fires before CME close (winter: bar closes at 22:00 UTC), compute wait
    earliest_start = cme_close_utc + timedelta(minutes=2)
    wait_needed_s = max(0, (earliest_start - simulated_run_utc).total_seconds())

    # New check threshold
    today_session_open_utc = (cme_close_utc - timedelta(hours=25)).replace(tzinfo=None)

    # Simulate Darwinex EET-stamped bar for target_date:
    # Bar opens at 00:00 EET on target_date → converted to UTC Unix → pandas Timestamp
    bar_open_eet = datetime.combine(target_date, time(0, 0), tzinfo=eet_tz)
    bar_open_utc = bar_open_eet.astimezone(utc_tz).replace(tzinfo=None)
    simulated_bar_ts = pd.Timestamp(bar_open_utc)

    # OLD check (broken): compare UTC date of bar timestamp vs UTC date of run time
    old_check_passes = simulated_bar_ts.date() >= simulated_run_utc.date()

    # NEW check (fixed): compare UTC timestamps
    new_check_passes = simulated_bar_ts >= pd.Timestamp(today_session_open_utc)

    eet_offset = bar_open_eet.utcoffset()
    logger.info(
        f"[BAR_POLL_SIM] target_date={target_date} "
        f"cme_close_utc={cme_close_utc.strftime('%H:%M UTC')} "
        f"eet_offset={eet_offset} "
        f"bar_open_utc={bar_open_utc} "
        f"session_open_threshold={today_session_open_utc} "
        f"wait_needed_s={wait_needed_s:.0f}"
    )
    logger.info(
        f"[BAR_POLL_SIM] OLD_check(date_compare): bar_date={simulated_bar_ts.date()} "
        f">= run_date={simulated_run_utc.date()} -> {old_check_passes} "
        f"{'PASS' if old_check_passes else 'FAIL <-- BROKEN'}"
    )
    logger.info(
        f"[BAR_POLL_SIM] NEW_check(ts_compare): bar_ts={simulated_bar_ts} "
        f">= threshold={today_session_open_utc} -> {new_check_passes} "
        f"{'PASS' if new_check_passes else 'FAIL'}"
    )

    return {
        "target_date": target_date,
        "cme_close_utc": cme_close_utc,
        "eet_offset_hours": eet_offset.total_seconds() / 3600,
        "bar_open_utc": bar_open_utc,
        "today_session_open_utc": today_session_open_utc,
        "wait_needed_s": wait_needed_s,
        "old_check_passes": old_check_passes,
        "new_check_passes": new_check_passes,
    }


def simulate_account_equity():
    """Return mock account equity."""
    return 1000000.0  # $1M virtual equity


def simulate_position():
    """Return mock current position."""
    return 0  # Starting from flat


def simulate_market_order(symbol: str, volume: int, direction: str,
                          risk_reducing: bool = False,
                          base_price: float = 21000.0) -> dict:
    """
    Simulate order execution with realistic slippage.
    Returns mock fill data.
    """
    import random
    
    if risk_reducing:
        # Emergency order: worse fill but guaranteed
        slippage = random.uniform(0.5, 2.0)  # 0.5-2 pts slippage
        if direction == "SELL":
            fill_price = base_price - slippage
        else:
            fill_price = base_price + slippage
        order_type = "EMERGENCY"
    else:
        # Smart order: better fill but might fail
        if random.random() < 0.95:  # 95% fill rate
            slippage = random.uniform(0.1, 0.5)  # 0.1-0.5 pts slippage
            if direction == "SELL":
                fill_price = base_price - slippage
            else:
                fill_price = base_price + slippage
            order_type = "SMART"
        else:
            raise RuntimeError("Smart order failed - market moved")
    
    return {
        "order_id": f"SANDBOX_{random.randint(1000000, 9999999)}",
        "fill_price": fill_price,
        "volume": volume,
        "retcode": 10009,  # TRADE_RETCODE_DONE
        "initial_mid": base_price,
        "spread_at_order": random.uniform(0.25, 0.75),
        "limit_price": fill_price,
        "attempts": 1,
        "order_type": order_type,
    }


def log_sandbox_trade(row: dict) -> None:
    """Append a row to sandbox_test_log.csv."""
    fieldnames = [
        "test_date", "test_time", "regime", "target_alloc", "target_contracts", 
        "prev_contracts", "delta_contracts", "signal_close_price", "initial_mid", 
        "fill_price", "slippage_total_pts", "slippage_total_bps",
        "slippage_spread_pts", "slippage_walkback_pts",
        "spread_at_order", "fill_attempts", "order_type", "execution_time_ms",
        "equity", "vxn_level", "commission", "contract_symbol", "test_result"
    ]
    file_exists = os.path.exists(SANDBOX_LOG)
    with open(SANDBOX_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_sandbox_test(target_date: date = None) -> None:
    """
    Run sandbox test for specified date.
    If target_date is None, uses yesterday's date.
    Exercises: bar poll simulation, signal generation, order routing.
    """
    setup_logging("sandbox")
    logger.info("=" * 60)
    logger.info("SANDBOX TEST START")

    if target_date is None:
        # Skip back to last weekday if today is Monday (yesterday = Sunday)
        yesterday = date.today() - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)
        target_date = yesterday
    
    logger.info(f"test_date={target_date} sandbox_mode=True")

    config = load_config()
    import contract_roll as cr
    nq_symbol = cr.get_front_symbol(target_date)

    try:
        # ── Simulate MT5 Connection ──
        simulate_mt5_connection()

        # ── Bar Poll Simulation (exercises DST + EET timezone logic) ──
        logger.info("-" * 40)
        logger.info("BAR POLL SIMULATION")
        poll_results = simulate_bar_poll(target_date)
        bar_poll_ok = poll_results["new_check_passes"]
        if not bar_poll_ok:
            raise RuntimeError(
                f"Bar poll NEW check failed for {target_date} — "
                f"bar_ts={poll_results['bar_open_utc']} < threshold={poll_results['today_session_open_utc']}"
            )
        if poll_results["old_check_passes"]:
            logger.warning("[BAR_POLL_SIM] old date-check accidentally passes for this date — "
                           "verify on a date where broker bar lands on prev UTC day")
        logger.info(f"bar_poll_simulation=PASS wait_needed={poll_results['wait_needed_s']:.0f}s")
        logger.info("-" * 40)
        
        # ── Fetch Historical Data (yesterday's data) ──
        logger.info("fetching_historical_data mode=YESTERDAY")
        
        # Fetch NQ data (using yfinance as fallback for sandbox)
        nq_ticker = "NQ=F"
        start_date = target_date - timedelta(days=400)
        
        df = yf.download(nq_ticker, start=start_date.strftime('%Y-%m-%d'), 
                          end=target_date.strftime('%Y-%m-%d'), progress=False)
        
        if df.empty or len(df) < 200:
            raise RuntimeError(f"Insufficient NQ data for {target_date}")
        
        # Fetch VXN for the same date range as NQ (yfinance direct for historical compat)
        vxn_raw = yf.download("^VXN", start=start_date.strftime('%Y-%m-%d'),
                              end=target_date.strftime('%Y-%m-%d'),
                              progress=False, multi_level_index=False)
        vxn = vxn_raw["Close"].dropna()
        if vxn.empty:
            raise RuntimeError(f"VXN download returned no data for {target_date}")
        vxn_aligned = vxn.reindex(df.index).ffill()
        
        if vxn_aligned.isna().all():
            raise RuntimeError("VXN alignment failed — no overlapping dates")
        
        # Get yesterday's data (last complete bar)
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        
        last_close = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
        last_vxn = vxn_aligned.iloc[-1] if isinstance(vxn_aligned, pd.Series) else vxn_aligned[-1]
        
        logger.info(f"data_fetched nq_bars={len(df)} vxn_bars={len(vxn_aligned)} "
                   f"last_nq_close={last_close:.2f} last_vxn={last_vxn:.2f}")
        
        # ── Strategy Signal ──
        alloc = get_strategy_signals(close, high, low, vxn_aligned)
        target_alloc = alloc.iloc[-1]
        
        if np.isnan(target_alloc):
            logger.error("reason=nan_allocation test_result=FAILED")
            return
        
        # Detect regime
        regime = detect_regime(close, high, low, vxn_aligned)
        
        # ── Position Sizing ──
        equity = simulate_account_equity()
        target_contracts = notional_to_contracts(target_alloc, equity, last_close)
        
        # Max position cap
        max_cts = 22  # From backtest limits
        target_contracts = max(-max_cts, min(max_cts, target_contracts))
        
        # ── Position Reconciliation ──
        actual_position = simulate_position()
        delta = target_contracts - actual_position
        
        logger.info(
            f"regime={regime} alloc={target_alloc:.2f} target_cts={target_contracts} "
            f"actual_cts={actual_position} delta={delta:+d} equity={equity:.0f} "
            f"vxn={last_vxn:.2f} close={last_close:.2f}"
        )
        
        # ── Simulate Order Execution ──
        test_start = datetime.now()
        test_result = "SUCCESS"
        fill_price = last_close  # default
        commission = 0.0
        order_type = "NONE"
        initial_mid = last_close  # default for no-trade case
        spread_at_order = 0.0
        fill_attempts = 0
        slippage_total_pts = 0.0
        slippage_spread_pts = 0.0
        slippage_walkback_pts = 0.0
        slippage_total_bps = 0.0
        
        if delta == 0:
            logger.info("delta=0 no_trade test_result=NO_ACTION")
        else:
            direction = "BUY" if delta > 0 else "SELL"
            volume = abs(delta)
            
            crisis_mode = (regime == "CRISIS")
            risk_reducing = crisis_mode or (abs(target_contracts) < abs(actual_position))
            
            order_mode = "CRISIS_EMERGENCY" if crisis_mode else ("RISK_OFF" if risk_reducing else "RISK_ON")
            logger.info(
                f"order_mode={order_mode} target={target_contracts} actual={actual_position} delta={delta:+d}"
            )
            
            try:
                fill = simulate_market_order(nq_symbol, volume, direction,
                                              risk_reducing=risk_reducing,
                                              base_price=last_close)
                fill_price = fill["fill_price"]
                initial_mid = fill.get("initial_mid", fill_price)
                spread_at_order = fill.get("spread_at_order", 0.0)
                fill_attempts = fill.get("attempts", 1)
                order_type = fill.get("order_type", "UNKNOWN")
                commission = volume * COMMISSION_PER_SIDE
                
                # Slippage decomposition
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
                    f"attempts={fill_attempts} commission={commission:.2f} "
                    f"order_type={order_type}"
                )
                
            except Exception as e:
                logger.error(f"order_failed error={e} test_result=FAILED")
                test_result = "FAILED"
                fill_price = last_close
                initial_mid = last_close
                spread_at_order = 0.0
                fill_attempts = 1
                slippage_total_pts = 0.0
                slippage_spread_pts = 0.0
                slippage_walkback_pts = 0.0
                slippage_total_bps = 0.0
        
        test_end = datetime.now()
        execution_time_ms = (test_end - test_start).total_seconds() * 1000
        
        # ── Log Sandbox Test Results ──
        log_sandbox_trade({
            "test_date": target_date.strftime("%Y-%m-%d"),
            "test_time": datetime.now().strftime("%H:%M:%S"),
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
            "order_type": order_type,
            "execution_time_ms": f"{execution_time_ms:.0f}",
            "equity": f"{equity:.0f}",
            "vxn_level": f"{last_vxn:.2f}",
            "commission": f"{commission:.2f}",
            "contract_symbol": nq_symbol,
            "test_result": test_result,
        })
        
        # ── Test Summary ──
        logger.info(f"test_complete result={test_result} execution_time={execution_time_ms:.0f}ms")
        
        if delta != 0 and test_result == "SUCCESS":
            logger.info(f"trade_executed direction={direction} volume={volume} "
                       f"fill_price={fill_price:.2f} slippage={slippage_total_pts:+.2f}pts")
        
    except Exception as e:
        logger.error(f"sandbox_test_failed error={e}", exc_info=True)
        logger.info("test_result=FAILED")
    
    logger.info("SANDBOX TEST END")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Allow command line date override: python sandbox_test.py 2026-02-18
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print("Usage: python sandbox_test.py [YYYY-MM-DD]")
            sys.exit(1)
    else:
        target_date = None  # Use yesterday
    
    run_sandbox_test(target_date)
