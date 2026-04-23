#!/usr/bin/env python3
"""
Sandbox test for contract rollover scenarios.
Tests the rollover logic around historical rollover dates.
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
import contract_roll

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(SCRIPT_DIR, "trade_log.csv")
ROLLOVER_TEST_LOG = os.path.join(SCRIPT_DIR, "rollover_test_log.csv")


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


def simulate_mt5_position(symbol: str, quantity: int) -> int:
    """Simulate getting MT5 position for testing."""
    return quantity


def simulate_market_order(symbol: str, volume: int, direction: str) -> dict:
    """Simulate order execution for rollover testing."""
    import random
    
    base_price = 15000.0  # Mock NQ price
    slippage = random.uniform(0.1, 0.5)
    
    if direction == "SELL":
        fill_price = base_price - slippage
    else:
        fill_price = base_price + slippage
    
    return {
        "order_id": f"ROLL_TEST_{random.randint(1000000, 9999999)}",
        "fill_price": fill_price,
        "volume": volume,
        "retcode": 10009,  # TRADE_RETCODE_DONE
    }


def log_rollover_test(row: dict) -> None:
    """Log rollover test results."""
    fieldnames = [
        "test_date", "test_scenario", "current_symbol", "target_symbol", 
        "position_size", "roll_needed", "days_to_close_only", "roll_executed",
        "close_fill_price", "open_fill_price", "roll_slippage", "test_result"
    ]
    file_exists = os.path.exists(ROLLOVER_TEST_LOG)
    with open(ROLLOVER_TEST_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def test_rollover_scenario(test_date: date, scenario: str, current_symbol: str, position_size: int = 5) -> dict:
    """Test a specific rollover scenario."""
    logger.info(f"Testing {scenario} on {test_date}")
    
    # Get rollover status
    roll_needed, new_symbol, days_left = contract_roll.needs_roll(current_symbol, test_date)
    
    result = {
        "test_date": test_date.strftime("%Y-%m-%d"),
        "test_scenario": scenario,
        "current_symbol": current_symbol,
        "target_symbol": new_symbol or "N/A",
        "position_size": position_size,
        "roll_needed": roll_needed,
        "days_to_close_only": days_left,
        "roll_executed": False,
        "close_fill_price": "N/A",
        "open_fill_price": "N/A",
        "roll_slippage": "N/A",
        "test_result": "SUCCESS"
    }
    
    logger.info(f"roll_needed={roll_needed} new_symbol={new_symbol} days_left={days_left}")
    
    # Simulate roll execution if needed
    if roll_needed and new_symbol and position_size != 0:
        logger.info(f"Simulating roll: {current_symbol} -> {new_symbol}, qty={position_size}")
        
        try:
            # Simulate close old position
            close_direction = "SELL" if position_size > 0 else "BUY"
            close_fill = simulate_market_order(current_symbol, abs(position_size), close_direction)
            
            # Simulate open new position
            open_direction = "BUY" if position_size > 0 else "SELL"
            open_fill = simulate_market_order(new_symbol, abs(position_size), open_direction)
            
            # Calculate roll slippage
            if position_size > 0:  # Long position
                roll_slippage = close_fill["fill_price"] - open_fill["fill_price"]
            else:  # Short position
                roll_slippage = open_fill["fill_price"] - close_fill["fill_price"]
            
            result.update({
                "roll_executed": True,
                "close_fill_price": f"{close_fill['fill_price']:.2f}",
                "open_fill_price": f"{open_fill['fill_price']:.2f}",
                "roll_slippage": f"{roll_slippage:.2f}"
            })
            
            logger.info(f"Roll simulated: close={close_fill['fill_price']:.2f} "
                       f"open={open_fill['fill_price']:.2f} slippage={roll_slippage:.2f}")
            
        except Exception as e:
            logger.error(f"Roll simulation failed: {e}")
            result["test_result"] = "FAILED"
    
    return result


def run_rollover_tests():
    """Run comprehensive rollover scenario tests."""
    setup_logging("rollover_test")
    logger.info("=" * 60)
    logger.info("ROLLOVER SCENARIO TESTING")
    logger.info("=" * 60)
    
    # Test scenarios around key dates
    scenarios = [
        # December 2025 rollover (NQ_Z -> NQ_H)
        (date(2025, 12, 5), "Pre-Roll (5 days before)", "NQ_Z", 3),
        (date(2025, 12, 8), "Roll By Date", "NQ_Z", 3),
        (date(2025, 12, 10), "Post-Roll (2 days after)", "NQ_Z", 3),
        (date(2025, 12, 15), "Close-Only Date", "NQ_Z", 3),
        (date(2025, 12, 20), "After Expiry", "NQ_Z", 0),  # Should be flat
        
        # March 2026 rollover (NQ_H -> NQ_M) 
        (date(2026, 3, 5), "Pre-Roll (5 days before)", "NQ_H", 5),
        (date(2026, 3, 10), "Roll By Date", "NQ_H", 5),
        (date(2026, 3, 12), "Post-Roll (2 days after)", "NQ_H", 5),
        (date(2026, 3, 17), "Close-Only Date", "NQ_H", 5),
        (date(2026, 3, 22), "After Expiry", "NQ_H", 0),  # Should be flat
        
        # Edge cases
        (date(2026, 3, 10), "Zero Position (no roll needed)", "NQ_H", 0),
        (date(2026, 3, 10), "Short Position Roll", "NQ_H", -2),
    ]
    
    # Show current schedule
    logger.info("Current Expiry Schedule:")
    schedule = contract_roll.get_expiry_schedule(date(2026, 2, 20))
    for entry in schedule:
        logger.info(f"  {entry['symbol']}: roll_by={entry['roll_by']} "
                   f"close_only={entry['close_only']} expiry={entry['expiry']}")
    logger.info("")
    
    # Run each scenario
    for test_date, scenario, current_symbol, position_size in scenarios:
        result = test_rollover_scenario(test_date, scenario, current_symbol, position_size)
        log_rollover_test(result)
        logger.info(f"Result: {result['test_result']}")
        logger.info("-" * 40)
    
    # Summary
    logger.info("=" * 60)
    logger.info("ROLLOVER TEST SUMMARY")
    logger.info("=" * 60)
    
    # Read and summarize results
    if os.path.exists(ROLLOVER_TEST_LOG):
        df = pd.read_csv(ROLLOVER_TEST_LOG)
        total_tests = len(df)
        successful = len(df[df['test_result'] == 'SUCCESS'])
        rolls_executed = len(df[df['roll_executed'] == True])
        
        logger.info(f"Total scenarios tested: {total_tests}")
        logger.info(f"Successful tests: {successful}")
        logger.info(f"Rolls executed: {rolls_executed}")
        
        if rolls_executed > 0:
            roll_results = df[df['roll_executed'] == True]
            avg_slippage = roll_results['roll_slippage'].astype(float).mean()
            logger.info(f"Average roll slippage: {avg_slippage:.2f} points")
        
        logger.info(f"Results logged to: {ROLLOVER_TEST_LOG}")
    
    logger.info("ROLLOVER TESTING COMPLETE")


if __name__ == "__main__":
    run_rollover_tests()
