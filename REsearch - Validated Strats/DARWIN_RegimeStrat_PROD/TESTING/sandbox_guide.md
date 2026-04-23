# Sandbox Testing Guide

## Overview
The sandbox environment tests the complete live trading pipeline using historical data, simulating today's execution with yesterday's market data.

## Files
- `sandbox_test.py` - Main sandbox test runner
- `sandbox_test_log.csv` - Test results log (auto-created)

## Usage

### Basic test (yesterday's data)
```bash
python sandbox_test.py
```

### Test specific date
```bash
python sandbox_test.py 2026-02-18
```

## What it tests

### 1. Data Pipeline
- Fetches NQ and VXN historical data
- Validates data alignment and freshness
- Handles missing data scenarios

### 2. Strategy Logic
- Regime detection (CRISIS, TREND, RECOVERY, RANGE, DEFAULT)
- Signal generation with same logic as live
- Position sizing (notional-based)

### 3. Order Routing
- **Risk-off/Crisis**: Emergency order simulation (worse fill, guaranteed)
- **Risk-on**: Smart order simulation (better fill, 95% success rate)
- Realistic slippage modeling

### 4. Edge Cases
- NaN allocations
- Failed orders (smart order 5% failure rate)
- Zero delta scenarios
- Crisis regime handling

## Simulated Components

### MT5 Connection
- Mock connection (always succeeds)
- Mock account equity ($1M)
- Mock position tracking

### Order Execution
| Order Type | Fill Rate | Slippage | Description |
|------------|-----------|----------|-------------|
| Emergency | 100% | 0.5-2.0 pts | Risk-off/Crisis exits |
| Smart | 95% | 0.1-0.5 pts | Risk-on entries |
| Smart Failed | 5% | N/A | Market moved against |

### Slippage Model
- **Total slippage**: Fill vs signal close
- **Spread cost**: Fill vs initial mid-price  
- **Walkback cost**: Market drift during execution

## Output

### Console Log
```
2026-02-20 14:30:00 - sandbox_test - INFO - test_date=2026-02-19 sandbox_mode=True
2026-02-20 14:30:01 - sandbox_test - INFO - data_fetched nq_bars=399 vxn_bars=399 last_nq_close=15234.50 last_vxn=28.75
2026-02-20 14:30:01 - sandbox_test - INFO - regime=TREND alloc=0.85 target_cts=5 actual_cts=0 delta=+5
2026-02-20 14:30:01 - sandbox_test - INFO - order_mode=RISK_ON target=5 actual=0 delta=+5
2026-02-20 14:30:01 - sandbox_test - INFO - fill_price=15234.75 initial_mid=15234.50 signal_close=15234.50 slip_total=+0.25pts
2026-02-20 14:30:01 - sandbox_test - INFO - test_complete result=SUCCESS execution_time=45ms
```

### CSV Log (sandbox_test_log.csv)
```csv
test_date,test_time,regime,target_alloc,target_contracts,prev_contracts,delta_contracts,...
2026-02-19,14:30:01,TREND,0.8500,5,0,5,15234.50,15234.50,15234.75,0.25,1.6,...
```

## Test Scenarios to Run

### 1. Normal Market Day
```bash
python sandbox_test.py 2026-02-18  # Regular trading day
```

### 2. Crisis Day (high VXN)
```bash
python sandbox_test.py 2026-03-04  # Find a high VIX/VXN day
```

### 3. Range Market
```bash
python sandbox_test.py 2026-01-15  # Low volatility day
```

### 4. Recovery Day
```bash
python sandbox_test.py 2026-02-12  # Post-dip recovery
```

## Validation Checklist

For each test run, verify:
- [ ] Regime detected correctly
- [ ] Allocation makes sense for regime
- [ ] Position sizing matches backtest logic
- [ ] Order type matches risk profile
- [ ] Slippage within expected ranges
- [ ] CSV log populated correctly
- [ ] Console logs readable and complete

## Integration with Live Pipeline

The sandbox uses the same:
- `strategy_logic.py` - Identical signal generation
- `futures_config.py` - Same contract specs
- `vxn_data.py` - Same volatility data source
- Logging format - Matches live structure

Only differences:
- MT5 calls replaced with simulations
- yfinance for NQ data (instead of MT5)
- No actual trades or emails

## Continuous Testing

Add to daily validation:
```bash
# Run yesterday's test
python sandbox_test.py

# Check results
python analyze_sandbox_results.py  # (create this to summarize test log)
```

This ensures any code changes work end-to-end before live deployment.
