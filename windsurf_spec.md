# Windsurf Spec: Alpaca MCP Quant

## Goal

Expose an MCP tool `execute_vectorbt_strategy(symbol, strategy_code)` that enables Claude to backtest strategy ideas using VectorBT against Alpaca historical bars.

## Core behavior

- Fetch last ~1000 `1Day` stock bars from Alpaca for `symbol`.
- Build a Pandas DataFrame `df` with OHLCV and UTC timestamp index.
- Execute `strategy_code` in a restricted sandbox:
  - Preloaded: `df`, `vbt`, `pd`, `np`
  - No imports allowed
  - `strategy_code` must define `pf` (`vbt.Portfolio`)
- Return JSON:
  - `pf.stats()` (JSON-safe)
  - plot-friendly timeseries: `equity_curve`, `cum_returns`, `benchmark_cum_returns`

## Safety

- Avoid look-ahead bias by enforcing 1-bar lag between signals and execution.
- Execute trades at the next bar open.

## Notes

- Recommended runtime: Python 3.12 (VectorBT/Numba compatibility).
