# Windsurf Plan

This file captures the development plan used to implement the `execute_vectorbt_strategy` tool.

---

# Dynamic Quant Sandbox Plan

Implement a new MCP tool `execute_vectorbt_strategy` inside the existing `alpaca-mcp-server` that pulls recent Alpaca bars, preloads them as `df` alongside `vbt`, and safely executes user-provided `strategy_code` to produce a `vbt.Portfolio` and return `pf.stats()` as JSON.

## 0) Prerequisite: Make repo visible in the current workspace
- **Current state**: The opened workspace `c:\Users\virya\Documents\ALPACA_BACKTESTING` is empty, so I cannot locate `alpaca-mcp-server` files to extend.
- **Server location you provided**: `C:\Users\virya\AppData\Roaming\uv\tools\alpaca-mcp-server`
  - Tool code appears to live in `Lib\site-packages\alpaca_mcp_server\`.
  - MCP framework appears to be in `Lib\site-packages\mcp\`.
- **You will do** (pick one):
  - **Option A**: Open `C:\Users\virya\AppData\Roaming\uv\tools\alpaca-mcp-server` as the active workspace.
  - **Option B**: Copy the directory `Lib\site-packages\alpaca_mcp_server\` into `c:\Users\virya\Documents\ALPACA_BACKTESTING` (or another dev folder), then point the server to use that copy (editable/dev workflow).
- **Definition of done**: I can read the `alpaca_mcp_server` package files (tool registration, Alpaca client creation, and existing tool patterns).

**Chosen**: Option B.

## 0.1) Decision: edit installed tool vs. editable dev copy
- **Fastest demo path**: Modify files in `C:\Users\virya\AppData\Roaming\uv\tools\alpaca-mcp-server\Lib\site-packages\alpaca_mcp_server\` directly.
  - Pros: quickest to demo.
  - Cons: changes may be overwritten by tool updates / reinstalls.
- **Safer dev path**: Make a local copy of `alpaca_mcp_server` and run/install it in editable mode.
  - Pros: reproducible, version-controllable.
  - Cons: slightly more setup.

**Chosen**: Editable/dev copy.

## 1) Repo integration approach (once visible)
- **Locate**:
  - FastMCP server instantiation (where tools are registered).
  - Existing Alpaca client creation (API keys, paper env default, market data client).
  - Existing tool patterns (return types, error handling, parameter validation).
- **Add**:
  - A new tool function `execute_vectorbt_strategy(symbol: str, strategy_code: str)`.
  - A small internal module/service layer (e.g. `backtest/` or `services/backtest.py`) to keep logic modular.

## 2) Data acquisition (Alpaca Market Data API v2)
- **Bars endpoint**: Fetch the last ~1000 bars for `symbol`.
- **Timeframe**:
  - Use `1Day` bars by default.
- **DataFrame**:
  - Convert bars to a Pandas DataFrame indexed by timestamp.
  - Ensure OHLCV columns exist and are numeric (`open`, `high`, `low`, `close`, `volume`).
  - Expose the DataFrame to strategy code as `df`.

## 3) Dynamic strategy execution (VectorBT + exec sandbox)
- **Pre-loaded context** (available to `strategy_code`):
  - `df`: Pandas DataFrame with OHLCV.
  - `vbt`: VectorBT module.
  - `pd`, `np`: Pandas/Numpy helpers.
- **Contract**:
  - `strategy_code` must define `pf` as a `vbt.Portfolio`.
  - Example code the LLM can write:
    - `signals = df.close.vbt.crossing_above(df.close.rolling(20).mean())`
    - `pf = vbt.Portfolio.from_signals(df.close, signals)`
- **Sandboxing**:
  - Execute via `exec(strategy_code, sandbox_globals, sandbox_locals)`.
  - Provide a restricted set of builtins and allowed symbols.
  - Disallow `import` statements by omitting `__import__` from builtins (only `df`, `vbt`, `pd`, `np` are available).
  - Validate that `pf` exists after execution and is a `vbt.Portfolio`.
  - On errors, return a structured JSON error payload with message + traceback snippet.

## 4) Return value
Return `pf.stats()` as a JSON object.
- Convert the stats output to JSON-serializable primitives (handle NaN/Inf).
- Optionally include a minimal echo of inputs (symbol, timeframe, bar_count) for traceability.

Include a plot-friendly timeseries payload so the client can chart performance.
- Suggested fields:
  - `equity_curve`: list of `{t, equity}` points from `pf.value()`.
  - `cum_returns`: list of `{t, r}` points from `pf.value().pct_change().add(1).cumprod() - 1` (or VectorBT equivalent).
  - `benchmark_cum_returns` (optional): buy & hold cumulative returns on the same timestamps.

## 5) Performance + robustness
- **Target**: < 2 seconds on ~1000 bars (default timeframe).
- **Validation**:
  - Reject empty `strategy_code`.
  - Guard against empty/no data responses.
  - If `pf` is not defined after execution, return a JSON error that explains the contract (must define `pf`).
  - If `pf.stats()` contains NaN/Inf, normalize to JSON-safe values.
  - Ensure sandbox restricts dangerous builtins and filesystem/network access.
- **Dependencies**:
  - Ensure `vectorbt>=0.26` and `alpaca-trade-api>=3.2` are in project deps.

## 6) Minimal test/demo checklist (manual or automated, depending on repo)
- Call `execute_vectorbt_strategy(symbol='NVDA', strategy_code=...)` with a minimal strategy that defines `pf`.
- Confirm the tool returns JSON containing the expected `pf.stats()` keys and that error cases are informative.
