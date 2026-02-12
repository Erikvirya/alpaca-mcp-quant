# Alpaca MCP Quant: Backtester for Alpaca MCP

The problem: LLMs can hallucinate trading strategies that sound good but lose money.

The solution: an MCP server tool that forces a VectorBT backtest against Alpaca historical market data before you act on an idea.

## Features

- **Natural Language Backtesting**: Ask Claude to test a strategy idea and get backtest metrics (Sharpe, drawdown, etc.).
- **Institutional Data**: Powered by Alpaca Market Data API v2.
- **Dynamic Quant Sandbox**: Execute Claude-authored `strategy_code` in a restricted sandbox with `df`, `vbt`, `pd`, `np` preloaded.
- **Chart-Ready Output**: Returns `pf.stats()` plus equity/cumulative return timeseries for plotting.
- **Look-Ahead Bias Mitigation**: Signals are lagged by 1 bar and trades are executed at the next bar open.

## Quickstart

### 1) Install

Python 3.12 recommended.

```bash
pip install -r requirements.txt
```

### 2) Configure Alpaca credentials

Set environment variables:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_PAPER_TRADE=True`

### 3) Run the MCP server (stdio)

```bash
python -m alpaca_mcp_server.cli serve --transport stdio
```

### 4) Use the tool

Call the MCP tool:

- `execute_vectorbt_strategy(symbol, strategy_code)`

Example `strategy_code`:

```python
ma = df.close.rolling(20).mean()
entries = df.close.vbt.crossing_above(ma)
exits = df.close.vbt.crossing_below(ma)
pf = vbt.Portfolio.from_signals(df.close, entries, exits, freq="1D")
```

The tool returns JSON containing:

- `stats`
- `equity_curve`
- `cum_returns`
- `benchmark_cum_returns`

## Connect Claude Desktop (Windows)

### 1) Install deps into the same Python you will run from Claude

If you use a virtual environment, install requirements there:

```bash
pip install -r requirements.txt
```

### 2) Create/update your Claude Desktop MCP config

Edit (or create) this file:

`C:\Users\<YOU>\AppData\Roaming\Claude\claude_desktop_config.json`

Example config (replace paths + keys):

```json
{
  "mcpServers": {
    "alpaca-mcp-quant": {
      "command": "C:/Users/<YOU>/venvs/alpaca-mcp312/Scripts/python.exe",
      "args": ["-m", "alpaca_mcp_server.cli", "serve", "--transport", "stdio"],
      "env": {
        "PYTHONPATH": "C:/Users/<YOU>/Documents/ALPACA_BACKTESTING/alpaca-mcp-quant",
        "ALPACA_API_KEY": "YOUR_KEY",
        "ALPACA_SECRET_KEY": "YOUR_SECRET",
        "ALPACA_PAPER_TRADE": "True"
      }
    }
  }
}
```

Notes:

- `PYTHONPATH` must point at the repo root so `alpaca_mcp_server` can be imported.
- Use forward slashes (`/`) in JSON paths to avoid escaping issues.
- After editing the config, restart Claude Desktop.

### 3) Verify in Claude

After restart, you should see the server connect and the tool become available. Try asking Claude to run:

- `execute_vectorbt_strategy` with a simple moving average crossover strategy.

