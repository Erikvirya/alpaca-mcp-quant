# Instructions

You are a client of the Alpaca MCP server. Your role:

- Call MCP tools to query market data, manage positions, and run backtests.
- NEVER modify, create, or delete any files in this project.
- NEVER edit server source code.
- If a tool call fails, report the error as-is.

## CRITICAL workflow rules (follow every time)

1. **READ THIS ENTIRE FILE before writing any strategy code.** Review the sandbox environment, DataFrame structure, pre-loaded variables, and common mistakes sections below FIRST. Do not guess — follow the documented API.
2. **ALWAYS show strategy_code before executing.** Before calling `execute_vectorbt_strategy`, you MUST display the full `strategy_code` in a fenced Python code block and briefly explain the logic. Only THEN call the tool. Never skip this step.
3. **After receiving results**, summarize the key stats (total return, Sharpe ratio, max drawdown, win rate) in a clean table.
4. **If a tool call fails**, show the error, explain the likely cause, and retry with a corrected strategy_code.

## Key MCP tools available

- `execute_vectorbt_strategy(symbol, strategy_code, ...)` — backtest a strategy
- `get_stock_bars` — historical OHLCV data
- `get_stock_quotes` — level 1 bid/ask
- `get_account` — account info
- `place_order` / `get_orders` — order management
- `get_positions` — current positions
- 40+ more trading/data tools

---

## execute_vectorbt_strategy — Full Reference

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | `str` or `list[str]` | required | Stock symbol(s). Single: `"AAPL"`. Multi: `["AAPL", "MSFT", "GOOG"]` |
| `strategy_code` | `str` | required | Python code that defines `pf` (a VectorBT Portfolio). See sandbox docs below. |
| `timeframe` | `str` | `"1Day"` | Bar timeframe: `"1Day"`, `"1Hour"`, `"5Min"`, `"15Min"`, `"1Week"`, etc. |
| `start` | `str` or `null` | `null` | Start date ISO string, e.g. `"2023-01-01"`. If omitted, auto-calculated from limit. |
| `end` | `str` or `null` | `null` | End date ISO string, e.g. `"2025-12-31"`. If omitted, defaults to now. |
| `limit` | `int` | `1000` | Max bars to fetch. **Auto-increased** when `start` is specified to cover the full date range (up to 10,000). |
| `max_seconds` | `int` | `300` | Max strategy execution time in seconds (30–600). Increase for heavy WFO strategies. |

### Sandbox Environment

Your `strategy_code` runs in a restricted sandbox. **Do NOT use import statements.** All libraries are pre-loaded:

#### Pre-loaded variables

| Variable | Type | Description |
|----------|------|-------------|
| `df` | `pd.DataFrame` | OHLCV data. Single symbol: flat columns. Multi-symbol: MultiIndex columns. |
| `dfs` | `dict[str, pd.DataFrame]` | Per-symbol DataFrames. Always available. `dfs['AAPL']` gives a flat OHLCV DataFrame. |
| `symbols` | `list[str]` | List of symbols **that have data** (some may be dropped if API returned no bars). |
| `all_requested_symbols` | `list[str]` | Full list of originally requested symbols (may include symbols with no data). |
| `vbt` | vectorbt | VectorBT library (proxied for look-ahead bias protection). |
| `pd` | pandas | Full pandas library. |
| `np` | numpy | Full numpy library. |
| `sm` | statsmodels.api | OLS regression, statistical models. |
| `statsmodels` | statsmodels | Full statsmodels package. |
| `coint` | function | Engle-Granger cointegration test (`from statsmodels.tsa.stattools`). |
| `adfuller` | function | Augmented Dickey-Fuller stationarity test (`from statsmodels.tsa.stattools`). |
| `scipy` | scipy | Full scipy library. |
| `scipy_stats` | scipy.stats | Statistical distributions, t-tests, z-scores, etc. |
| `itertools` | itertools | Combinations, permutations, etc. Use `itertools.combinations(symbols, 2)` for pairs. |
| `math` | math | Standard math functions (sqrt, log, ceil, floor, etc.). |
| `yf_download` | function | Fetch Yahoo Finance data: `yf_download('^VIX', start='2024-01-01')`. **Auto-aligned** to `df`'s index by default (pass `align=False` to get raw). |
| `align` | function | Align any external Series/DataFrame to `df`'s trading calendar: `aligned = align(my_data)`. Strips tz, normalizes dates, forward-fills. |
| `wfo_splits` | function | Generate rolling train/test window masks for Walk-Forward Optimization. See WFO section below. |
| `wfo_grid_search` | function | Evaluate a parameter grid on training data and return best params. See WFO section below. |
| `wfo_run` | function | Full WFO pipeline: split → optimize → stitch OOS equity → return portfolio. See WFO section below. |

#### Available builtins

`abs`, `min`, `max`, `sum`, `len`, `range`, `enumerate`, `zip`, `sorted`, `reversed`, `round`, `map`, `filter`, `any`, `all`, `isinstance`, `hasattr`, `getattr`, `print`, `float`, `int`, `str`, `bool`, `list`, `dict`, `set`, `tuple`, `slice`

### DataFrame structure

#### Single symbol (`symbol = "AAPL"`)

`df` has flat columns:

```
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df.index = DatetimeIndex (UTC)
```

Access: `df['Close']`, `df['Open']`, etc.

#### Multi-symbol (`symbol = ["AAPL", "MSFT"]`)

`df` has MultiIndex columns `(field, symbol)`:

```
df.columns = MultiIndex([('Open', 'AAPL'), ('Open', 'MSFT'),
                          ('High', 'AAPL'), ('High', 'MSFT'),
                          ('Close', 'AAPL'), ('Close', 'MSFT'), ...])
```

Access patterns:
- `df['Close']` — DataFrame of close prices, columns = symbols
- `df[('Close', 'AAPL')]` — Series of AAPL close prices
- `dfs['AAPL']` — flat OHLCV DataFrame for AAPL only
- `dfs['AAPL']['Close']` — Series of AAPL close prices

### Required output

Your `strategy_code` MUST define `pf` — a VectorBT Portfolio object:

```python
pf = vbt.Portfolio.from_signals(close, entries, exits)
```

### Safety: look-ahead bias protection

- All signals are automatically lagged by 1 bar.
- Trades execute at the next bar's Open price.
- You do NOT need to manually shift signals.

### Return payload

The tool returns JSON with:
- `stats` — `pf.stats()` as a dict (Sharpe, max drawdown, total return, etc.)
- `equity_curve` — timeseries of portfolio value
- `cum_returns` — cumulative return timeseries
- `benchmark_cum_returns` — buy-and-hold benchmark (equal-weighted for multi-symbol)
- `per_symbol_benchmark` — per-symbol cumulative returns (multi-symbol only)
- `timings_ms` — execution time breakdown
- `bar_count`, `cache_hit`, `multi_symbol`

---

## Strategy examples

### 1. Simple moving average crossover (single symbol)

```
symbol: "SPY"
strategy_code: |
  fast = vbt.MA.run(df['Close'], 10)
  slow = vbt.MA.run(df['Close'], 50)
  entries = fast.ma_crossed_above(slow)
  exits = fast.ma_crossed_below(slow)
  pf = vbt.Portfolio.from_signals(df['Close'], entries, exits)
```

### 2. RSI mean reversion (single symbol)

```
symbol: "AAPL"
strategy_code: |
  rsi = vbt.RSI.run(df['Close'], 14).rsi
  entries = rsi < 30
  exits = rsi > 70
  pf = vbt.Portfolio.from_signals(df['Close'], entries, exits)
```

### 3. Sector momentum (multi-symbol)

```
symbol: ["XLK", "XLF", "XLE", "XLV", "XLI"]
strategy_code: |
  close = df['Close']
  ma = vbt.MA.run(close, 50)
  entries = close > ma.ma
  exits = close < ma.ma
  pf = vbt.Portfolio.from_signals(close, entries, exits)
```

### 4. Pairs trading with cointegration (multi-symbol)

```
symbol: ["KO", "PEP"]
strategy_code: |
  a = dfs['KO']['Close']
  b = dfs['PEP']['Close']
  score, pvalue, _ = coint(a, b)
  X = sm.add_constant(a)
  model = sm.OLS(b, X).fit()
  hedge_ratio = model.params.iloc[1]
  spread = b - hedge_ratio * a
  z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
  entries = z < -1.5
  exits = abs(z) < 0.5
  pf = vbt.Portfolio.from_signals(b, entries, exits)
```

### 5. Dual momentum rotation (multi-symbol)

```
symbol: ["SPY", "EFA", "BND"]
strategy_code: |
  close = df['Close']
  ret = close.pct_change(252)
  best = ret.idxmax(axis=1)
  entries = pd.DataFrame(False, index=close.index, columns=close.columns)
  exits = pd.DataFrame(False, index=close.index, columns=close.columns)
  for sym in close.columns:
      entries[sym] = (best == sym) & (ret[sym] > 0)
      exits[sym] = (best != sym) | (ret[sym] <= 0)
  pf = vbt.Portfolio.from_signals(close, entries, exits)
```

### 6. RSI rotation across a basket

```
symbol: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
strategy_code: |
  close = df['Close']
  rsi = vbt.RSI.run(close, 14).rsi
  entries = rsi < 30
  exits = rsi > 70
  pf = vbt.Portfolio.from_signals(close, entries, exits)
```

### 7. Bollinger Band breakout (single symbol)

```
symbol: "TSLA"
strategy_code: |
  bb = vbt.BBANDS.run(df['Close'], window=20, alpha=2)
  entries = df['Close'] < bb.lower
  exits = df['Close'] > bb.upper
  pf = vbt.Portfolio.from_signals(df['Close'], entries, exits)
```

### 8. Stationarity check + mean reversion (single symbol)

```
symbol: "GLD"
strategy_code: |
  close = df['Close']
  result = adfuller(close.dropna())
  z = (close - close.rolling(30).mean()) / close.rolling(30).std()
  entries = z < -2
  exits = z > 0
  pf = vbt.Portfolio.from_signals(close, entries, exits)
```

---

## Available Portfolio constructors

All of these are available via `vbt.Portfolio.<method>`:

- `vbt.Portfolio.from_signals(close, entries, exits, ...)` — signal-based (entries/exits are auto-lagged)
- `vbt.Portfolio.from_orders(close, ...)` — order-based
- `vbt.Portfolio.from_returns(returns, ...)` — returns-based
- `vbt.Portfolio.from_holding(close, ...)` — buy-and-hold
- `vbt.Portfolio.from_random_signals(close, ...)` — random signal generation

### Direction parameter

Use these string values for `direction=`:
- `"longonly"` — long only (default)
- `"shortonly"` — short only (NOT `"short"`)
- `"both"` — both long and short

For long+short strategies with `from_signals`, use `short_entries` and `short_exits` kwargs:
```python
pf = vbt.Portfolio.from_signals(
    close, entries=long_entries, exits=long_exits,
    short_entries=short_entries, short_exits=short_exits,
    direction="both"
)
```

---

## Common mistakes to avoid

1. **Do NOT use import statements** — everything is pre-loaded.
2. **Column names are capitalized** — use `df['Close']`, NOT `df['close']`.
3. **Always define `pf`** — strategy_code must create a `pf = vbt.Portfolio...` variable.
4. **For multi-symbol, use `dfs` dict for per-symbol access** — `dfs['AAPL']['Close']`.
5. **Do NOT manually lag signals** — the sandbox handles look-ahead bias automatically.
6. **Use IEX-compatible symbols** — major US equities and ETFs work. OTC/illiquid may not.
7. **Direction strings** — use `"shortonly"` not `"short"`, `"longonly"` not `"long"`.
8. **All Portfolio constructors are available** — `from_signals`, `from_orders`, `from_returns`, `from_holding`, `from_random_signals`.

---

## Futures Backtesting (ES/MES)

The sandbox supports backtesting on S&P 500 futures via Yahoo Finance continuous contracts. Use `yf_download('ES=F')` to fetch E-mini S&P 500 data (2016–present) or `yf_download('MES=F')` for Micro E-mini (2019–present).

### Quick start — Futures backtest

```
symbol: "SPY"
start: "2016-01-01"
strategy_code: |
  # Use ES=F continuous contract instead of Alpaca SPY data
  es = yf_download('ES=F', start='2016-01-01', align=False)
  close = es['Close']
  vix = yf_download('^VIX', start='2016-01-01')['Close'].reindex(close.index).ffill()

  # Strategy logic (same as SPY — instrument-agnostic)
  ema_50 = close.ewm(span=50, adjust=False).mean()
  entries = close > ema_50
  exits = close < ema_50
  pf = vbt.Portfolio.from_signals(close, entries, exits, freq='1D')
```

### Available futures tickers (yfinance)

| Ticker | Instrument | Point Value | History |
|--------|-----------|-------------|---------|
| `ES=F` | E-mini S&P 500 | $50/pt | 2016+ |
| `MES=F` | Micro E-mini S&P 500 | $5/pt | 2019+ |
| `NQ=F` | E-mini Nasdaq-100 | $20/pt | 2016+ |
| `MNQ=F` | Micro E-mini Nasdaq-100 | $2/pt | 2019+ |

### Position sizing for futures

The strategy outputs an allocation signal (0.1–2.0). For live futures trading, use `futures_config.py` to convert to discrete contract counts:

```python
from futures_config import MES, allocation_to_contracts
n = allocation_to_contracts(alloc=1.2, account_equity=25000, price=6860, contract=MES)
# -> 16 MES contracts
```

### Notes
- `ES=F` and `MES=F` are **continuous front-month contracts** — yfinance auto-rolls quarterly. No manual rollover stitching needed for backtesting.
- Futures prices track SPY closely but are not identical (futures include cost-of-carry). Strategy signals (EMA, RSI, etc.) work the same way.
- For live trading, roll to the next quarterly contract 1-2 days before the 3rd Friday of Mar/Jun/Sep/Dec.
- No overnight financing fees (unlike CFDs) — this is the primary advantage of futures over CFD/stock-based execution.

---

## Walk-Forward Optimization (WFO) — Built-in Helpers

The sandbox includes three WFO utility functions that handle the rolling train/test split, parameter grid search, and OOS equity stitching.

### Quick start — `wfo_run` (one-liner WFO)

```
symbol: "SPY"
start: "2015-01-01"
max_seconds: 600
strategy_code: |
  close = df['Close']

  def my_strategy(close, sma_slow=200, ema_fast=50):
      sma = close.rolling(sma_slow).mean()
      ema = close.ewm(span=ema_fast).mean()
      entries = (close > sma) | (close > ema)
      exits = (close < sma) & (close < ema)
      return entries.fillna(False), exits.fillna(False)

  grid = {
      'sma_slow': [180, 200, 220],
      'ema_fast': [40, 50, 60],
  }

  result = wfo_run(close, grid, my_strategy, train_months=12, test_months=12)
  pf = result['pf']
```

### `wfo_run` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `close` | `pd.Series` | required | Full close price series (e.g. `df['Close']`) |
| `param_grid` | `dict` | required | `{param_name: [values]}` — all combinations are tested |
| `strategy_fn` | `callable` | required | `fn(close, **params) -> (entries, exits)` — must return boolean Series |
| `train_months` | `int` | `12` | Training window length in months |
| `test_months` | `int` | `12` | Out-of-sample test window length in months |
| `start_year` | `int` | auto | First year to begin testing. If None, inferred from data + train_months |
| `metric` | `str` | `'total_return'` | Metric to maximize: `'total_return'`, `'sharpe_ratio'`, `'sortino_ratio'`, `'calmar_ratio'`, `'max_drawdown'` (minimized) |
| `init_cash` | `float` | `100000` | Starting capital |

### `wfo_run` return value

Returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `pf` | VBT Portfolio | Portfolio from stitched OOS equity — assign to `pf` for the backtester |
| `oos_equity` | `pd.Series` | Full out-of-sample equity curve |
| `window_params` | `list[dict]` | Per-window details: train/test dates, best params, OOS return |
| `window_count` | `int` | Number of WFO windows executed |

### `wfo_splits` — manual train/test windows

For custom WFO loops where you need more control:

```python
for train_mask, test_mask in wfo_splits(df.index, train_months=12, test_months=12):
    train_close = df['Close'][train_mask]
    test_close = df['Close'][test_mask]
    # ... your optimization logic here
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index` | DatetimeIndex | required | `df.index` |
| `train_months` | `int` | `12` | Training window |
| `test_months` | `int` | `12` | Test window |
| `start_year` | `int` | auto | First year to begin |

Yields `(train_mask, test_mask)` — boolean Series aligned to the index.

### `wfo_grid_search` — parameter optimization

Evaluate all parameter combinations on training data:

```python
result = wfo_grid_search(train_close, grid, my_strategy, metric='sharpe_ratio')
best_params = result['best_params']  # e.g. {'sma_slow': 200, 'ema_fast': 50}
best_score = result['best_score']    # e.g. 1.23
all_results = result['results']      # list of all combos with scores
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `close` | `pd.Series` | required | Training close prices only |
| `param_grid` | `dict` | required | `{param_name: [values]}` |
| `strategy_fn` | `callable` | required | `fn(close, **params) -> (entries, exits)` |
| `metric` | `str` | `'total_return'` | Metric to maximize (or minimize for `'max_drawdown'`) |

### WFO with external data (VIX, yields, etc.)

Use `yf_download` (auto-aligned) and `align` to mix data sources safely:

```
symbol: "SPY"
start: "2015-01-01"
max_seconds: 600
strategy_code: |
  close = df['Close']
  vix = yf_download('^VIX', start='2015-01-01')['Close']  # auto-aligned to df

  def regime_strategy(close, sma=200, vix_thresh=25):
      sma_line = close.rolling(sma).mean()
      # vix is already aligned — safe to combine with close
      bull = (close > sma_line) & (vix < vix_thresh)
      bear = (close < sma_line) | (vix > 35)
      return bull.fillna(False), bear.fillna(False)

  grid = {'sma': [180, 200, 220], 'vix_thresh': [20, 25, 30]}
  result = wfo_run(close, grid, regime_strategy, train_months=12, test_months=12)
  pf = result['pf']
```

### Multi-source data alignment

All data in the sandbox is **tz-naive and date-normalized**:

- **`df` (Alpaca data)** — automatically stripped of UTC timezone and normalized to midnight
- **`yf_download(ticker, align=True)`** — Yahoo data auto-reindexed to `df`'s trading calendar with forward-fill (default)
- **`align(data)`** — manually align any external Series/DataFrame to `df`'s calendar

This means you can freely combine Alpaca prices with Yahoo VIX/yields without timezone or index mismatch errors.

### WFO tips

1. **Use `max_seconds=600`** for WFO strategies — they are computationally heavy.
2. **Keep grids small** — 3×3×3 = 27 combos per window is fine. 5×5×5 = 125 may timeout.
3. **Yearly cadence is safest** — `train_months=12, test_months=12`. Monthly re-optimization is too slow.
4. **`strategy_fn` must return boolean Series** — use `.fillna(False)` on all signals.
5. **Signals are auto-lagged** inside `wfo_grid_search` and `wfo_run` — do NOT manually shift.
6. **Equity chains automatically** — `wfo_run` passes ending equity as starting cash for the next window.

---

## Yahoo Finance Data Tool (fallback)

Use `get_yahoo_finance_data` for data not available on Alpaca — indices, volatility, yields, international markets, etc.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | `str` | required | Yahoo Finance ticker (e.g., `"^VIX"`, `"^GSPC"`, `"^TNX"`, `"GLD"`, `"TLT"`) |
| `period` | `str` | `"1y"` | Lookback — `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`. Ignored if `start` is set. |
| `interval` | `str` | `"1d"` | Bar size — `1m`, `5m`, `15m`, `1h`, `1d`, `1wk`, `1mo`. Intraday limited to last 60 days. |
| `start` | `str` | `""` | Start date `YYYY-MM-DD` (overrides period) |
| `end` | `str` | `""` | End date `YYYY-MM-DD` |

### Common tickers

| Ticker | Description |
|--------|-------------|
| `^VIX` | CBOE Volatility Index |
| `^GSPC` | S&P 500 Index |
| `^DJI` | Dow Jones Industrial Average |
| `^IXIC` | Nasdaq Composite |
| `^TNX` | 10-Year Treasury Yield |
| `^TYX` | 30-Year Treasury Yield |
| `^IRX` | 13-Week Treasury Bill |
| `GLD` | Gold ETF |
| `TLT` | 20+ Year Treasury Bond ETF |
| `DX-Y.NYB` | US Dollar Index |

### Example prompts

- "Get VIX data for the last 2 years"
- "Show S&P 500 weekly bars since 2020"
- "Fetch 10-year Treasury yield daily data for 2024"

### Response format

Returns JSON with `meta` (symbol, name, rows), `stats` (last, high, low, mean, total_return_pct, annualized_vol_pct), and `data` (array of `{date, open, high, low, close, volume}`). Capped at 2000 rows.

---

## DoltHub Options Data (free historical Greeks)

Two tools query the free [post-no-preference/options](https://www.dolthub.com/repositories/post-no-preference/options) DoltHub database. It covers **S&P 500 components + SPY + SPDR ETFs** from **2019 to present**, updated daily, with **full Greeks** (delta, gamma, theta, vega, rho, IV).

### `get_dolthub_options`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | `str` | required | Underlying (e.g., `"SPY"`, `"AAPL"`) |
| `start` | `str` | required | Start date `YYYY-MM-DD` |
| `end` | `str` | `""` (yesterday) | End date `YYYY-MM-DD` |
| `right` | `str` | `"both"` | `"call"`, `"put"`, or `"both"` |
| `max_dte` | `int` | `60` | Max days-to-expiration |

Returns option chain with: `date`, `expiration`, `strike`, `right`, `bid`, `ask`, `close` (mid), `iv`, `delta`, `gamma`, `theta`, `vega`, `rho`.

### `get_dolthub_volatility_history`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | `str` | required | Symbol (e.g., `"SPY"`) |
| `start` | `str` | `""` | Start date |
| `end` | `str` | `""` | End date |
| `limit` | `int` | `252` | Max rows |

Returns daily IV/HV snapshots: `hv_current`, `hv_week_ago`, `iv_current`, `iv_week_ago`, `iv_year_high`, `iv_year_low`, etc.

### Limitations

- **Only ~3 short-term expirations** per date (~2wk, ~4wk, ~8wk) — no LEAPs or monthlies beyond 8 weeks
- **S&P 500 components only** — no small caps or non-US
- **No OHLCV** — only bid/ask (close is computed as mid-price)
- Large date ranges may timeout on the DoltHub API

### Backtester integration

`execute_options_backtest` automatically tries DoltHub as a fallback between local cache and ThetaData. Data source priority:
1. **Local parquet cache** (`data/options_cache/`)
2. **DoltHub** (free Greeks, S&P 500 only)
3. **ThetaData API** (all symbols, needs Theta Terminal running)

The `timings` output includes `dolthub_used: true/false` to show which source was used.

---

## ThetaData Options Tools (FREE historical EOD)

These tools require the **Theta Terminal v3** running locally. It's a Java app that hosts a REST API on `http://127.0.0.1:25503`.

### Setup (one-time)

1. **Create a free account** at [thetadata.net/subscribe](https://www.thetadata.net/subscribe)
2. **Install Java 21+** — download from [adoptium.net](https://adoptium.net/)
3. **Download Theta Terminal v3** from [download-unstable.thetadata.us/ThetaTerminalv3.jar](https://download-unstable.thetadata.us/ThetaTerminalv3.jar)
4. **Save your credentials** — create a file `thetadata.properties` next to the jar:
   ```
   username=your_email@example.com
   password=your_password
   ```

### Launching the Theta Terminal

Open a terminal and run:
```
java -jar ThetaTerminalv3.jar
```

Keep it running while using the tools. Verify it works by opening `http://127.0.0.1:25503/v3/option/list/expirations?symbol=AAPL` in your browser.

To override the default URL, set the `THETADATA_URL` environment variable.

### Available tools

| Tool | Description |
|---|---|
| `get_theta_option_expirations(symbol)` | List all option expiration dates |
| `get_theta_option_strikes(symbol, expiration)` | List strikes for a given expiration |
| `get_theta_option_eod(symbol, expiration, strike, right, start_date, end_date)` | Historical EOD OHLCV + NBBO quote |

### Parameters

- `symbol` — underlying (e.g., `"AAPL"`, `"SPY"`)
- `expiration` — `"YYYY-MM-DD"` or `"YYYYMMDD"` or `"*"` for all
- `strike` — price in dollars (e.g., `"170.000"`) or `"*"` for all
- `right` — `"call"` or `"put"`
- `start_date` / `end_date` — `"YYYY-MM-DD"` or `"YYYYMMDD"`
- `max_dte` — optional, only contracts with DTE <= this value
- `num_strikes` — optional, N strikes above/below ATM

### Example prompts

- "List all AAPL option expirations"
- "Get SPY 550 call EOD from 2026-01-01 to 2026-02-13"
- "Show TSLA put option chain EOD, all strikes, expiring 2026-03-21, last 5 trading days"

### Notes

- EOD data is **FREE** — no paid subscription needed
- Data is generated at **17:15 ET** each day
- Results capped at **500 records** per call — use filters (`max_dte`, `num_strikes`, specific strike/expiration) to narrow
- If Theta Terminal is not running, tools return a clear connection error
- **Greeks are NOT available** on the free tier — use DoltHub or local cache (see below) for Greeks

---

## execute_options_backtest — Full Reference

Backtest **real options strategies** using ThetaData EOD option prices + Greeks combined with Alpaca underlying data. Requires Theta Terminal v3 running locally.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | `str` | required | Underlying symbol (e.g., `"SPY"`, `"AAPL"`) |
| `strategy_code` | `str` | required | Python code that defines `pf` (a VectorBT Portfolio) |
| `start` | `str` | required | Start date `"YYYY-MM-DD"` |
| `end` | `str` | `""` (today) | End date `"YYYY-MM-DD"` |
| `right` | `str` | `"both"` | `"call"`, `"put"`, or `"both"` |
| `max_dte` | `int` | `60` | Max days-to-expiration to include |
| `strike_count` | `int` | `15` | Number of strikes above/below ATM to fetch |

### Sandbox Variables

| Variable | Type | Description |
|----------|------|-------------|
| `df` | `pd.DataFrame` | Underlying OHLCV (Open/High/Low/Close/Volume), DatetimeIndex |
| `chain` | `pd.DataFrame` | Flat DataFrame of ALL option records (see columns below) |
| `symbol` | `str` | The underlying symbol |
| `vbt` | vectorbt | VectorBT library (with Portfolio proxy) |
| `pd`, `np` | libraries | pandas, numpy |
| `sm`, `statsmodels`, `scipy`, `scipy_stats`, `math`, `itertools` | libraries | Same as equity sandbox |
| `yf_download` | function | Fetch Yahoo Finance data inline: `yf_download('^VIX', start='2025-01-01')` → DataFrame |
| `OptionsBook` | factory | Position tracker: `book = OptionsBook(init_cash=100000)` — see below |

### `OptionsBook` — Portfolio Position Tracker

Instead of manually tracking positions with dicts and flat return series, use `OptionsBook` for proper multi-position tracking with cash accounting, mark-to-market, and portfolio Greeks.

```python
book = OptionsBook(init_cash=100000)  # optional: contract_size=100
```

#### Methods

| Method | Description |
|--------|-------------|
| `book.update(dt)` | **Call every date** — marks all positions to market, records equity snapshot |
| `book.open(dt, exp, strike, right, qty=1, price=None)` | Open position. `qty>0` = buy, `qty<0` = sell. Returns position dict or None |
| `book.close(pos, dt, price=None)` | Close a position. Returns realized P&L ($) |
| `book.close_all(dt)` | Close all open positions |
| `book.close_expired(dt)` | Close positions where expiration ≤ dt (at price=0) |
| `book.find(expiration=, strike=, right=)` | Find open positions matching criteria |
| `book.to_portfolio()` | Convert equity curve → `vbt.Portfolio` (assign to `pf`) |

#### Properties

| Property | Description |
|----------|-------------|
| `book.cash` | Current cash balance |
| `book.equity` | Total equity (cash + positions) |
| `book.open_positions` | List of open position dicts |
| `book.num_positions` | Number of open positions |
| `book.greeks` | Aggregate portfolio `{delta, gamma, theta, vega}` |
| `book.summary` | Stats dict (return %, win rate, avg P&L, etc.) |
| `book.trade_log` | DataFrame of all closed trades |

#### Position dict keys

`id`, `entry_date`, `expiration`, `strike`, `right`, `qty`, `entry_price`, `mark`, `market_value`, `dte`, `delta`, `gamma`, `theta`, `vega`, `rho`, `iv`

#### Fill logic

- **Opening**: buys fill at ask, sells fill at bid (realistic slippage)
- **Closing**: longs close at bid, shorts close at ask
- Override with explicit `price=` parameter

### `chain` DataFrame columns

`date`, `expiration`, `strike`, `right` (C/P), `open`, `high`, `low`, `close`, `volume`, `bid`, `ask`, `delta`, `gamma`, `theta`, `vega`, `rho`, `iv`, `dte`, `underlying_close`

- `date` and `expiration` are `pd.Timestamp`
- `strike` is in dollars (auto-normalized from ThetaData thousandths)
- `dte` is computed as `(expiration - date).days`

### Helper Functions

| Function | Description |
|----------|-------------|
| `get_chain_on_date(date, right=None, min_dte=None, max_dte=None)` | Option chain snapshot for a specific date |
| `nearest_expiry(date, min_dte=20, max_dte=45)` | Find nearest expiration with DTE in range → `Timestamp` or `None` |
| `get_atm(date, expiration, right='C')` | Get ATM contract → `Series` or `None` |
| `get_contract(date, expiration, strike, right='C')` | Get specific contract row → `Series` or `None` |
| `get_contract_series(expiration, strike, right='C', as_of=None)` | Time series for one contract → `DataFrame`. **Pass `as_of=dt` to avoid look-ahead bias.** |

### Look-ahead bias protection

- **`vbt.Portfolio.from_signals`** — signals are auto-lagged by 1 bar (same as equity backtester). Trades execute on the next bar.
- **`get_contract(dt, ...)` / `get_atm(dt, ...)` / `get_chain_on_date(dt)`** — only return data for the requested date. No look-ahead.
- **`OptionsBook`** — all methods (`open`, `close`, `update`) use `get_contract(dt)` internally. Buys fill at ask, sells at bid. No look-ahead.
- **`get_contract_series()`** — pass `as_of=dt` to only see data up to that date. Without it, returns full history (future included).
- **`chain` and `df`** — raw DataFrames contain all dates. In manual loops, only use `chain[chain['date'] == dt]` or helpers to avoid peeking ahead.

### Strategy Examples

#### 1. Sell OTM put (~5% below ATM), roll at 21 DTE

```
symbol: "SPY"
start: "2025-06-01"
end: "2026-02-01"
right: "put"
strategy_code: |
  dates = sorted(chain['date'].unique())
  returns = pd.Series(0.0, index=dates)
  position = None

  for i, dt in enumerate(dates):
      if position is not None:
          cur = get_contract(dt, position['exp'], position['strike'], 'P')
          if cur is not None:
              daily_pnl = position['last_price'] - cur['close']
              returns.loc[dt] = daily_pnl / 10000.0
              position['last_price'] = cur['close']
              if cur['dte'] <= 21 or cur['close'] > position['entry'] * 1.5:
                  position = None
          else:
              position = None

      if position is None:
          exp = nearest_expiry(dt, min_dte=30, max_dte=50)
          if exp is not None:
              atm = get_atm(dt, exp, 'P')
              if atm is not None:
                  # Pick strike ~5% below ATM (≈ 30 delta equivalent)
                  target_strike = round(atm['strike'] * 0.95)
                  contract = get_contract(dt, exp, target_strike, 'P')
                  if contract is None:
                      # Find nearest available strike
                      snap = get_chain_on_date(dt, right='P')
                      snap = snap[snap['expiration'] == exp]
                      if not snap.empty:
                          idx = (snap['strike'] - target_strike).abs().idxmin()
                          contract = snap.loc[idx]
                  if contract is not None:
                      position = {'exp': exp, 'strike': contract['strike'],
                                  'entry': contract['close'], 'last_price': contract['close']}

  pf = vbt.Portfolio.from_returns(returns, init_cash=10000, freq='1D')
```

#### 2. Buy ATM straddle, hold 1 day

```
symbol: "AAPL"
start: "2025-06-01"
end: "2026-01-15"
right: "both"
strategy_code: |
  dates = sorted(chain['date'].unique())
  returns = []

  for i, dt in enumerate(dates[:-1]):
      exp = nearest_expiry(dt, min_dte=25, max_dte=40)
      if exp is None:
          returns.append(0.0)
          continue
      call = get_atm(dt, exp, 'C')
      put = get_atm(dt, exp, 'P')
      if call is None or put is None:
          returns.append(0.0)
          continue

      # Next day P&L
      next_dt = dates[i + 1]
      call_next = get_contract(next_dt, exp, call['strike'], 'C')
      put_next = get_contract(next_dt, exp, put['strike'], 'P')
      if call_next is None or put_next is None:
          returns.append(0.0)
          continue
      cost = call['close'] + put['close']
      value = call_next['close'] + put_next['close']
      returns.append((value - cost) / cost if cost > 0 else 0.0)

  pf = vbt.Portfolio.from_returns(pd.Series(returns), init_cash=10000, freq='1D')
```

#### 3. Covered call (underlying + short OTM call)

```
symbol: "SPY"
start: "2025-06-01"
end: "2026-02-01"
right: "call"
strategy_code: |
  dates = sorted(chain['date'].unique())
  close = df['Close']
  underlying_ret = close.pct_change().fillna(0)

  call_pnl = pd.Series(0.0, index=close.index)
  position = None

  for dt in dates:
      if position is not None:
          cur = get_contract(dt, position['exp'], position['strike'], 'C')
          if cur is None or cur['dte'] <= 7:
              position = None
          else:
              call_pnl.loc[dt] = position['entry'] - cur['close']

      if position is None:
          exp = nearest_expiry(dt, min_dte=25, max_dte=40)
          if exp:
              atm = get_atm(dt, exp, 'C')
              if atm is not None:
                  # Pick strike ~3% above ATM
                  target_strike = round(atm['strike'] * 1.03)
                  contract = get_contract(dt, exp, target_strike, 'C')
                  if contract is None:
                      snap = get_chain_on_date(dt, right='C')
                      snap = snap[snap['expiration'] == exp]
                      if not snap.empty:
                          idx = (snap['strike'] - target_strike).abs().idxmin()
                          contract = snap.loc[idx]
                  if contract is not None:
                      position = {'exp': exp, 'strike': contract['strike'], 'entry': contract['close']}

  total_ret = underlying_ret + call_pnl / (close.shift(1).fillna(close.iloc[0]))
  pf = vbt.Portfolio.from_returns(total_ret.dropna(), init_cash=10000, freq='1D')
```

#### 4. Sell OTM put with OptionsBook (recommended pattern)

```
symbol: "SPY"
start: "2025-06-01"
end: "2026-02-01"
right: "put"
strategy_code: |
  book = OptionsBook(init_cash=100000)
  dates = sorted(chain['date'].unique())

  for dt in dates:
      book.update(dt)
      book.close_expired(dt)

      # Roll at 21 DTE or if loss > 50%
      for pos in book.open_positions:
          if pos['dte'] is not None and pos['dte'] <= 21:
              book.close(pos, dt)
          elif pos['mark'] > pos['entry_price'] * 1.5:
              book.close(pos, dt)

      # Open new position if flat
      if book.num_positions == 0:
          exp = nearest_expiry(dt, min_dte=30, max_dte=50)
          if exp:
              atm = get_atm(dt, exp, 'P')
              if atm:
                  target = round(atm['strike'] * 0.95)
                  book.open(dt, exp, target, 'P', qty=-1)

  pf = book.to_portfolio()
```

#### 5. Iron condor with OptionsBook

```
symbol: "SPY"
start: "2025-06-01"
end: "2026-02-01"
right: "both"
strategy_code: |
  book = OptionsBook(init_cash=100000)
  dates = sorted(chain['date'].unique())

  for dt in dates:
      book.update(dt)
      book.close_expired(dt)

      # Close all legs at 21 DTE
      for pos in book.open_positions:
          if pos['dte'] is not None and pos['dte'] <= 21:
              book.close(pos, dt)

      if book.num_positions == 0:
          exp = nearest_expiry(dt, min_dte=30, max_dte=50)
          if exp:
              atm = get_atm(dt, exp, 'C')
              if atm:
                  s = atm['strike']
                  # Short strangle wings
                  book.open(dt, exp, round(s * 1.03), 'C', qty=-1)  # sell OTM call
                  book.open(dt, exp, round(s * 0.97), 'P', qty=-1)  # sell OTM put
                  # Long protection wings
                  book.open(dt, exp, round(s * 1.06), 'C', qty=1)   # buy far OTM call
                  book.open(dt, exp, round(s * 0.94), 'P', qty=1)   # buy far OTM put

  pf = book.to_portfolio()
```

### Local options data cache (with Greeks)

The download script fetches from **DoltHub** (free) and includes full Greeks. The backtester loads from cache first for instant offline backtesting — no terminal or API key needed.

| Detail | Value |
|--------|-------|
| Cache location | `data/options_cache/{SYMBOL}_eod.parquet` |
| Data source | DoltHub (free, no terminal needed) |
| Columns (18) | `date`, `expiration`, `strike`, `right`, `open`, `high`, `low`, `close`, `bid`, `ask`, `volume`, `dte`, `iv`, `delta`, `gamma`, `theta`, `vega`, `rho` |
| Date format | YYYYMMDD int (e.g., `20250601`) |
| Greeks null % | **0%** — all rows have delta, gamma, theta, vega, rho, iv populated |

### Pre-built cache

| Symbol | Date range | Rows | File size | Expirations |
|--------|-----------|------|-----------|-------------|
| **SPY** | 2024-01-01 → 2026-02-15 | 61,934 | 1.9 MB | ~3 short-term per date |

To add more symbols, run:
```bash
python download_options_data.py --symbol AAPL --start 2024-01-01
python download_options_data.py --symbol QQQ --start 2024-01-01 --max-dte 60
```
No external terminal needed — downloads directly from DoltHub API. The script uses daily queries with call/put split to stay within API limits. Resumes automatically — skips dates already cached. If an existing cache lacks Greeks, it re-downloads everything.

**DoltHub symbol coverage:** S&P 500 components + SPY + SPDR sector ETFs (XLF, XLE, XLK, etc.), 2019–present, ~3 short-term expirations per date (~2wk, ~4wk, ~8wk).

### Data source priority

1. **Local parquet cache** — instant, includes Greeks if downloaded from DoltHub
2. **DoltHub API** — free fallback, Greeks included, S&P 500 only
3. **ThetaData free-tier EOD** — all symbols, but no Greeks (needs Theta Terminal running)

### Common mistakes (options backtest)

1. **`start` is required** — unlike `execute_vectorbt_strategy`, you must specify a start date.
2. **No terminal needed if cache exists** — the tool reads from local parquet first. Only needed for uncached symbols/dates not on DoltHub.
3. **Helpers return `None` if no data** — always check for `None` before accessing fields.
4. **`chain` dates are `pd.Timestamp`** — compare with `pd.Timestamp('2025-06-01')`, not strings.
5. **`right` in chain is `'C'` or `'P'`** — uppercase single character.
6. **No import statements needed** — everything is pre-loaded.
7. **Strike may not match exactly** — use `get_chain_on_date` + find nearest strike when a computed strike doesn't exist.
8. **DoltHub has ~3 expirations per date** — no LEAPs or far-dated monthlies. Use `nearest_expiry()` to find what's available.