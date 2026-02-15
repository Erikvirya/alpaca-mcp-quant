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
| `limit` | `int` | `1000` | Max bars to fetch (used when start/end not fully specified). |

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
| `get_theta_option_greeks_eod(symbol, expiration, strike, right, start_date, end_date)` | Historical EOD + all Greeks (delta, gamma, theta, vega, IV, etc.) |

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
- "Get SPY 550 call EOD with Greeks from 2026-01-01 to 2026-02-13"
- "Show TSLA put option chain EOD, all strikes, expiring 2026-03-21, last 5 trading days"

### Notes

- EOD data is **FREE** — no paid subscription needed
- Data is generated at **17:15 ET** each day
- Results capped at **500 records** per call — use filters (`max_dte`, `num_strikes`, specific strike/expiration) to narrow
- If Theta Terminal is not running, tools return a clear connection error

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
| `get_contract_series(expiration, strike, right='C')` | Full time series for one contract → `DataFrame` |

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

### Local options data cache

The tool checks for a local parquet cache before hitting the ThetaData API. This allows offline backtesting and avoids free-tier rate limits.

| Detail | Value |
|--------|-------|
| Cache location | `data/options_cache/{SYMBOL}_eod.parquet` |
| SPY cached range | 2024-02-01 → 2026-02-13 (2.6M rows, 36 MB) |
| Columns | `date`, `expiration`, `strike`, `right`, `open`, `high`, `low`, `close`, `volume`, `bid`, `ask` |
| Date format | YYYYMMDD int (e.g., `20250601`) |

**How it works:** If the cache file exists and covers the requested date range, the tool loads from disk instantly. Otherwise it falls back to the ThetaData API (requires Theta Terminal running).

**To download or update the cache:**
```bash
python download_options_data.py --symbol SPY --start 2024-02-01
python download_options_data.py --symbol AAPL --start 2025-01-01 --max-dte 90
```
The script resumes automatically — it skips months already in the cache. Requires Theta Terminal running during download.

### Free tier vs paid tier

- **Free tier** — EOD prices (OHLCV, bid/ask, volume). All helpers except delta-based selection work.
- **Standard tier ($30/mo)** — adds Greeks (delta, gamma, theta, vega, rho, IV). If you upgrade, the tool auto-detects and includes Greeks columns in `chain`.
- **No delta column on free tier** — select contracts by strike price relative to ATM instead (e.g., `atm['strike'] * 0.95` for ~5% OTM put).

### Common mistakes (options backtest)

1. **`start` is required** — unlike `execute_vectorbt_strategy`, you must specify a start date.
2. **Theta Terminal not needed if cache exists** — the tool reads from local parquet first. Only needed for uncached symbols/dates.
3. **Helpers return `None` if no data** — always check for `None` before accessing fields.
4. **`chain` dates are `pd.Timestamp`** — compare with `pd.Timestamp('2025-06-01')`, not strings.
5. **`right` in chain is `'C'` or `'P'`** — uppercase single character.
6. **No import statements needed** — everything is pre-loaded.
7. **Strike may not match exactly** — use `get_chain_on_date` + find nearest strike when a computed strike doesn't exist.