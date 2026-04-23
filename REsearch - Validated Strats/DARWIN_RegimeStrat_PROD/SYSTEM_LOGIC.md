# DARWIN RegimeStrat PROD — Full System Logic

> NQ Futures Daily Regime-Switching Strategy on Darwinex Zero MT5  
> Last updated: Feb 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Structure](#2-file-structure)
3. [Daily Execution Flow](#3-daily-execution-flow)
4. [Strategy Logic](#4-strategy-logic)
5. [Position Sizing](#5-position-sizing)
6. [Contract Roll Logic](#6-contract-roll-logic)
7. [Order Execution](#7-order-execution)
8. [Timezone & DST Handling](#8-timezone--dst-handling)
9. [VXN Data Waterfall](#9-vxn-data-waterfall)
10. [Risk Guardrails](#10-risk-guardrails)
11. [Slippage Decomposition](#11-slippage-decomposition)
12. [Logging & Notifications](#12-logging--notifications)
13. [Configuration](#13-configuration)
14. [Key Constants Reference](#14-key-constants-reference)

---

## 1. System Overview

A fully automated daily trading system for NQ (Nasdaq 100) futures running on the Darwinex Zero MT5 platform. The strategy generates a single allocation signal per day using a regime-switching model, converts it to a discrete contract count, and executes via passive limit orders designed to minimise slippage.

**Execution trigger:** Windows Task Scheduler fires `execute_trade.py` at **21:02 UTC Mon–Fri**.

**Instruments:** NQ quarterly futures on Darwinex Zero MT5 (`NQ_H`, `NQ_M`, `NQ_U`, `NQ_Z`).

**Data sources:**
- Price (OHLCV): MT5 terminal D1 bars (CME feed via Darwinex)
- Volatility index: VXN (Nasdaq Volatility Index) — CBOE → yfinance → local cache

---

## 2. File Structure

```
DARWIN_RegimeStrat_PROD/
├── execute_trade.py        # Main daily execution script (Task Scheduler entry point)
├── strategy_logic.py       # Regime detection + allocation signal generation
├── mt5_data.py             # MT5 adapter: data fetch, order placement, position queries
├── contract_roll.py        # Quarterly contract roll logic and scheduling
├── vxn_data.py             # VXN data fetch with CBOE/yfinance/cache waterfall
├── futures_config.py       # NQ contract spec, commission constant
├── slippage.py             # Slippage decomposition utilities
├── notify.py               # Email notification formatting and sending
├── logging_config.py       # Structured logging setup
├── watchdog.py             # Process watchdog / heartbeat
├── symbol_validator.py     # MT5 symbol availability checks
├── config.env              # Credentials (MT5 login, email SMTP — NOT committed)
├── requirements_vps.txt    # Python dependencies
└── TESTING/                # Sandbox and test scripts (not deployed to prod)
    ├── sandbox_test.py              # Full pipeline dry-run (yfinance data)
    ├── test_sandbox_execution.py    # Smart order execution unit tests (14 scenarios)
    └── test_sandbox_rollover.py     # Contract roll unit tests (8 scenarios)
```

---

## 3. Daily Execution Flow

```
21:02 UTC (Task Scheduler)
│
├─ 1. GUARD CHECKS
│     ├─ Weekend check (skip Sat/Sun)
│     ├─ Already-traded-today check (trade_log.csv guard)
│     └─ MT5 connect → fail → email alert → exit
│
├─ 2. CONTRACT ROLL CHECK
│     ├─ Get held symbol + quantity from MT5 positions
│     ├─ Compare to front-month symbol (get_front_symbol)
│     ├─ If roll needed:
│     │     ├─ days_left <= 1  → emergency_roll (direct market orders)
│     │     ├─ days_left > 1   → smart_roll (passive limit orders)
│     │     │     └─ on failure: alert email, pin to held symbol, retry tomorrow
│     │     └─ On success: update current_symbol to new contract
│     └─ Pin nq_symbol to held contract if roll incomplete (prevents double exposure)
│
├─ 3. WAIT FOR D1 BAR (DST-aware)
│     ├─ Compute CME close time in UTC via ZoneInfo("America/Chicago")
│     │     Summer (CDT): 16:00 CT = 21:00 UTC
│     │     Winter (CST): 16:00 CT = 22:00 UTC
│     ├─ Sleep until cme_close_utc + 2 min (up to ~60 min wait in winter)
│     └─ Poll MT5 every 30s for up to 30 min
│           Bar ready if: bar_timestamp_utc >= cme_close_utc - 25h
│           (robust to Darwinex EET broker timezone shifting bar dates)
│
├─ 4. FETCH VXN
│     └─ CBOE CSV → yfinance fallback → local cache (max 3 bday stale)
│
├─ 5. GENERATE SIGNAL
│     ├─ get_strategy_signals(close, high, low, vxn)
│     └─ Returns target allocation (-0.2 to 2.0)
│
├─ 6. SIZE POSITION
│     ├─ target_contracts = round(equity * alloc / (price * point_value))
│     ├─ Clamp to max position cap
│     └─ NaN allocation guard → hold current position + alert
│
├─ 7. RECONCILE WITH MT5
│     ├─ actual_position = mt5.positions_get()  (MT5 = source of truth)
│     └─ delta = target_contracts - actual_position
│
├─ 8. PLACE ORDER (if delta != 0)
│     ├─ Margin check (only if increasing exposure)
│     ├─ Determine order mode:
│     │     crisis_mode    = (regime == "CRISIS")   → direct emergency order
│     │     risk_reducing  = crisis OR reducing size → smart + emergency fallback
│     │     risk_on        = increasing size, normal → smart only
│     ├─ place_market_order() with up to 3 retries (60s delay each)
│     └─ On permanent failure: email alert, return (no trade_log entry)
│
├─ 9. LOG + NOTIFY
│     ├─ trade_log.csv append (fill price, slippage, regime, contracts, order_type)
│     └─ Email notification with full fill details
│
└─ 10. SHUTDOWN
      └─ mt5.shutdown()
```

---

## 4. Strategy Logic

**File:** `strategy_logic.py` → `get_strategy_signals(close, high, low, vix_series)`

Returns a `pd.Series` of target allocations from `-0.2` to `2.0`.

### Indicators (all pure pandas/numpy)

| Indicator | Parameters | Usage |
|---|---|---|
| EMA-50 | span=50 | Trend filter |
| EMA-20 | span=20 | Recovery re-entry |
| RSI | 14, Wilder smoothing | Momentum filter |
| ADX | 14 | Trend strength |
| Bollinger Bands | 20-period, 2.0 std | Range mean-reversion |
| Realised Vol | 20-day rolling std × √252 | Vol targeting scalar |
| ATR | 14, EMA | Graduated entry ramp |

### Regime Definitions

| Regime | Conditions | Base Allocation |
|---|---|---|
| **CRISIS** | VXN > 35 | Crisis scalar applied (near-zero) |
| **RECOVERY** | Close > EMA20, RSI > 50, VXN < 30, not crisis | `vol_scalar` clipped ≥ 0.8 |
| **RANGE** | ADX < 25, VXN < 25, not trend/recovery/crisis | Mean-reversion: 0.5 / -0.2 / 0.1 |
| **TREND** | Close > EMA50, not crisis | `vol_scalar` (capped 0.5–2.0) |
| **DEFAULT** | Everything else | Graduated 0.2–0.4 based on ATR distance |

### Regime Priority (applied in order)

1. **Default graduated base** — smooth ramp 0.2→0.4 as price approaches EMA50
2. **Recovery** — overrides default with vol-targeted floor ≥ 0.8
3. **Range** — bidirectional mean-reversion overlay
4. **Trend graduated entry** — smooth ramp 0.4→vol_scalar within 1 ATR above EMA50
5. **Full trend** — vol_scalar (beyond 1 ATR above EMA50)
6. **RSI momentum fade** — reduce 30% when RSI drops >15pts from 14-day peak (early exit)
7. **Crisis overlay** (applied last) — smooth scalar from 1.0 (VXN≤25) → 0.05 (VXN=50)

### Vol Targeting

```
vol_scalar = 0.16 / realised_vol_20d
vol_scalar = clip(0.5, 2.0)    # leverage cap 2x
```

Target annualised volatility: **16%**. If realised vol is 8%, scalar = 2.0 (max leverage). If vol is 32%, scalar = 0.5 (half position).

### Range Regime Allocation

```
close < lower_BB  →  +0.5  (long BB dip)
close > upper_BB  →  -0.2  (short BB rip)
otherwise         →  +0.1  (near-flat)
```

---

## 5. Position Sizing

**File:** `execute_trade.py` → `notional_to_contracts()`

```python
target_notional = equity * abs(alloc)
contract_notional = price * NQ.point_value      # price × $20
n = round(target_notional / contract_notional)
```

- `NQ.point_value = 20` ($20 per NQ point)
- Sign from allocation direction (positive = long, negative = short)
- Result clamped to max position cap

**Example:** $500k equity, alloc=0.8, NQ at 21,000:
```
target_notional = 500,000 × 0.8 = $400,000
contract_notional = 21,000 × 20 = $420,000
n = round(400,000 / 420,000) = round(0.952) = 1 contract
```

---

## 6. Contract Roll Logic

**File:** `contract_roll.py`

### Symbol Naming
Darwinex Zero format: `NQ_H` (Mar), `NQ_M` (Jun), `NQ_U` (Sep), `NQ_Z` (Dec)

### Roll Schedule

| Date | Definition |
|---|---|
| **Expiry** | 3rd Friday of expiry month |
| **Close-only** | Expiry − 3 calendar days |
| **Roll-by** | Close-only − 5 business days |

`get_front_symbol()` returns the next contract once `today >= roll_by`.

### Roll Decision Logic (`execute_trade.py`)

```
1. Get held symbol from MT5 positions (source of truth)
2. Compare to get_front_symbol(today)
3. If held != front:
   a. Verify new symbol is available and has quotes
   b. days_left = days until close-only date
   c. days_left <= 1  → emergency_roll (market orders, no spread check)
   d. days_left > 1   → smart_roll (passive limit orders)
      - Close old: place_market_order(old, risk_reducing=True)
      - Open new:  place_market_order(new, risk_reducing=True)
      - On failure: send alert, pin nq_symbol to old contract, retry next day
4. If roll incomplete, pin nq_symbol to held contract
   (prevents buying front-month while still holding old contract)
```

### Roll Order Routing

| Scenario | Order Type | Fallback |
|---|---|---|
| Normal roll day (`days_left > 1`) | Smart limit (passive) | Alert + retry tomorrow |
| Emergency roll (`days_left <= 1`) | Direct market order | None (must fill) |
| Roll open leg fails after close leg | Emergency market (risk_reducing=True) | — |

---

## 7. Order Execution

**File:** `mt5_data.py`

### Entry Point: `place_market_order()`

```python
def place_market_order(symbol, volume, direction,
                       risk_reducing=False, crisis_mode=False):
```

| Flag | Behaviour |
|---|---|
| `crisis_mode=True` | Skip smart entirely → direct emergency market order |
| `risk_reducing=True` | Smart order first; if fails → emergency fallback |
| Both False | Smart order only; failure raises exception |

### Smart Limit Order: `place_smart_order()`

Passive GTC resting limit order strategy designed to capture mid-price or better.

#### Spread Gate

Before placing any order, check current spread:
```
if spread > MAX_SPREAD_PTS (0.75):
    wait up to 30s for spread to narrow (6 × 5s checks)
    if still wide → raise RuntimeError("Spread too wide")
```

#### Order Placement Loop

```
attempt=1:
  - Record initial_mid = (bid + ask) / 2
  - Place GTC BUY_LIMIT at mid (or SELL_LIMIT at mid)
  - Wait PASSIVE_WAIT_S = 3.0s for passive fill (market ticks to us)
  - If filled → return fill

attempt=2..9 (WALKBACK_RETRIES=8):
  - Cancel previous unfilled GTC order (TRADE_ACTION_REMOVE)
  - Step limit_price by WALKBACK_STEP_PTS = 0.25 toward aggressor:
      BUY:  limit += 0.25 (closer to ask)
      SELL: limit -= 0.25 (closer to bid)
  - Check for immediate fill (limit >= ask for BUY, limit <= bid for SELL)
  - If immediate fill → place order at new limit, confirm fill
  - If not immediate → wait WALKBACK_WAIT_S = 1.0s for passive fill

After all attempts fail → raise RuntimeError
```

#### Typical Fill Scenarios

| Market Condition | Expected Outcome | Attempts |
|---|---|---|
| Normal spread (0.25pt), passive market | Fill at mid | 1 |
| Normal spread, market doesn't tick to us | Fill at mid+0.25 (ask) | 2 |
| Spread = 0.5pt | Fill at mid+0.25 | 2 |
| Spread = 0.75pt | Fill at mid+0.50 | 3 |
| Spread stays > 0.75pt | Fail → emergency fallback (if risk_reducing) | — |

#### Fill Detection (`_poll_order_fill`)

Polls `mt5.orders_get(ticket=order_id)` every 0.25s. When order leaves pending queue, checks `mt5.history_deals_get()` with:
- **Backward window: 1 hour** (handles Darwinex EET broker timezone, UTC+2/+3)
- **Forward window: 30 min**
- Filters by `order_id` to avoid false matches

#### Cancel Between Steps

Each walkback step uses `TRADE_ACTION_REMOVE` to cancel the previous GTC order before placing a new one at the stepped price. This avoids:
- Dual outstanding orders
- TRADE_ACTION_MODIFY race condition (order could fill between poll and modify)

### Emergency Order: `place_emergency_order()`

Direct `TRADE_ACTION_DEAL` with `deviation=50` (50 NQ points ≈ uncapped). No spread check. Guaranteed fill at any price.

Used when:
- `crisis_mode=True` (VXN > 35, need immediate execution)
- Smart order fails with `risk_reducing=True` (reducing position, can't afford to miss fill)
- Emergency contract roll (`days_left <= 1`)

---

## 8. Timezone & DST Handling

Three independent clocks involved:

| Clock | Timezone | Usage |
|---|---|---|
| **CME** | `America/Chicago` (CDT/CST) | Defines when D1 bar closes |
| **Server** | UTC | Script timing, trade log dates |
| **Broker** | EET (UTC+2 winter / UTC+3 summer) | MT5 bar timestamps |

### CME Close → UTC

```python
chicago_now = datetime.now(ZoneInfo("America/Chicago"))
cme_close_today = datetime.combine(chicago_now.date(), time(16, 0),
                                   tzinfo=ZoneInfo("America/Chicago"))
cme_close_utc = cme_close_today.astimezone(ZoneInfo("UTC"))
```

| Season | CT offset | CME close UTC | Task Scheduler wait |
|---|---|---|---|
| Summer (CDT) | UTC-5 | 21:00 UTC | ~0 min (fires 21:02) |
| Winter (CST) | UTC-6 | 22:00 UTC | ~60 min (fires 21:02) |

US and UK DST transitions occur on different dates (US: 2nd Sun Mar / 1st Sun Nov; UK: last Sun Mar / last Sun Oct). Our code uses `America/Chicago` directly — never London time — so the ~1–2 week mismatch windows have no impact.

### Bar Freshness Check

Darwinex MT5 bar timestamps are broker-local (EET) open times stored as Unix timestamps. When converted to UTC pandas Timestamps, the bar for "Thursday" lands on **Wednesday** in UTC:

```
Thursday 00:00 EET (UTC+2) = Wednesday 22:00 UTC → .date() = Wednesday
```

**Old check (broken):**
```python
if latest_bar_date >= today.date():   # Wednesday >= Thursday → always FAIL
```

**New check (fixed):**
```python
today_session_open_utc = (cme_close_utc - timedelta(hours=25)).replace(tzinfo=None)
if latest_bar_ts >= today_session_open_utc:   # 22:00 UTC >= 21:00 UTC → PASS
```

---

## 9. VXN Data Waterfall

**File:** `vxn_data.py`

```
1. CBOE direct CSV download (primary)
   https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv
   → save to local cache on success

2. yfinance ("^VXN") fallback
   → save to local cache on success

3. Local cache (vxn_cache.csv)
   → accepted if cache age <= 3 business days
   → rejected (RuntimeError) if stale beyond limit
```

Cache location: `./data/vxn_cache.csv` (relative to script directory).

The 3-day stale limit allows operation through long weekends (e.g., Monday holiday).

---

## 10. Risk Guardrails

### Pre-Trade Guards

| Guard | Condition | Action |
|---|---|---|
| Weekend | `today.weekday() >= 5` | Skip, exit cleanly |
| Already traded | Date found in `trade_log.csv` | Skip, exit cleanly |
| MT5 connect fail | `mt5.initialize()` returns False | Email alert, exit |
| Stale D1 bar | Bar not ready after 30 min poll | Email alert, exit |
| VXN all sources fail | All 3 waterfall sources fail | RuntimeError → email + exit |
| NaN allocation | `alloc = NaN` | Hold position, send alert, exit |

### Order-Time Guards

| Guard | Condition | Action |
|---|---|---|
| Wide spread | Spread > 0.75pt after 30s patience | Abort smart order |
| Insufficient margin | `free_margin < NQ_MARGIN × volume` | Abort, send alert |
| Smart order fail (risk-on) | All walkback attempts exhausted | Raise → retry up to 3× |
| Smart order fail (risk-off) | All walkback attempts exhausted | Emergency fallback |

### Margin Check

```python
NQ_MARGIN = 16_500  # $ per contract (Darwinex Zero confirmed)
required_margin = NQ_MARGIN * volume
if free_margin < required_margin:
    → abort order, send alert
```

Only applied when **increasing** exposure (not when reducing).

### Order Retry Logic

3 attempts with 60-second delay between each. On 3rd failure:
- Email alert sent
- Trade NOT logged (so next day's run starts fresh)

---

## 11. Slippage Decomposition

**File:** `slippage.py`

Every fill is decomposed into two components:

```
slippage_total = fill_price - signal_close_price  (signed by direction)

slippage_spread   = fill_price - initial_mid       (cost of crossing spread)
slippage_walkback = initial_mid - signal_close     (mid drift during order wait)

slippage_total = slippage_spread + slippage_walkback
```

| Component | Best case | Typical | Worst case |
|---|---|---|---|
| spread | 0 pts (passive fill at mid) | +0.12 pts | +0.75 pts (full spread) |
| walkback | 0 pts (filled on attempt 1) | 0 pts | +2.0 pts (8 steps × 0.25) |

Logged to `trade_log.csv` in both points and basis points.

---

## 12. Logging & Notifications

### Structured Logging

`logging_config.py` sets up:
- Console output (INFO level)
- File output to `./logs/` directory (DEBUG level)
- Format: `YYYY-MM-DD HH:MM:SS | module | LEVEL | message`

Key log events:
```
mt5_connected=True login=... balance=... equity=...
roll_needed=True old_symbol=... new_symbol=... days_to_close_only=...
cme_close_chicago=16:00 CDT cme_close_utc=21:00 UTC wait_needed_s=0
d1_bar_ready=True latest_bar_ts=... waited=30s
regime=RECOVERY alloc=0.84 target_cts=2 actual_cts=0 delta=+2
order_mode=RISK_ON ...
smart_fill_passive fill=21000.12 limit=21000.12 attempt=1
EMERGENCY_FILL fill_price=... spread=...
trade_complete fill=21000.25 slippage_total=+0.12pts
```

### trade_log.csv Fields

```
date, time, regime, target_alloc, target_contracts, prev_contracts,
delta_contracts, signal_close_price, initial_mid, fill_price,
slippage_total_pts, slippage_total_bps, slippage_spread_pts,
slippage_walkback_pts, spread_at_order, fill_attempts, equity,
vxn_level, commission, contract_symbol, order_type
```

### Email Notifications

Sent via `notify.py` (SMTP config in `config.env`):

| Event | Subject |
|---|---|
| Trade executed | `DARWIN Trade: BUY 2 NQ_H @ 21000.25` |
| No trade (delta=0) | `DARWIN Heartbeat: No trade today` |
| Allocation NaN | `DARWIN ALERT: NaN allocation` |
| Stale D1 bar | `DARWIN ALERT: STALE D1 BAR` |
| Order failed (all retries) | `DARWIN ALERT: Order failed` |
| Roll failed | `DARWIN ALERT: Roll failed — retry tomorrow` |
| Emergency roll | `DARWIN WARNING: Emergency roll executed` |

---

## 13. Configuration

**File:** `config.env` (not committed to version control)

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=Darwinex-Live
MT5_PATH=C:\Program Files\Darwinex MT5\terminal64.exe

EMAIL_SENDER=alerts@yourdomain.com
EMAIL_RECIPIENT=you@yourdomain.com
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_PASSWORD=your_app_password
```

### Task Scheduler Setup

- **Trigger:** Daily, Mon–Fri, 21:02 UTC
- **Action:** `python.exe execute_trade.py`
- **Working directory:** `DARWIN_RegimeStrat_PROD/`
- **Single instance:** Stop if already running

---

## 14. Key Constants Reference

| Constant | Value | Location | Description |
|---|---|---|---|
| `NQ_MARGIN` | $16,500 | `execute_trade.py` | Margin per NQ contract (Darwinex Zero) |
| `NQ.point_value` | $20 | `futures_config.py` | Dollar value per NQ point |
| `COMMISSION_PER_SIDE` | $4.00 | `futures_config.py` | Commission per contract per side |
| `MAX_SPREAD_PTS` | 0.75 | `mt5_data.py` | Max spread before order aborted |
| `PASSIVE_WAIT_S` | 3.0s | `mt5_data.py` | Initial wait at mid-price |
| `WALKBACK_STEP_PTS` | 0.25 | `mt5_data.py` | Price step per walkback attempt |
| `WALKBACK_WAIT_S` | 1.0s | `mt5_data.py` | Wait between walkback steps |
| `WALKBACK_RETRIES` | 8 | `mt5_data.py` | Max walkback attempts |
| `EMERGENCY_DEVIATION` | 50 pts | `mt5_data.py` | Uncapped deviation for emergency orders |
| `ROLL_BUFFER_BDAYS` | 5 | `contract_roll.py` | Business days before close-only to roll |
| `CLOSE_ONLY_OFFSET_DAYS` | 3 | `contract_roll.py` | Calendar days before expiry for close-only |
| `MAX_CACHE_STALE_DAYS` | 3 | `vxn_data.py` | Max VXN cache age in business days |
| `BAR_POLL_INTERVAL_S` | 30s | `execute_trade.py` | D1 bar poll frequency |
| `BAR_POLL_MAX_WAIT_S` | 1800s | `execute_trade.py` | Max poll budget after CME close |
| `MAX_RETRIES` | 3 | `execute_trade.py` | Order retry attempts |
| `RETRY_DELAY_S` | 60s | `execute_trade.py` | Delay between order retries |
| `CME_CLOSE_TIME` | 16:00 CT | `execute_trade.py` | CME NQ daily close (Chicago time) |
| `VOL_TARGET` | 16% annualised | `strategy_logic.py` | Realised vol target |
| `VOL_SCALAR_CAP` | 2.0x | `strategy_logic.py` | Maximum leverage from vol targeting |
| `CRISIS_VXN` | 35 | `strategy_logic.py` | VXN threshold for crisis regime |
| `RSI_FADE_THRESHOLD` | 15 pts | `strategy_logic.py` | RSI drop from peak triggering early exit |
| `RSI_FADE_REDUCTION` | 30% | `strategy_logic.py` | Allocation reduction on RSI fade |
