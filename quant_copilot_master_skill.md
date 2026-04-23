# Quant Copilot Master Skill: Institutional System Development & Execution

This document synthesizes months of quantitative research, backtesting, and live production deployment on platforms like Darwinex Zero, MetaTrader 5 (MT5), and various crypto/options exchanges. It serves as the master knowledge base for developing robust, institutional-grade algorithmic trading systems.

---

## 1. Backtesting Biases & Fallacies

### The "Slippage Artifact" (Death by a Thousand Cuts)
*   **The Trap:** Zero-fee retail backtests often show massive Sharpe ratios for high-frequency strategies (e.g., 1-2 day holding periods).
*   **The Reality:** When a realistic 2 basis points (0.02%) of slippage and commission is applied per trade, the *average expected value* of a high-frequency trade often drops below zero.
*   **The Fix:** Lengthen the prediction horizon (e.g., from 2 days to 10 days) and raise model conviction thresholds (e.g., `predict_proba > 0.60`). This reduces trade churn and increases the average profit per trade, making slippage mathematically irrelevant.

### The "Look-Ahead" Bias (Data Timing)
*   **The Trap:** Using daily aggregate data (like GDELT narrative volume) on the same day the trade executes. Daily data is usually published at midnight UTC, but trades might execute at 15:45 ET. The model is implicitly "peeking" into the future.
*   **The Fix:** Force a strict 1-day lag (`shift(1)`) on all external daily data features before feature engineering and inference. If the strategy's edge collapses, the edge was fake.

### The "Sequence Luck" Bias (Monte Carlo Reality)
*   **The Trap:** A strategy shows positive Total Return and Max Drawdown in a single Out-Of-Sample (OOS) walk-forward.
*   **The Fix:** You must run a **Monte Carlo Bootstrap** resampled on *daily portfolio returns* (not just trade sequences). If the 5th percentile (worst-case scenario) or the median of 1,000 shuffled return paths is deeply negative, your positive backtest was an artifact of "sequence luck." 

### The "Shorting" Fallacy in Structural Bull Markets
*   **The Trap:** Forcing a machine learning model to trade symmetrically (Long/Short) on major indices (S&P 500, Nasdaq).
*   **The Reality:** The market has a massive underlying positive drift. The cost of being wrong on a short is exponentially higher than being wrong on a long. 
*   **The Fix:** For macro narrative strategies, run **Long-Only** on indices. Use the model's bearish predictions to go to cash (FLAT) to preserve capital and minimize drawdowns, rather than actively shorting.

---

## 2. MetaTrader 5 (MT5) Execution Challenges

### Invalid Volume (Error `10014`)
*   **The Issue:** Attempting to send fractional lot sizes (e.g., `0.45`) for real exchange-traded futures contracts (CME Micro ES, NQ, etc.).
*   **The Fix:** Futures must be traded in exact whole integers. Use `max(1.0, round(calculated_lots))`.

### Invalid Stops (Error `10016`)
*   **The Issue:** A dynamic stop loss (like 2.5x ATR) calculates a price that is closer to the current market price than the broker's minimum allowable "Stops Level" (especially on volatile assets like JPY or Silver).
*   **The Fix:** Dynamically query `mt5.symbol_info(symbol).trade_stops_level`, convert it to price points, add a safety buffer, and clamp your calculated Stop Loss so it never breaches this minimum distance.

### Terminal Call Failed
*   **The Issue:** `mt5.copy_rates_from_pos()` or `mt5.symbol_info_tick()` fails silently or returns an error.
*   **The Fix:** MT5 requires the symbol to be active in the Market Watch. Always call `mt5.symbol_select(symbol, True)` and `time.sleep(0.1)` before querying history or ticks.

### Smart Limit Order Execution (Slippage Mitigation)
*   **The Issue:** Firing raw Market Orders (`ORDER_TYPE_BUY`) incurs maximum spread slippage.
*   **The Fix:** Implement a "Walkback" execution loop:
    1. Check if the spread is abnormally wide; if so, wait.
    2. Place a passive `GTC` Limit Order at the mid-price.
    3. Wait 3 seconds. If unfilled, cancel (`TRADE_ACTION_REMOVE`) and step the price 0.25 points closer to the aggressor side.
    4. Repeat until filled or a maximum number of retries is hit, then fallback to an emergency market order.

---

## 3. Institutional Risk Management & Darwinex Zero

### Institutional Position Sizing (True Risk Parity)
Never use fixed lot sizes or arbitrary volatility multipliers. Size positions based on **Account Equity** and **Contract Notional Value**:
1.  **Target Daily Risk:** `Account Equity * (Annual Vol Target / sqrt(252))`
2.  **Contract Multiplier:** `trade_tick_value / trade_tick_size`
3.  **Asset Daily Risk per 1 Contract:** `(Price * Multiplier) * (Historical Annual Vol / sqrt(252))`
4.  **Exact Lots:** `Target Daily Risk / Asset Daily Risk per 1 Contract`.

### Portfolio Volatility Allocation
If you target 10% volatility per asset and 5 assets fire a LONG signal simultaneously, your portfolio volatility spikes to 50% (assuming correlation). 
*   **The Fix:** Divide your 10% target by the number of assets in the active universe (`0.10 / N`). This guarantees the aggregate portfolio never breaches the target VaR limit.

### Darwinex D-Score Optimization
*   **Consistency (Cs) / Risk Stability:** Achieved entirely through strict Volatility Targeting (10% VaR) and portfolio-level risk parity.
*   **Loss Aversion (La):** Requires hard ATR-based stop-losses submitted directly to the broker to define tail-risk.
*   **Experience (Ex):** Requires sufficient "D-Periods". A strategy must hold exposure consistently. Aim for an aggregate **15-20 trades per month** across the portfolio with holding periods of 5-10 days.

---

## 4. Validated Alpha Strategies & Data Sources

### The "Narrative Velocity" Strategy
*   **Edge:** Uses GDELT Project API to track the *velocity* (rate of change) of macroeconomic news volume and tone (sentiment).
*   **Mechanics:** Converts raw timeline volume into 60-day rolling Z-scores ("Narrative Shocks").
*   **Best Assets:** Safe-haven and pure macro assets (Euro, GBP, AUD, Gold, 10-Year T-Notes) react cleanest to these shocks.

### The "Volatility Risk Premium (VRP)" Scalper
*   **Edge:** Selling overpriced Implied Volatility (IV) when it significantly exceeds Realized Volatility (RV). e.g., Middle East "War Premium" in Crude Oil options.
*   **Mechanics:** Sell Iron Flies (Short ATM Straddle + Long Wings for protection) and maintain daily Delta-Neutral hedges using underlying futures.

### Structural Dislocation Arbitrage
*   **Edge:** Mean-reversion of historically stretched ratios.
*   **Examples:** Small-Cap vs Large-Cap valuation gaps driven by interest rate sensitivity (RTY/ES spread). Dr. Copper vs Gold (HG/GC spread) driven by electrification supercycles vs. safe-haven hoarding.

### Neural Option Pricing (ResNets & KANs)
*   **Edge:** Black-Scholes fails on stochastic volatility and jump diffusion. Deep Residual Networks and Kolmogorov-Arnold Networks (KANs) reduce pricing errors by >60%.
*   **Application:** Calculate BS Fair Value vs. Neural Net Market Value. Buy Gamma when Neural Net indicates volatility is underpriced; Sell Volatility during complacency regimes.

### Data Engineering Best Practices
*   **GDELT API:** Strictly enforce exponential backoff loops (`time.sleep`) to handle inevitable `429 Too Many Requests` errors.
*   **Data Waterfalls:** Never rely on a single API. E.g., for VXN (Nasdaq Vol), try CBOE direct CSV -> yfinance fallback -> local cache (up to 3 days stale).
*   **Stateless Architecture:** Production bots should not rely on local JSON files to track open trades. Always query MT5 (`mt5.positions_get()`) as the absolute source of truth to reconcile state upon script boot.