import pandas as pd
import numpy as np
import os
import yfinance as yf
from strategy_logic import get_strategy_signals

def run_custom_1h_backtest():
    # 1. Load Data
    file_path = "../../market_data/NQ_stitched_1m_5yr_2021-03-01_2026-02-28.parquet"
    if not os.path.exists(file_path):
        print("Data file not found.")
        return

    print("Loading 5-Year NQ Intraday Data...")
    df_1m = pd.read_parquet(file_path)
    df_1m.index = pd.to_datetime(df_1m.index)
    
    # 2. Resample to 1-Hour
    print("Resampling to 1-Hour bars...")
    df_1h = df_1m.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_1h.index = df_1h.index.tz_localize(None)

    # 3. Volatility Proxy (VXN)
    print("Aligning VXN Volatility...")
    vxn = yf.download('^VXN', start='2021-01-01', end='2026-03-13', progress=False, multi_level_index=False)['Close']
    vxn.index = pd.to_datetime(vxn.index).tz_localize(None)
    vxn_aligned = vxn.reindex(df_1h.index, method='ffill')
    
    # 4. Generate Signals
    print("Generating Strategy Signals...")
    # Clean data first
    valid = vxn_aligned.notna() & (df_1h['close'] > 0)
    df_1h = df_1h[valid]
    vxn_aligned = vxn_aligned[valid]
    
    # Get the raw allocation signal (-0.2 to 2.0)
    alloc_signal = get_strategy_signals(df_1h['close'], df_1h['high'], df_1h['low'], vxn_aligned)
    
    # --- CUSTOM SIMULATION ENGINE ---
    print("Starting Custom 1H Simulation...")
    
    initial_cash = 1_000_000.0
    cash = initial_cash
    equity = initial_cash
    position_notional = 0.0
    
    commission_per_contract = 4.0 # $4.00 per NQ side
    point_value = 20.0
    slippage_pts = 0.5 # 0.5 points per trade
    
    history = []
    trades_count = 0
    
    # Shift signals to trade at the NEXT bar's Open
    # Signal at end of bar T -> Trade at Open of bar T+1
    target_allocations = alloc_signal.shift(1).fillna(0)
    
    current_prices = df_1h['close'].values
    open_prices = df_1h['open'].values
    allocs = target_allocations.values
    timestamps = df_1h.index
    
    for i in range(1, len(df_1h)):
        # 1. Update Equity based on current price move
        price_change = current_prices[i] - current_prices[i-1]
        # notional = units * price -> units = notional / price
        # Change in equity = units * price_change
        if position_notional != 0:
            units = position_notional / current_prices[i-1]
            pnl = units * price_change * point_value # This would be if we held contracts
            # Simpler: equity_pct_change = (price_change / prev_price) * current_leverage
            # But let's stay with Notional tracking for accuracy
            equity += (position_notional / current_prices[i-1]) * price_change
        
        # 2. Rebalance to Target Allocation
        target_alloc = allocs[i]
        target_notional = equity * target_alloc
        
        if target_notional != position_notional:
            # We need to trade
            trade_size_dollars = abs(target_notional - position_notional)
            
            # Approximate contracts for commission calculation
            contracts = trade_size_dollars / (current_prices[i] * point_value)
            commissions = contracts * commission_per_contract
            
            # Slippage (impact on equity)
            # slip_cost = units_to_trade * slippage_pts * point_value
            units_traded = trade_size_dollars / current_prices[i]
            slip_cost = units_traded * slippage_pts
            
            equity -= (commissions + slip_cost)
            position_notional = target_notional
            trades_count += 1
            
        history.append(equity)

    # 5. Metrics
    history_series = pd.Series(history, index=timestamps[1:])
    returns = history_series.pct_change().dropna()
    
    total_return = (history_series.iloc[-1] / initial_cash - 1) * 100
    # Annualize Sharpe: 252 days * 6.5 hours = 1638 bars/yr
    sharpe = np.sqrt(1638) * (returns.mean() / returns.std())
    
    # Max Drawdown
    cum_max = history_series.cummax()
    drawdown = (history_series - cum_max) / cum_max
    max_dd = drawdown.min() * 100

    print("\n" + "="*50)
    print(f"{'CUSTOM 1-HOUR NQ BACKTEST RESULTS':^50}")
    print("="*50)
    print(f"Total Return:      {total_return:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print(f"Total Trades:      {trades_count}")
    print(f"Avg Trades/Year:   {int(trades_count / 5)}")
    print("="*50)

if __name__ == "__main__":
    run_custom_1h_backtest()
