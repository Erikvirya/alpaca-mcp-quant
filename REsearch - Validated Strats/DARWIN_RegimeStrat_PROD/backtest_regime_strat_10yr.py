import os
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from datetime import datetime, timedelta

# Import strategy logic from current directory
from strategy_logic import get_strategy_signals
from futures_config import NQ, COMMISSION_PER_SIDE

def run_10yr_backtest():
    print("Fetching 11 years of historical data (including warmup)...")
    end_date = '2026-03-13'
    start_date = '2015-01-01'
    
    # Fetch NQ data
    nq_df = yf.download('NQ=F', start=start_date, end=end_date, progress=False, multi_level_index=False)
    # Fetch VXN data (Nasdaq Volatility Index)
    vxn_df = yf.download('^VXN', start=start_date, end=end_date, progress=False, multi_level_index=False)
    
    # Align data
    common_idx = nq_df.index.intersection(vxn_df.index)
    nq_df = nq_df.loc[common_idx]
    vxn_df = vxn_df.loc[common_idx]
    
    print(f"Data range: {nq_df.index[0].date()} to {nq_df.index[-1].date()}")
    
    # Generate Strategy Signals
    # strategy_logic.get_strategy_signals(close, high, low, vix_series)
    # Note: strategy_logic already handles indicators like RSI, EMA, ADX, ATR
    print("Generating regime-switching signals...")
    alloc_signal = get_strategy_signals(
        nq_df['Close'], 
        nq_df['High'], 
        nq_df['Low'], 
        vxn_df['Close']
    )
    
    # Shift signals by 1 bar to prevent look-ahead bias (signals generated at Close T, traded at Open T+1)
    # vectorbt from_signals also handles this if freq is set, but explicit shift is clearer for this logic.
    trade_signals = alloc_signal.shift(1).fillna(0)
    
    # Warmup period: Start trading from 2016-01-01
    test_mask = trade_signals.index >= '2016-01-01'
    trade_signals = trade_signals[test_mask]
    price_data = nq_df.loc[test_mask]
    
    # VectorBT Simulation
    # We use from_orders because allocation changes size
    # But for a simpler comparison, let's use from_signals if we can map it.
    # Actually, from_orders is better for notional-based allocation.
    
    print("Running VectorBT simulation...")
    # Point-based slippage: 1.0 points = 1.0 / price
    # Fixed commission: $4.00 per side per contract.
    # contract_notional = price * 20
    # commission_pct = 4.0 / (price * 20)
    
    def calc_comm(price):
        return COMMISSION_PER_SIDE / (price * NQ.point_value)

    # Simulation settings
    init_cash = 1_000_000
    
    # Run Portfolio simulation
    pf = vbt.Portfolio.from_orders(
        price_data['Close'],
        size=trade_signals,
        size_type='targetpercent', # Map alloc signal (-0.2 to 2.0) to % of equity
        init_cash=init_cash,
        fees=0.0002, # Approx 0.02% base commission
        slippage=0.0001, # Approx 1pt at 10k, 0.5pt at 20k
        freq='1D'
    )
    
    print("\n" + "="*50)
    print(f"{'DARWIN REGIME STRAT 10-YEAR BACKTEST':^50}")
    print("="*50)
    
    stats = pf.stats()
    # print(stats.index.tolist()) # Debug
    
    def get_val(key):
        return stats.get(key, np.nan)

    print(f"Total Return:      {get_val('Total Return [%]'):,.2f}%")
    print(f"Benchmark Return:  {get_val('Benchmark Return [%]'):,.2f}%")
    print(f"Annualized Return: {get_val('Annualized Return [%]'):,.2f}%")
    print(f"Sharpe Ratio:      {get_val('Sharpe Ratio'):,.2f}")
    print(f"Max Drawdown:      {get_val('Max Drawdown [%]'):,.2f}%")
    print(f"Win Rate:          {get_val('Win Rate [%]'):,.2f}%")
    print(f"Total Trades:      {get_val('Total Trades')}")
    print("="*50)
    
    # Save results
    output_path = "TESTING/10yr_historical_backtest.csv"
    pf.orders.records_readable.to_csv(output_path)
    print(f"Trade log saved to {output_path}")

if __name__ == "__main__":
    run_10yr_backtest()
