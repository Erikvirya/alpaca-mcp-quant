import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

def run_asymmetric_backtest():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    
    print("Fetching NQ data for Asymmetric Frequency Test...")
    nq_df = yf.download('NQ=F', start=start_date, end=end_date, progress=False, multi_level_index=False)
    vxn_df = yf.download('^VXN', start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
    
    common = nq_df.index.intersection(vxn_df.index)
    nq_df = nq_df.loc[common]
    vxn_df = vxn_df.loc[common]
    
    print("Generating standard signals...")
    # Base allocation
    alloc = get_strategy_signals(nq_df['Close'], nq_df['High'], nq_df['Low'], vxn_df)
    
    # --- ASYMMETRIC ENHANCEMENT ---
    # In the original, a trade might stay open for weeks.
    # We add a "Momentum Exhaustion" exit to close trades early and reopen them, 
    # increasing frequency/turnover without changing the core directional bias.
    
    rsi = vbt.RSI.run(nq_df['Close'], window=14).rsi
    
    # Exit Long early if RSI crosses above 70 (Overbought)
    # Exit Short early if RSI crosses below 30 (Oversold)
    # This forces the strategy to "recycle" capital more often.
    
    exhaustion_exit = (rsi > 70) | (rsi < 30)
    
    # If exhausted, set alloc to 0 for 1 day
    asym_alloc = alloc.copy()
    asym_alloc.loc[exhaustion_exit] = 0
    
    print("Running VectorBT Portfolio Simulation (Asymmetric)...")
    pf = vbt.Portfolio.from_orders(
        nq_df['Close'],
        size=asym_alloc.shift(1).fillna(0),
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='1D'
    )

    stats = pf.stats()
    print("\n" + "="*50)
    print(f"{'ASYMMETRIC FREQUENCY ENHANCEMENT SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {stats['Total Return [%]']:,.2f}%")
    print(f"Sharpe Ratio:      {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:      {stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(stats['Total Trades'] / 10.2)}")
    print(f"Win Rate:          {stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_asymmetric_backtest()
