import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import os

# Custom Logic implementation directly in backtest to test sensitivity
def get_high_freq_signals(close, high, low, vxn):
    # 1. Base Indicators
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi = vbt.RSI.run(close, window=14).rsi
    
    # TIGHTER Bollinger Bands for faster turnover (1.5 SD instead of 2.0)
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    upper_bb = bb_ma + (1.5 * bb_std)
    lower_bb = bb_ma - (1.5 * bb_std)
    
    # 2. Logic (Directly from strategy_logic.py but with TIGHTER exits)
    # Trend: Long if Price > EMA50
    # Range: Mean Revert if Price hits 1.5 SD
    
    alloc = pd.Series(0.0, index=close.index)
    
    # Trend Regime
    trend_mask = close > ema50
    alloc.loc[trend_mask] = 1.0 # Base trend alloc
    
    # Range Regime (Mean Reversion Overlay)
    mr_long = close < lower_bb
    mr_short = close > upper_bb
    
    # In RANGE (not trending), use 0.5x leverage for MR
    alloc.loc[~trend_mask & mr_long] = 0.5
    alloc.loc[~trend_mask & mr_short] = -0.2
    
    # Global Volatility Protection (Crisis Scalar)
    crisis_scalar = (1.0 - ((vxn - 25) / 25).clip(0, 1)).clip(lower=0.05)
    
    return alloc * crisis_scalar

def run_tight_logic_test():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    
    print("Fetching NQ data for Tight BB (High Freq) Test...")
    nq_df = yf.download('NQ=F', start=start_date, end=end_date, progress=False, multi_level_index=False)
    vxn_df = yf.download('^VXN', start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
    
    common = nq_df.index.intersection(vxn_df.index)
    nq_df = nq_df.loc[common]
    vxn_df = vxn_df.loc[common]
    
    alloc = get_high_freq_signals(nq_df['Close'], nq_df['High'], nq_df['Low'], vxn_df)
    
    print("Running VectorBT Simulation (Tight 1.5 SD)...")
    pf = vbt.Portfolio.from_orders(
        nq_df['Close'],
        size=alloc.shift(1).fillna(0),
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='1D'
    )

    stats = pf.stats()
    print("\n" + "="*50)
    print(f"{'TIGHT EXIT (1.5 SD) FREQUENCY ENHANCEMENT':^50}")
    print("="*50)
    print(f"Total Return:      {stats['Total Return [%]']:,.2f}%")
    print(f"Sharpe Ratio:      {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:      {stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(stats['Total Trades'] / 10.2)}")
    print(f"Win Rate:          {stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_tight_logic_test()
