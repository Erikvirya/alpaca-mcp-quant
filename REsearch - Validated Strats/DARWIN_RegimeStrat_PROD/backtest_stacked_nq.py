import pandas as pd
import numpy as np
import os
import vectorbt as vbt
import yfinance as yf
from strategy_logic import get_strategy_signals

def run_stacked_nq_test():
    # 1. Load Data
    file_path = "../../market_data/NQ_stitched_1m_5yr_2021-03-01_2026-02-28.parquet"
    print("Loading 5-Year NQ Data for Stacked Strategy...")
    df_1m = pd.read_parquet(file_path)
    df_1m.index = pd.to_datetime(df_1m.index)
    
    # Resample to 1H for the Intraday Engine
    df_1h = df_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df_1h.index = df_1h.index.tz_localize(None)
    
    # 2. Get Core Daily Signals (Resampled from 1H for alignment)
    print("Generating Core Regime Signals...")
    vxn_daily = yf.download('^VXN', start='2021-01-01', end='2026-03-13', progress=False, multi_level_index=False)['Close']
    vxn_daily.index = pd.to_datetime(vxn_daily.index).tz_localize(None)
    vxn_1h = vxn_daily.reindex(df_1h.index).ffill()
    
    core_alloc = get_strategy_signals(df_1h['close'], df_1h['high'], df_1h['low'], vxn_1h)
    
    # 3. Generate Intraday Mean-Reversion Signals (The "Stacked" layer)
    print("Generating Intraday Mean-Reversion (Stacked) signals...")
    # RSI(10) on 1H
    delta = df_1h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rsi_1h = 100 - (100 / (1 + gain/loss))
    
    # Bollinger Bands on 1H
    bb_ma = df_1h['close'].rolling(20).mean()
    bb_std = df_1h['close'].rolling(20).std()
    lower_bb = bb_ma - (2.0 * bb_std)
    upper_bb = bb_ma + (2.0 * bb_std)
    
    # MR Logic: Only active if Core Strategy is NOT in a strong trend (alloc < 1.0)
    mr_long = (rsi_1h < 30) & (df_1h['close'] < lower_bb) & (core_alloc < 1.0)
    mr_short = (rsi_1h > 70) & (df_1h['close'] > upper_bb) & (core_alloc < 1.0)
    
    mr_alloc = pd.Series(0.0, index=df_1h.index)
    mr_alloc.loc[mr_long] = 0.3 # Add 0.3x leverage on intraday dips
    mr_alloc.loc[mr_short] = -0.1 # Light short on intraday rips
    
    # 4. Final Stacked Allocation
    final_alloc = core_alloc + mr_alloc
    final_alloc = final_alloc.clip(-0.2, 2.0) # Maintain overall risk caps
    
    # 5. Simulation
    print("Running VectorBT Portfolio (Stacked Logic)...")
    pf = vbt.Portfolio.from_orders(
        df_1h['close'],
        size=final_alloc.shift(1).fillna(0),
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='1h'
    )

    stats = pf.stats()
    print("\n" + "="*50)
    print(f"{'STACKED NQ STRATEGY (DAILY + INTRADAY) SUMMARY':^50}")
    print("="*50)
    # Note: VectorBT might still have compounding issues on 1H if not careful
    # But since we use targetpercent, it should stay within bounds.
    
    # Manual Return calculation to verify
    rets = pf.returns()
    sharpe = np.sqrt(252 * 6.5) * (rets.mean() / rets.std()) # 6.5 trading hours per day
    
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Total Return:      {stats['Total Return [%]']:,.2f}%")
    print(f"Max Drawdown:      {stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(stats['Total Trades'] / 5)}")
    print("="*50)

if __name__ == "__main__":
    run_stacked_nq_test()
