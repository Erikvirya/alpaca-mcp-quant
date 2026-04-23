import pandas as pd
import numpy as np
import os
import vectorbt as vbt
import yfinance as yf

# Import strategy logic from current directory
from strategy_logic import get_strategy_signals

def run_4h_frequency_test():
    file_path = "../../market_data/NQ_stitched_1m_5yr_2021-03-01_2026-02-28.parquet"
    if not os.path.exists(file_path):
        print(f"Data not found at {file_path}.")
        return

    print(f"Loading 5-Year NQ Data...")
    df_1m = pd.read_parquet(file_path)
    df_1m.index = pd.to_datetime(df_1m.index)
    
    # Resample to 4-Hour
    df_4h = df_1m.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_4h = df_4h[df_4h['close'] > 0]
    df_4h.index = df_4h.index.tz_localize(None)

    # Volatility Alignment
    vxn_daily = yf.download('^VXN', start='2021-01-01', end='2026-03-13', progress=False, multi_level_index=False)['Close']
    vxn_daily.index = pd.to_datetime(vxn_daily.index).tz_localize(None)
    vxn_4h = vxn_daily.reindex(df_4h.index).ffill()
    
    # Combine and drop all NaNs to ensure alignment
    full_df = pd.concat([df_4h, vxn_4h.rename('vxn')], axis=1).dropna()
    
    print("Generating 4H strategy signals...")
    alloc = get_strategy_signals(full_df['close'], full_df['high'], full_df['low'], full_df['vxn'])
    
    # Prepare clean inputs for VectorBT
    prices = full_df['close']
    size_signal = alloc.shift(1).fillna(0)
    
    # Final check for infinite or non-positive prices
    mask = np.isfinite(prices) & (prices > 0)
    prices = prices[mask]
    size_signal = size_signal[mask]

    print("Running VectorBT Portfolio (Fixed Compounding)...")
    pf = vbt.Portfolio.from_orders(
        prices,
        size=size_signal,
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='4h'
    )

    stats = pf.stats()
    print("\n" + "="*50)
    print(f"{'4-HOUR REGIME STRATEGY SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {stats['Total Return [%]']:,.2f}%")
    # print(f"Annualized Ret:    {stats['Annualized Return [%]']:,.2f}%")
    print(f"Sharpe Ratio:      {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:      {stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(stats['Total Trades'])}")
    print(f"Win Rate:          {stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_4h_frequency_test()
