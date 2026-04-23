import pandas as pd
import numpy as np
import os
import vectorbt as vbt
import yfinance as yf
from strategy_logic import get_strategy_signals

def run_4h_turbo_backtest():
    # 1. Load 5-Year High-Fidelity Data
    file_path = "../../market_data/NQ_stitched_1m_5yr_2021-03-01_2026-02-28.parquet"
    if not os.path.exists(file_path):
        print("Data file not found.")
        return

    print("Loading 5-Year NQ Intraday Data...")
    df_1m = pd.read_parquet(file_path)
    df_1m.index = pd.to_datetime(df_1m.index)
    
    # 2. Resample to 4-Hour (240 mins)
    print("Resampling to 4-Hour bars...")
    df_4h = df_1m.resample('240min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    # Standardize Index (Naive for yfinance alignment)
    if df_4h.index.tz is not None:
        df_4h.index = df_4h.index.tz_convert(None)
    else:
        df_4h.index = df_4h.index.tz_localize(None)

    # 3. Align Volatility Proxy (VXN)
    print("Aligning VXN Volatility...")
    vxn = yf.download('^VXN', start='2021-01-01', end='2026-03-13', progress=False, multi_level_index=False)['Close']
    vxn.index = pd.to_datetime(vxn.index).tz_localize(None)
    vxn_aligned = vxn.reindex(df_4h.index, method='ffill')
    
    # Clean all NaNs before logic
    valid_mask = vxn_aligned.notna() & (df_4h['close'] > 0)
    df_4h = df_4h[valid_mask]
    vxn_aligned = vxn_aligned[valid_mask]

    # 4. Generate Signals
    print("Generating 4-Hour Regime Signals...")
    alloc = get_strategy_signals(df_4h['close'], df_4h['high'], df_4h['low'], vxn_aligned)
    
    # 5. Execute with VectorBT (Target Percent to handle compounding)
    print("Running VectorBT Simulation...")
    # Shift signals by 1 to ensure we trade at the NEXT 4H bar's open
    trade_signals = alloc.shift(1).fillna(0)
    
    pf = vbt.Portfolio.from_orders(
        df_4h['close'],
        size=trade_signals,
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='4h'
    )

    # 6. Results
    stats = pf.stats()
    # Manual Sharpe for 4H accuracy
    rets = pf.returns()
    # 252 days * 6 bars/day (for 24h futures) = 1512
    sharpe = np.sqrt(1512) * (rets.mean() / rets.std())

    print("\n" + "="*50)
    print(f"{'4-HOUR NQ TURBO STRATEGY SUMMARY':^50}")
    print("="*50)
    print(f"Total Trades:      {int(stats['Total Trades'])}")
    print(f"Trades Per Year:   {int(stats['Total Trades'] / 5)}")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Total Return:      {stats['Total Return [%]']:,.2f}%")
    print(f"Max Drawdown:      {stats['Max Drawdown [%]']:,.2f}%")
    print(f"Win Rate:          {stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_4h_turbo_backtest()
