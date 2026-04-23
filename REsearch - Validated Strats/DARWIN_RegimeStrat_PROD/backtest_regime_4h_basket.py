import pandas as pd
import numpy as np
import os
import vectorbt as vbt
import yfinance as yf
from strategy_logic import get_strategy_signals

def run_4h_dual_basket():
    # File paths (Adjusted to point from within subdirectory to root market_data)
    nq_path = "../../market_data/NQ_stitched_1m_5yr_2021-03-01_2026-02-28.parquet"
    es_path = "../../market_data/ES_1m_1yr.parquet"
    
    if not os.path.exists(nq_path) or not os.path.exists(es_path):
        print("Required data files not found.")
        return

    print("Loading NQ and ES intraday data...")
    df_nq_1m = pd.read_parquet(nq_path)
    df_es_1m = pd.read_parquet(es_path)
    
    df_nq_1m.index = pd.to_datetime(df_nq_1m.index)
    df_es_1m.index = pd.to_datetime(df_es_1m.index)
    
    def prepare_4h(df):
        res = df.resample('240min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        res = res[res['close'] > 0]
        res.index = res.index.tz_localize(None)
        return res

    print("Resampling to 4-Hour bars...")
    df_nq = prepare_4h(df_nq_1m)
    df_es = prepare_4h(df_es_1m)
    
    print(f"NQ index sample: {df_nq.index[:2]}")
    print(f"ES index sample: {df_es.index[:2]}")
    
    # Alignment: Check if both are naive or both TZ-aware
    if df_nq.index.tz is not None: df_nq.index = df_nq.index.tz_convert(None)
    if df_es.index.tz is not None: df_es.index = df_es.index.tz_convert(None)
    
    common_idx = df_nq.index.intersection(df_es.index)
    if len(common_idx) == 0:
        print("ERROR: No overlap found between NQ and ES resampled data.")
        # Print date ranges for debugging
        print(f"NQ Range: {df_nq.index.min()} to {df_nq.index.max()}")
        print(f"ES Range: {df_es.index.min()} to {df_es.index.max()}")
        return

    # Volatility Alignment
    print("Fetching Volatility proxies...")
    vxn = yf.download('^VXN', start='2024-01-01', end='2026-03-13', progress=False, multi_level_index=False)['Close']
    vix = yf.download('^VIX', start='2024-01-01', end='2026-03-13', progress=False, multi_level_index=False)['Close']
    vxn.index = pd.to_datetime(vxn.index).tz_localize(None)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    
    vxn_4h = vxn.reindex(common_idx).ffill()
    vix_4h = vix.reindex(common_idx).ffill()

    print("Generating dual-asset signals...")
    nq_alloc = get_strategy_signals(df_nq['close'], df_nq['high'], df_nq['low'], vxn_4h)
    es_alloc = get_strategy_signals(df_es['close'], df_es['high'], df_es['low'], vix_4h)
    
    # 50/50 weighting
    all_prices = pd.DataFrame({'NQ': df_nq['close'], 'ES': df_es['close']})
    # Target percent of equity: (Alloc / 2) per asset
    all_signals = pd.DataFrame({
        'NQ': (nq_alloc / 2.0).shift(1).fillna(0),
        'ES': (es_alloc / 2.0).shift(1).fillna(0)
    })

    print("Running VectorBT Portfolio Simulation...")
    # Using from_orders with targetpercent to handle rebalancing correctly
    pf = vbt.Portfolio.from_orders(
        all_prices,
        size=all_signals,
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='240min'
    )

    total_stats = pf.stats()
    # Calculate Sharpe manually to ensure correctness at high frequency
    returns = pf.returns().mean(axis=1)
    # 4H bars = 6.5 / 4 ~= 1.6 bars per day. Let's use 252 * 2 bars per year approx.
    # Actually 24h trading for futures, but indices usually trade heavily 6.5h.
    # We use 252 * (24/4) = 1512 bars/yr for futures.
    sharpe = np.sqrt(1512) * (returns.mean() / returns.std())

    print("\n" + "="*50)
    print(f"{'4-HOUR DUAL BASKET (NQ+ES) SUMMARY':^50}")
    print("="*50)
    print(f"Period:            {common_idx[0].date()} to {common_idx[-1].date()}")
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Annualized Ret:    {total_stats['Annualized Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_4h_dual_basket()
