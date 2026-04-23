import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

# REFINED Basket: Only the strongest performers
SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'YM': {'ticker': 'YM=F', 'vol': '^VXD'}
}

def run_buffered_multi_asset():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching historical data for BUFFERED multi-asset basket...")
    for name, config in SYMBOLS.items():
        price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
        vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
        common = price_df.index.intersection(vol_df.index)
        price_df, vol_df = price_df.loc[common], vol_df.loc[common]
        
        alloc = get_strategy_signals(price_df['Close'], price_df['High'], price_df['Low'], vol_df)
        
        # --- BUFFER LOGIC (Hysteresis) ---
        # Only change allocation if the shift is significant (> 0.05 absolute)
        buffered_alloc = [0.0]
        current = 0.0
        target_series = (alloc / 3.0).values
        for target in target_series:
            if abs(target - current) > 0.05: # 5% minimum shift to trade
                current = target
            buffered_alloc.append(current)
        
        all_prices[name] = price_df['Close']
        all_signals[name] = pd.Series(buffered_alloc[:-1], index=price_df.index).shift(1).fillna(0)

    price_matrix = pd.DataFrame(all_prices).ffill()
    signal_matrix = pd.DataFrame(all_signals).fillna(0)
    test_mask = price_matrix.index >= '2016-01-01'
    price_matrix, signal_matrix = price_matrix[test_mask], signal_matrix[test_mask]

    print(f"Running Portfolio Simulation (3 Assets, 5% Buffer)...")
    pf = vbt.Portfolio.from_orders(
        price_matrix, size=signal_matrix, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    returns = pf.returns().mean(axis=1) 
    sharpe = np.sqrt(252) * (returns.mean() / returns.std())
    total_stats = pf.stats()

    print("\n" + "="*50)
    print(f"{'BUFFERED MULTI-ASSET REGIME SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(total_stats['Total Trades'] / 10.2)}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)
    
    asset_sharpes = pf.sharpe_ratio()
    for sym in asset_sharpes.index:
        print(f"{sym:<5}: Sharpe={asset_sharpes[sym]:.2f}")

if __name__ == "__main__":
    run_buffered_multi_asset()
