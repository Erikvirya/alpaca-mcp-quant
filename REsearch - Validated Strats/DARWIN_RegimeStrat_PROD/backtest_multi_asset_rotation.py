import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

# Multi-Asset ROTATION: Allocate to the single best performer to keep Sharpe high and frequency up.
SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'YM': {'ticker': 'YM=F', 'vol': '^VXD'}
}

def run_rotation_multi_asset():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching data for Index Rotation basket...")
    for name, config in SYMBOLS.items():
        price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
        vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
        common = price_df.index.intersection(vol_df.index)
        price_df, vol_df = price_df.loc[common], vol_df.loc[common]
        
        alloc = get_strategy_signals(price_df['Close'], price_df['High'], price_df['Low'], vol_df)
        all_prices[name] = price_df['Close']
        all_signals[name] = alloc

    price_matrix = pd.DataFrame(all_prices).ffill()
    base_signal_matrix = pd.DataFrame(all_signals).fillna(0)
    
    # Calculate Relative Strength (20-day return)
    rel_strength = price_matrix.pct_change(20)
    
    # Rotation Logic: Identify the #1 performer
    best_asset = rel_strength.idxmax(axis=1)
    
    # Create target matrix: 1.0 for the best asset, 0.0 for others
    # Then multiply by the regime allocation signal
    rotation_weights = pd.DataFrame(0.0, index=price_matrix.index, columns=price_matrix.columns)
    for date, sym in best_asset.items():
        if pd.notna(sym):
            rotation_weights.loc[date, sym] = 1.0
            
    # Shift weights to avoid look-ahead
    rotation_weights = rotation_weights.shift(1).fillna(0)
    
    # Final Signal: Only trade the BEST asset according to its regime logic
    final_signals = base_signal_matrix * rotation_weights
    
    test_mask = price_matrix.index >= '2016-01-01'
    
    print("Running VectorBT Simulation (Top-1 Rotation)...")
    pf = vbt.Portfolio.from_orders(
        price_matrix[test_mask], 
        final_signals[test_mask], 
        size_type='targetpercent',
        init_cash=1_000_000, 
        fees=0.0002, 
        slippage=0.0001, 
        freq='1D'
    )

    stats = pf.stats()
    print("\n" + "="*50)
    print(f"{'MULTI-ASSET ROTATION SUMMARY (TOP-1 PERFORMER)':^50}")
    print("="*50)
    print(f"Total Return:      {stats['Total Return [%]']:,.2f}%")
    
    # Portfolio Returns
    rets = pf.returns().sum(axis=1) # Sum because only one asset is active at a time
    sharpe = np.sqrt(252) * (rets.mean() / rets.std())
    
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(stats['Total Trades'] / 10.2)}")
    print(f"Win Rate:          {stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_rotation_multi_asset()
