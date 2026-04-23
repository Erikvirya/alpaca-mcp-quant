import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

# Asset configuration
SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'RTY': {'ticker': 'RTY=F', 'vol': '^VIX'}, # Use VIX as proxy if RVX fails
    'YM': {'ticker': 'YM=F', 'vol': '^VXD'}
}

def run_multi_asset_test():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    
    all_prices = {}
    all_signals = {}
    
    print("Fetching historical data for multi-asset basket...")
    
    for name, config in SYMBOLS.items():
        try:
            price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
            vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
            
            # Align
            common = price_df.index.intersection(vol_df.index)
            price_df = price_df.loc[common]
            vol_df = vol_df.loc[common]
            
            print(f"Generating signals for {name}...")
            alloc = get_strategy_signals(
                price_df['Close'],
                price_df['High'],
                price_df['Low'],
                vol_df
            )
            
            # We will use 1.0x total target leverage across the basket
            # So each asset gets 0.25x of the 'alloc' signal
            all_prices[name] = price_df['Close']
            all_signals[name] = (alloc / 4.0).shift(1).fillna(0)
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")

    # Combine into DataFrames for VectorBT
    price_matrix = pd.DataFrame(all_prices).ffill()
    signal_matrix = pd.DataFrame(all_signals).fillna(0)
    
    # Trim to test period
    test_mask = price_matrix.index >= '2016-01-01'
    price_matrix = price_matrix[test_mask]
    signal_matrix = signal_matrix[test_mask]

    print(f"Running Portfolio Simulation (Basket size: {len(all_prices)} instruments)...")
    
    pf = vbt.Portfolio.from_orders(
        price_matrix,
        size=signal_matrix,
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='1D'
    )

    print("\n" + "="*50)
    print(f"{'MULTI-ASSET REGIME STRATEGY SUMMARY':^50}")
    print("="*50)
    
    # Use mean aggregation for the whole portfolio stats
    total_stats = pf.stats()
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    
    # Calculate Sharpe manually if vbt returns inf for multi-asset mean
    returns = pf.returns()
    portfolio_returns = returns.mean(axis=1) # Equal weight of the active sub-portfolios
    sharpe = np.sqrt(252) * (portfolio_returns.mean() / portfolio_returns.std())
    
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(total_stats['Total Trades'] / 10.2)}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)
    
    print("\nPer-Asset Statistics:")
    asset_sharpes = pf.sharpe_ratio()
    asset_returns = pf.total_return()
    for sym in asset_sharpes.index:
        print(f"{sym:<5}: Sharpe={asset_sharpes[sym]:.2f}, Return={asset_returns[sym]*100:,.2f}%")

if __name__ == "__main__":
    run_multi_asset_test()
