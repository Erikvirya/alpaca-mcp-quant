import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import os
from strategy_logic import get_strategy_signals

# UNIVERSE: NQ, ES, CL, GC (All highly liquid on Darwinex Zero)
UNIVERSE = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'CL': {'ticker': 'CL=F', 'vol': '^OVX'}, # Crude Oil Volatility
    'GC': {'ticker': 'GC=F', 'vol': '^GVZ'}  # Gold Volatility
}

def run_cross_asset_frequency_test():
    start_date = '2021-01-01'
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching historical data for 5-Year Cross-Asset Basket...")
    for name, config in UNIVERSE.items():
        try:
            # Try to load from parquet if available, else download
            p_path = f"../../market_data/{name}_daily_5yr.parquet"
            if os.path.exists(p_path):
                price_df = pd.read_parquet(p_path)
                if 'close' in price_df.columns: price_df = price_df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'})
            else:
                price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
            
            vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
            
            # Standardize Index
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
            vol_df.index = pd.to_datetime(vol_df.index).tz_localize(None)
            
            common = price_df.index.intersection(vol_df.index)
            price_df, vol_df = price_df.loc[common], vol_df.loc[common]
            
            print(f"Generating signals for {name}...")
            alloc = get_strategy_signals(price_df['Close'], price_df['High'], price_df['Low'], vol_df)
            
            all_prices[name] = price_df['Close']
            # Divide by 4 to maintain total portfolio risk parity
            all_signals[name] = (alloc / 4.0).shift(1).fillna(0)
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")

    price_matrix = pd.DataFrame(all_prices).ffill().dropna()
    signal_matrix = pd.DataFrame(all_signals).fillna(0).loc[price_matrix.index]

    print(f"Running Portfolio Simulation (Basket: {list(all_prices.keys())})...")
    pf = vbt.Portfolio.from_orders(
        price_matrix, size=signal_matrix, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    returns = pf.returns()
    portfolio_returns = returns.mean(axis=1) 
    sharpe = np.sqrt(252) * (portfolio_returns.mean() / portfolio_returns.std())
    total_stats = pf.stats()

    print("\n" + "="*50)
    print(f"{'5-YEAR CROSS-ASSET FREQUENCY SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(total_stats['Total Trades'] / 5.2)}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)
    
    print("\nIndividual Asset Sharpe:")
    asset_sharpes = pf.sharpe_ratio()
    for sym in asset_sharpes.index:
        print(f"{sym:<5}: Sharpe={asset_sharpes[sym]:.2f}")

if __name__ == "__main__":
    run_cross_asset_frequency_test()
