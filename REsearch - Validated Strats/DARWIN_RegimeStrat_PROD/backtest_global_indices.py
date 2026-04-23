import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import os
from strategy_logic import get_strategy_signals

# GLOBAL EQUITY BASKET: NQ, ES (US), FDAX (Germany), FESX (Europe)
# All highly liquid on Darwinex Zero.
UNIVERSE = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'DAX': {'ticker': 'FDAX.EX', 'vol': '^VIX'}, # VIX proxy for DAX
    'ESTX': {'ticker': 'FESX.EX', 'vol': '^VIX'} # VIX proxy for EuroStoxx
}

def run_global_index_backtest():
    start_date = '2021-01-01'
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching historical data for Global Index Basket...")
    for name, config in UNIVERSE.items():
        try:
            # Note: European tickers might need different handling via yfinance or Databento
            # Let's try standard Yahoo tickers first
            price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
            vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
            
            if price_df.empty or vol_df.empty:
                # Fallback for European indices
                if name == 'DAX': ticker = '^GDAXI'
                elif name == 'ESTX': ticker = '^STOXX50E'
                else: continue
                price_df = yf.download(ticker, start=start_date, end=end_date, progress=False, multi_level_index=False)
            
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
            vol_df.index = pd.to_datetime(vol_df.index).tz_localize(None)
            
            common = price_df.index.intersection(vol_df.index)
            price_df, vol_df = price_df.loc[common], vol_df.loc[common]
            
            print(f"Generating signals for {name}...")
            alloc = get_strategy_signals(price_df['Close'], price_df['High'], price_df['Low'], vol_df)
            
            all_prices[name] = price_df['Close']
            # Divide by number of assets (4)
            all_signals[name] = (alloc / 4.0).shift(1).fillna(0)
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")

    if not all_prices:
        print("No data available.")
        return

    price_matrix = pd.DataFrame(all_prices).ffill().dropna()
    signal_matrix = pd.DataFrame(all_signals).fillna(0).loc[price_matrix.index]

    print(f"Running Portfolio Simulation (Basket: {list(all_prices.keys())})...")
    pf = vbt.Portfolio.from_orders(
        price_matrix, size=signal_matrix, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    returns = pf.returns().mean(axis=1) 
    sharpe = np.sqrt(252) * (returns.mean() / returns.std())
    total_stats = pf.stats()

    print("\n" + "="*50)
    print(f"{'GLOBAL EQUITY INDEX BASKET SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(total_stats['Total Trades'] / 5.2)}")
    print("="*50)
    
    print("\nIndividual Contribution (Sharpe):")
    asset_sharpes = pf.sharpe_ratio()
    for sym in asset_sharpes.index:
        print(f"{sym:<5}: Sharpe={asset_sharpes[sym]:.2f}")

if __name__ == "__main__":
    run_global_index_backtest()
