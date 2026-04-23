import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

# WEIGHTED Basket: 50% NQ (Alpha), 25% ES, 25% YM
SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN', 'weight': 0.50},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX', 'weight': 0.25},
    'YM': {'ticker': 'YM=F', 'vol': '^VXD', 'weight': 0.25}
}

def run_weighted_basket_test():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching data for Weighted 3-Asset Basket (50/25/25)...")
    for name, config in SYMBOLS.items():
        price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
        vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
        common = price_df.index.intersection(vol_df.index)
        price_df, vol_df = price_df.loc[common], vol_df.loc[common]
        
        alloc = get_strategy_signals(price_df['Close'], price_df['High'], price_df['Low'], vol_df)
        all_prices[name] = price_df['Close']
        # Apply the symbol weight to the allocation signal
        all_signals[name] = (alloc * config['weight']).shift(1).fillna(0)

    price_matrix = pd.DataFrame(all_prices).ffill().dropna()
    signal_matrix = pd.DataFrame(all_signals).fillna(0).loc[price_matrix.index]

    print("Running Portfolio Simulation...")
    pf = vbt.Portfolio.from_orders(
        price_matrix, size=signal_matrix, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    # Portfolio Returns
    rets = pf.returns().mean(axis=1) 
    sharpe = np.sqrt(252) * (rets.mean() / rets.std())
    total_stats = pf.stats()

    print("\n" + "="*50)
    print(f"{'WEIGHTED 3-ASSET BASKET (NQ-HEAVY) SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Avg Trades/Year:   {int(total_stats['Total Trades'] / 10.2)}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)
    
    print("\nPer-Asset Sharpe:")
    print(pf.sharpe_ratio())

if __name__ == "__main__":
    run_weighted_basket_test()
