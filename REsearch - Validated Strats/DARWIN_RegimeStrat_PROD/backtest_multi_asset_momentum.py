import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

# Momentum Basket: Dynamically weight towards the strongest relative performer
SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'YM': {'ticker': 'YM=F', 'vol': '^VXD'}
}

def run_momentum_multi_asset():
    start_date = '2015-01-01'
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching data for Momentum-Weighted basket...")
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
    
    # Weighting Logic: Give 60% weight to the top performer, 30% to 2nd, 10% to 3rd
    ranks = rel_strength.rank(axis=1, ascending=False)
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    weights[ranks == 1.0] = 0.60
    weights[ranks == 2.0] = 0.30
    weights[ranks == 3.0] = 0.10
    
    # Shift weights to avoid look-ahead
    weights = weights.shift(1).fillna(0.33)
    
    # Final ML-Logic Signal: Combined Allocation * Momentum Weight
    final_signals = base_signal_matrix * weights
    
    test_mask = price_matrix.index >= '2016-01-01'
    pf = vbt.Portfolio.from_orders(
        price_matrix[test_mask], final_signals[test_mask], size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    returns = pf.returns()
    portfolio_returns = returns.mean(axis=1) 
    sharpe = np.sqrt(252) * (portfolio_returns.mean() / portfolio_returns.std())
    total_stats = pf.stats()

    print("\n" + "="*50)
    print(f"{'MOMENTUM-WEIGHTED MULTI-ASSET SUMMARY':^50}")
    print("="*50)
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_momentum_multi_asset()
