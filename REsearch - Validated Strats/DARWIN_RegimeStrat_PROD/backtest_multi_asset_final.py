import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from strategy_logic import get_strategy_signals

# FINAL PITCH BASKET: NQ (The King) + IBIT (The High-Beta Alpha)
SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'IBIT': {'ticker': 'IBIT', 'vol': '^VXN'} # Use VXN as proxy for IBIT volatility regime
}

def run_final_pitch_backtest():
    start_date = '2024-01-01' # IBIT launch period
    end_date = '2026-03-13'
    all_prices, all_signals = {}, {}
    
    print("Fetching data for Final Pitch Basket (NQ + IBIT)...")
    for name, config in SYMBOLS.items():
        price_df = yf.download(config['ticker'], start=start_date, end=end_date, progress=False, multi_level_index=False)
        vol_df = yf.download(config['vol'], start=start_date, end=end_date, progress=False, multi_level_index=False)['Close']
        common = price_df.index.intersection(vol_df.index)
        price_df, vol_df = price_df.loc[common], vol_df.loc[common]
        
        alloc = get_strategy_signals(price_df['Close'], price_df['High'], price_df['Low'], vol_df)
        all_prices[name] = price_df['Close']
        all_signals[name] = (alloc / 2.0).shift(1).fillna(0) # 50/50 risk split

    price_matrix = pd.DataFrame(all_prices).ffill().dropna()
    signal_matrix = pd.DataFrame(all_signals).fillna(0).loc[price_matrix.index]

    print("Running Portfolio Simulation...")
    pf = vbt.Portfolio.from_orders(
        price_matrix, size=signal_matrix, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    returns = pf.returns()
    portfolio_returns = returns.mean(axis=1) 
    sharpe = np.sqrt(252) * (portfolio_returns.mean() / portfolio_returns.std())
    total_stats = pf.stats()

    print("\n" + "="*50)
    print(f"{'FINAL INSTITUTIONAL PITCH BASKET (NQ + IBIT)':^50}")
    print("="*50)
    print(f"Total Return:      {total_stats['Total Return [%]']:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Max Drawdown:      {total_stats['Max Drawdown [%]']:,.2f}%")
    print(f"Total Trades:      {int(total_stats['Total Trades'])}")
    print(f"Win Rate:          {total_stats['Win Rate [%]']:,.2f}%")
    print("="*50)
    
    print("\nIndividual Contribution:")
    asset_sharpes = pf.sharpe_ratio()
    for sym in asset_sharpes.index:
        print(f"{sym:<5}: Sharpe={asset_sharpes[sym]:.2f}")

if __name__ == "__main__":
    run_final_pitch_backtest()
