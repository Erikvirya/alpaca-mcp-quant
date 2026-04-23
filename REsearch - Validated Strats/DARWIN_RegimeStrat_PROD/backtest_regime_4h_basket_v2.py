import pandas as pd
import numpy as np
import yfinance as yf
import os
from strategy_logic import get_strategy_signals

def run_4h_basket_test():
    # 1. Fetch 2 Years of 1-Hour data (YF Limit)
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=729)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    symbols = {
        'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
        'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
        'YM': {'ticker': 'YM=F', 'vol': '^VXD'}
    }
    
    dfs_4h = {}
    vols_4h = {}
    
    print(f"Fetching 2 years of 1-hour data for NQ, ES, YM...")
    for name, config in symbols.items():
        try:
            # Fetch 1h and resample to 4h
            p_df = yf.download(config['ticker'], start=start_date, end=end_date, interval='1h', progress=False, multi_level_index=False)
            v_df = yf.download(config['vol'], start=start_date, end=end_date, interval='1h', progress=False, multi_level_index=False)['Close']
            
            p_df.index = p_df.index.tz_localize(None)
            v_df.index = v_df.index.tz_localize(None)
            
            # Resample
            p_4h = p_df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            v_4h = v_df.resample('4h').last().ffill()
            
            common = p_4h.index.intersection(v_4h.index)
            dfs_4h[name] = p_4h.loc[common]
            vols_4h[name] = v_4h.loc[common]
            print(f"Loaded {len(common)} bars for {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    if len(dfs_4h) < 3:
        print("Insufficient data for full basket.")
        return

    # Align all to a single timeline
    all_idx = dfs_4h['NQ'].index
    for name in ['ES', 'YM']:
        all_idx = all_idx.intersection(dfs_4h[name].index)
    
    for name in symbols:
        dfs_4h[name] = dfs_4h[name].loc[all_idx]
        vols_4h[name] = vols_4h[name].loc[all_idx]

    # --- SIMULATION ---
    print("\nStarting Multi-Asset 4H Simulation...")
    equity = 1_000_000.0
    initial_cash = equity
    
    # State per asset: {name: {'notional': 0.0, 'units': 0.0}}
    positions = {name: 0.0 for name in symbols}
    
    comm_per_side = 4.0
    slip_pts = 0.5
    point_values = {'NQ': 20, 'ES': 50, 'YM': 5}
    
    history = []
    total_trades = 0
    
    # Generate signals
    signals = {}
    for name in symbols:
        # Use 1/3 weight per asset
        raw_alloc = get_strategy_signals(dfs_4h[name]['Close'], dfs_4h[name]['High'], dfs_4h[name]['Low'], vols_4h[name])
        signals[name] = (raw_alloc / 3.0).shift(1).fillna(0)

    dates = all_idx.tolist()
    
    for i in range(1, len(dates)):
        dt = dates[i]
        prev_dt = dates[i-1]
        
        # 1. Update Equity from price moves
        for name in symbols:
            if positions[name] != 0:
                price_change = dfs_4h[name].loc[dt, 'Close'] - dfs_4h[name].loc[prev_dt, 'Close']
                # PnL = (Previous_Notional / Previous_Price) * Change
                equity += (positions[name] / dfs_4h[name].loc[prev_dt, 'Close']) * price_change
        
        # 2. Rebalance with Tolerance (0.05 absolute change required)
        for name in symbols:
            target_alloc = signals[name].loc[dt]
            current_alloc = positions[name] / equity
            
            if abs(target_alloc - current_alloc) > 0.02: # 2% rebalance threshold
                target_notional = equity * target_alloc
                trade_notional = abs(target_notional - positions[name])
                
                # Approximate contracts
                price = dfs_4h[name].loc[dt, 'Close']
                pv = point_values[name]
                cts = trade_notional / (price * pv)
                
                # Friction
                cost = (cts * comm_per_side) + (cts * pv * slip_pts)
                equity -= cost
                
                positions[name] = target_notional
                total_trades += 1
        
        history.append(equity)

    # --- METRICS ---
    history_series = pd.Series(history, index=dates[1:])
    returns = history_series.pct_change().dropna()
    
    # 252 days * 6 bars/day = 1512
    sharpe = np.sqrt(1512) * (returns.mean() / returns.std())
    total_ret = (history_series.iloc[-1] / initial_cash - 1) * 100
    
    print("\n" + "="*50)
    print(f"{'4-HOUR MULTI-ASSET BASKET (NQ, ES, YM)':^50}")
    print("="*50)
    print(f"Period:            {dates[0].date()} to {dates[-1].date()}")
    print(f"Total Return:      {total_ret:,.2f}%")
    print(f"Portfolio Sharpe:  {sharpe:.2f}")
    print(f"Total Trades:      {total_trades}")
    print(f"Avg Trades/Year:   {int(total_trades / 2)}")
    print("="*50)

if __name__ == "__main__":
    run_4h_basket_test()
