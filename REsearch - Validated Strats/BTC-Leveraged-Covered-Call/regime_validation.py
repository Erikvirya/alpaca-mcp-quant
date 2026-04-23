import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import timedelta

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

def call_delta(S, K, T, r, sigma):
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1, 0.0, 1.0)

def find_strike_for_delta(S, T, r, sigma, target_delta=0.20):
    low, high = S * 0.5, S * 5.0
    for _ in range(30):
        mid = (low + high) / 2
        if call_delta(S, mid, T, r, sigma) > target_delta: low = mid
        else: high = mid
    return round(mid / 500) * 500

def run_multi_regime_test():
    print("Downloading BTC-USD data (2019-2026)...")
    df = yf.download('BTC-USD', start='2019-01-01', end='2026-03-13', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RV_20'] = df['returns'].rolling(20).std() * np.sqrt(365)
    df['Simulated_IV'] = (df['RV_20'] + 0.15).clip(lower=0.50)
    df = df.dropna()

    # --- CONFIG ---
    initial_cash = 100000
    leverage = 1.5
    margin_interest_annual = 0.12
    daily_margin_rate = margin_interest_annual / 365.0
    
    # State tracking
    total_btc_notional = initial_cash * leverage
    margin_debt = total_btc_notional - initial_cash
    shares_owned = total_btc_notional / df['Close'].iloc[0]
    shares_covered = shares_owned * 0.50
    
    active_call, fiat_pnl = None, 0.0
    daily_stats = []

    # Regime Definitions
    regimes = [
        ('2019-01-01', '2020-09-30', 'Accumulation'),
        ('2020-10-01', '2021-05-15', 'Parabolic Bull'),
        ('2021-05-16', '2022-04-30', 'Volatile Dist.'),
        ('2022-05-01', '2023-01-15', 'Crypto Winter'),
        ('2023-01-16', '2026-03-13', 'Institutional Recov.')
    ]

    for date in df.index:
        price, high, iv = float(df.loc[date, 'Close']), float(df.loc[date, 'High']), float(df.loc[date, 'Simulated_IV'])
        
        # Apply Daily Margin Interest
        fiat_pnl -= margin_debt * daily_margin_rate
        
        opt_pnl_today = 0.0
        
        if active_call:
            T = max((active_call['exp'] - date).days, 1) / 365.0
            h_delta = call_delta(high, active_call['strike'], T, 0.04, iv)
            reason = None
            if h_delta >= 0.80: reason, exec_p = 'Defense', high
            elif (active_call['exp'] - date).days <= 0: reason, exec_p = 'Expiration', price

            if reason:
                exit_mid = black_scholes_call(exec_p, active_call['strike'], T, 0.04, iv)
                exit_ask = exit_mid * 1.025
                if reason == 'Defense': exit_ask *= 1.05
                opt_pnl_today = (active_call['entry_bid'] - exit_ask) * shares_covered
                fiat_pnl += opt_pnl_today
                active_call = None

        if not active_call:
            T_entry = 30 / 365.0
            strike = find_strike_for_delta(price, T_entry, 0.04, iv, 0.20)
            entry_mid = black_scholes_call(price, strike, T_entry, 0.04, iv)
            active_call = {'exp': date + timedelta(days=30), 'strike': strike, 
                           'entry_mid': entry_mid, 'entry_bid': entry_mid * 0.975}

        equity = (shares_owned * price) - margin_debt + fiat_pnl
        
        # Get Current Regime
        current_regime = "Unknown"
        for start, end, name in regimes:
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                current_regime = name
                break
                
        daily_stats.append({
            'date': date,
            'price': price,
            'equity': equity,
            'opt_pnl': opt_pnl_today,
            'regime': current_regime
        })

    df_results = pd.DataFrame(daily_stats).set_index('date')
    
    print("\n" + "="*70)
    print(f"{'BTC REGIME ANALYSIS (1.5x LEVERAGED COVERED CALL)':^70}")
    print("="*70)
    print(f"{'Regime':<20} | {'BTC Ret':<10} | {'Strat Ret':<10} | {'Opt PnL':<12}")
    print("-" * 70)

    for start, end, name in regimes:
        regime_df = df_results.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if regime_df.empty: continue
        
        btc_start, btc_end = regime_df['price'].iloc[0], regime_df['price'].iloc[-1]
        btc_ret = (btc_end / btc_start - 1) * 100
        
        strat_start, strat_end = regime_df['equity'].iloc[0], regime_df['equity'].iloc[-1]
        strat_ret = (strat_end / strat_start - 1) * 100
        
        opt_cash = regime_df['opt_pnl'].sum()
        
        print(f"{name:<20} | {btc_ret:>8.1f}% | {strat_ret:>8.1f}% | ${opt_cash:>10,.0f}")

    total_return = (df_results['equity'].iloc[-1] / initial_cash - 1) * 100
    
    # Calculate Sharpe Ratios
    df_results['strat_pct'] = df_results['equity'].pct_change()
    df_results['bnh_pct'] = df_results['price'].pct_change()
    
    # Annualized Sharpe (365 days for BTC)
    strat_sharpe = np.sqrt(365) * (df_results['strat_pct'].mean() / df_results['strat_pct'].std())
    bnh_sharpe = np.sqrt(365) * (df_results['bnh_pct'].mean() / df_results['bnh_pct'].std())

    print("-" * 70)
    print(f"{'TOTAL (7 YEARS)':<20} | {((df_results['price'].iloc[-1]/df_results['price'].iloc[0])-1)*100:>8.1f}% | {total_return:>8.1f}% | ${df_results['opt_pnl'].sum():>10,.0f}")
    print("="*70)
    print(f"Strategy Sharpe Ratio: {strat_sharpe:.2f}")
    print(f"Buy & Hold Sharpe:     {bnh_sharpe:.2f}")
    print(f"Sharpe Alpha:          {strat_sharpe - bnh_sharpe:+.2f}")
    print("="*70)

if __name__ == "__main__":
    run_multi_regime_test()
