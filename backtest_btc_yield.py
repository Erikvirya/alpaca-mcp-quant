import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import timedelta
import os

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

def call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1, 0.0, 1.0)

def find_strike_for_delta(S, T, r, sigma, target_delta=0.20):
    low, high = S * 0.5, S * 5.0
    for _ in range(30):
        mid = (low + high) / 2
        if call_delta(S, mid, T, r, sigma) > target_delta: low = mid
        else: high = mid
    return round(mid / 500) * 500

def run_btc_cc_backtest():
    print("Downloading BTC-USD data...")
    df = yf.download('BTC-USD', start='2021-01-01', end='2026-03-13', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RV_20'] = df['returns'].rolling(20).std() * np.sqrt(365)
    df['Simulated_IV'] = (df['RV_20'] + 0.15).clip(lower=0.50) # Crypto IV is much higher
    df = df.dropna()

    starting_capital = 100000
    shares_owned = starting_capital / df['Close'].iloc[0]
    shares_covered = shares_owned * 0.50
    
    target_dte, target_delta, r = 30, 0.20, 0.04
    take_profit_pct, delta_defense, slippage_pct = 0.50, 0.80, 0.025 # 2.5% slippage from mid (5% spread)
    
    active_call, fiat_pnl = None, 0.0
    trade_log, daily_equity = [], []

    for date in df.index:
        price, high, iv = float(df.loc[date, 'Close']), float(df.loc[date, 'High']), float(df.loc[date, 'Simulated_IV'])
        
        if active_call:
            T = max((active_call['exp'] - date).days, 1) / 365.0
            h_delta = call_delta(high, active_call['strike'], T, r, iv)
            reason = None
            if h_delta >= delta_defense: reason, exec_p = 'Defense', high
            else:
                cur_mid = black_scholes_call(price, active_call['strike'], T, r, iv)
                if cur_mid <= active_call['entry_mid'] * take_profit_pct: reason, exec_p = 'TP 50%', price
                elif (active_call['exp'] - date).days <= 0: reason, exec_p = 'Exp', price

            if reason:
                exit_mid = black_scholes_call(exec_p, active_call['strike'], T, r, iv)
                exit_ask = exit_mid * (1 + slippage_pct)
                # Extra 5% panic slippage on defensive triggers
                if reason == 'Defense': exit_ask *= 1.05
                
                pnl = (active_call['entry_bid'] - exit_ask) * shares_covered
                fiat_pnl += pnl
                trade_log.append({'pnl': pnl})
                active_call = None

        if not active_call:
            T_entry = target_dte / 365.0
            strike = find_strike_for_delta(price, T_entry, r, iv, target_delta)
            entry_mid = black_scholes_call(price, strike, T_entry, r, iv)
            active_call = {'exp': date + timedelta(days=target_dte), 'strike': strike, 
                           'entry_mid': entry_mid, 'entry_bid': entry_mid * (1 - slippage_pct)}

        daily_equity.append({'date': date, 'equity': (shares_owned * price) + fiat_pnl})

    df_res = pd.DataFrame(daily_equity).set_index('date')
    total_opt_pnl = sum(t['pnl'] for t in trade_log)
    years = (df_res.index[-1] - df_res.index[0]).days / 365.25
    
    print("\n--- BTC 5-YEAR INCOME SUMMARY ---")
    print(f"Total Options Cashflow: ${total_opt_pnl:,.2f}")
    print(f"Annualized Yield Alpha: {(total_opt_pnl / starting_capital / years) * 100:.2f}%")
    print(f"Total Trades:           {len(trade_log)}")

if __name__ == "__main__":
    run_btc_cc_backtest()
