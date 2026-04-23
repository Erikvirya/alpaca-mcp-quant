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
    return round(mid, 2)

def run_high_year_ibkr_proxy():
    # JAN 2023 to MAR 2024 (Institutional Recovery vertical rally)
    print("Downloading BTC data for 2023-2024 High-Growth Proxy...")
    df = yf.download('BTC-USD', start='2023-01-01', end='2024-03-31', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RV_20'] = df['returns'].rolling(20).std() * np.sqrt(365)
    df['Simulated_IV'] = (df['RV_20'] + 0.15).clip(lower=0.50)
    df = df.dropna()

    # --- IBKR CONFIG ---
    initial_cash = 100000
    leverage = 1.5
    margin_interest_annual = 0.065 # 6.5% IBKR Pro
    opt_commission = 0.65
    slippage_pct = 0.0125
    
    total_notional = initial_cash * leverage
    margin_debt = total_notional - initial_cash
    shares_owned = total_notional / df['Close'].iloc[0]
    shares_covered = shares_owned * 0.50
    
    active_call, fiat_pnl = None, 0.0
    total_opt_pnl, trade_log = 0.0, []
    daily_equity = []

    for date in df.index:
        price, high, iv = float(df.loc[date, 'Close']), float(df.loc[date, 'High']), float(df.loc[date, 'Simulated_IV'])
        fiat_pnl -= margin_debt * (margin_interest_annual / 365.0)
        
        if active_call:
            T = max((active_call['exp'] - date).days, 1) / 365.0
            h_delta = call_delta(high, active_call['strike'], T, 0.04, iv)
            reason = None
            if h_delta >= 0.80: reason, exec_p = 'Defense', high
            else:
                cur_mid = black_scholes_call(price, active_call['strike'], T, 0.04, iv)
                if cur_mid <= active_call['entry_mid'] * 0.50: reason, exec_p = 'TP 50%', price
                elif (active_call['exp'] - date).days <= 0: reason, exec_p = 'Expiration', price

            if reason:
                exit_mid = black_scholes_call(exec_p, active_call['strike'], T, 0.04, iv)
                exit_ask = exit_mid * (1 + slippage_pct)
                if reason == 'Defense': exit_ask *= 1.05
                
                contracts = shares_covered / 100.0
                trade_pnl = (active_call['entry_bid'] - exit_ask) * shares_covered - (opt_commission * 2 * contracts)
                fiat_pnl += trade_pnl
                total_opt_pnl += trade_pnl
                trade_log.append({'reason': reason, 'pnl': trade_pnl})
                active_call = None

        if not active_call:
            T_entry = 30 / 365.0
            strike = find_strike_for_delta(price, T_entry, 0.04, iv, 0.20)
            entry_mid = black_scholes_call(price, strike, T_entry, 0.04, iv)
            active_call = {'exp': date + timedelta(days=30), 'strike': strike, 
                           'entry_mid': entry_mid, 'entry_bid': entry_mid * (1 - slippage_pct)}

        equity = (shares_owned * price) - margin_debt + fiat_pnl
        daily_equity.append({'date': date, 'equity': equity, 'price': price})

    df_res = pd.DataFrame(daily_equity).set_index('date')
    total_ret = (df_res['equity'].iloc[-1] / initial_cash - 1) * 100
    bnh_ret = (df_res['price'].iloc[-1] / df_res['price'].iloc[0] - 1) * 100
    
    print(f"\n--- IBKR PROXY: VERTICAL BULL YEAR (2023-2024) ---")
    print(f"Period:                {df.index[0].date()} to {df.index[-1].date()}")
    print(f"BTC Start Price:       ${df['Close'].iloc[0]:,.2f}")
    print(f"BTC End Price:         ${df['Close'].iloc[-1]:,.2f} (+{bnh_ret:.1f}%)")
    print(f"Final Equity:          ${df_res['equity'].iloc[-1]:,.2f}")
    print(f"Total Strategy Return: {total_ret:,.2f}%")
    print(f"Buy & Hold (1x Spot):  {bnh_ret:,.2f}%")
    print(f"Net Options Cashflow:  ${total_opt_pnl:,.2f}")
    print(f"Alpha (vs B&H):        {total_ret - bnh_ret:.2f}%")
    
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        print("\nExit Reasons:")
        print(df_trades['reason'].value_counts())

if __name__ == "__main__":
    run_high_year_ibkr_proxy()
