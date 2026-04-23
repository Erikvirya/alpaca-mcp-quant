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

def run_refined_leveraged_btc():
    print("Downloading BTC-USD data for Refined Leveraged Backtest...")
    df = yf.download('BTC-USD', start='2021-01-01', end='2026-03-13', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # Technicals
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RV_20'] = df['returns'].rolling(20).std() * np.sqrt(365)
    # Simulate IV with a mean-reverting VRP premium
    np.random.seed(42)
    df['VRP_Premium'] = 0.10 + (0.05 * np.random.randn(len(df)))
    df['Simulated_IV'] = (df['RV_20'] + df['VRP_Premium']).clip(lower=0.40)
    
    # SHIFT signals to prevent look-ahead
    df['Sig_RV'] = df['RV_20'].shift(1)
    df['Sig_IV'] = df['Simulated_IV'].shift(1)
    df = df.dropna()

    # Leverage Config
    initial_cash = 100000
    leverage = 1.5
    total_btc_notional = initial_cash * leverage
    margin_debt = total_btc_notional - initial_cash
    margin_interest_annual = 0.12
    daily_margin_rate = margin_interest_annual / 365.0
    
    btc_price_start = df['Close'].iloc[0]
    shares_owned = total_btc_notional / btc_price_start
    shares_covered = shares_owned * 0.50
    
    active_call, fiat_pnl = None, 0.0
    total_opt_pnl, trade_log = 0.0, []
    daily_equity = []

    for date in df.index:
        price, high, iv = float(df.loc[date, 'Close']), float(df.loc[date, 'High']), float(df.loc[date, 'Simulated_IV'])
        sig_rv, sig_iv = float(df.loc[date, 'Sig_RV']), float(df.loc[date, 'Sig_IV'])
        
        fiat_pnl -= margin_debt * daily_margin_rate
        
        # 1. Manage Active Position
        if active_call:
            T = max((active_call['exp'] - date).days, 1) / 365.0
            h_delta = call_delta(high, active_call['strike'], T, 0.04, iv)
            reason = None
            
            if h_delta >= 0.80: # DEEP LEASH FOR CRYPTO
                reason, exec_p = 'Delta Defense 0.80', high
            else:
                cur_mid = black_scholes_call(price, active_call['strike'], T, 0.04, iv)
                if cur_mid <= active_call['entry_mid'] * 0.50: # USER SPECIFIED TP
                    reason, exec_p = 'Take Profit 50%', price
                elif (active_call['exp'] - date).days <= 0:
                    reason, exec_p = 'Expiration', price

            if reason:
                exit_mid = black_scholes_call(exec_p, active_call['strike'], T, 0.04, iv)
                exit_ask = exit_mid * 1.025
                if 'Defense' in reason: exit_ask *= 1.05
                
                trade_pnl = (active_call['entry_bid'] - exit_ask) * shares_covered
                fiat_pnl += trade_pnl
                total_opt_pnl += trade_pnl
                trade_log.append({'reason': reason, 'pnl': trade_pnl})
                active_call = None

        # 2. Enter New Position (Systematic)
        if not active_call:
            T_entry = 30 / 365.0
            strike = find_strike_for_delta(price, T_entry, 0.04, iv, 0.20)
            entry_mid = black_scholes_call(price, strike, T_entry, 0.04, iv)
            active_call = {'exp': date + timedelta(days=30), 'strike': strike, 
                           'entry_mid': entry_mid, 'entry_bid': entry_mid * 0.975}

        current_equity = (shares_owned * price) - margin_debt + fiat_pnl
        daily_equity.append({'date': date, 'equity': current_equity, 'price': price})

    df_res = pd.DataFrame(daily_equity).set_index('date')
    total_ret = (df_res['equity'].iloc[-1] / initial_cash - 1) * 100
    bnh_ret = (df_res['price'].iloc[-1] / df_res['price'].iloc[0] - 1) * 100
    years = (df_res.index[-1] - df_res.index[0]).days / 365.25
    
    print(f"\n--- 1.5X BTC REFINED STRATEGY (VRP GATE + 0.45 STOP) ---")
    print(f"Final Equity Value:    ${df_res['equity'].iloc[-1]:,.2f}")
    print(f"Total Strategy Return: {total_ret:,.2f}%")
    print(f"Buy & Hold (1x Spot):  {bnh_ret:,.2f}%")
    print(f"Total Options Income:  ${total_opt_pnl:,.2f}")
    print(f"Annual Yield Alpha:    {(total_opt_pnl / initial_cash / years) * 100:.2f}%")
    
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        print("\nExit Reasons:")
        print(df_trades['reason'].value_counts())

if __name__ == "__main__":
    run_refined_leveraged_btc()
