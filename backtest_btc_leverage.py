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

def run_leveraged_btc_backtest():
    print("Downloading BTC-USD data for Leveraged Backtest...")
    df = yf.download('BTC-USD', start='2021-01-01', end='2026-03-13', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RV_20'] = df['returns'].rolling(20).std() * np.sqrt(365)
    df['Simulated_IV'] = (df['RV_20'] + 0.15).clip(lower=0.50)
    df = df.dropna()

    # --- LEVERAGE CONFIG ---
    initial_cash = 100000
    leverage = 1.5
    total_btc_notional = initial_cash * leverage # $150,000 exposure
    margin_debt = total_btc_notional - initial_cash # $50,000 borrowed
    margin_interest_annual = 0.12 # 12% APR
    daily_margin_rate = margin_interest_annual / 365.0
    
    btc_price_start = df['Close'].iloc[0]
    shares_owned = total_btc_notional / btc_price_start
    shares_covered = shares_owned * 0.50 # Covering 50% of the 1.5x position
    
    active_call, fiat_pnl = None, 0.0
    total_opt_pnl = 0.0
    daily_equity = []
    liquidated = False

    for date in df.index:
        price, high, iv = float(df.loc[date, 'Close']), float(df.loc[date, 'High']), float(df.loc[date, 'Simulated_IV'])
        
        # Apply Daily Margin Interest
        fiat_pnl -= margin_debt * daily_margin_rate
        
        if active_call:
            T = max((active_call['exp'] - date).days, 1) / 365.0
            h_delta = call_delta(high, active_call['strike'], T, 0.04, iv)
            reason = None
            if h_delta >= 0.80: reason, exec_p = 'Defense', high
            else:
                cur_mid = black_scholes_call(price, active_call['strike'], T, 0.04, iv)
                if cur_mid <= active_call['entry_mid'] * 0.50: reason, exec_p = 'TP 50%', price
                elif (active_call['exp'] - date).days <= 0: reason, exec_p = 'Exp', price

            if reason:
                exit_mid = black_scholes_call(exec_p, active_call['strike'], T, 0.04, iv)
                exit_ask = exit_mid * 1.025 # 2.5% slippage
                if reason == 'Defense': exit_ask *= 1.05
                
                trade_pnl = (active_call['entry_bid'] - exit_ask) * shares_covered
                fiat_pnl += trade_pnl
                total_opt_pnl += trade_pnl
                active_call = None

        if not active_call:
            T_entry = 30 / 365.0
            strike = find_strike_for_delta(price, T_entry, 0.04, iv, 0.20)
            entry_mid = black_scholes_call(price, strike, T_entry, 0.04, iv)
            active_call = {'exp': date + timedelta(days=30), 'strike': strike, 
                           'entry_mid': entry_mid, 'entry_bid': entry_mid * 0.975}

        # Liquidation Check: If Equity < 10% of Notional (Maintenance Margin)
        current_equity = (shares_owned * price) - margin_debt + fiat_pnl
        if current_equity <= (total_btc_notional * 0.10):
            print(f"!!! LIQUIDATED on {date.date()} at ${price:,.2f} !!!")
            liquidated = True
            break
            
        daily_equity.append({'date': date, 'equity': current_equity, 'price': price})

    if not liquidated:
        df_res = pd.DataFrame(daily_equity).set_index('date')
        total_ret = (df_res['equity'].iloc[-1] / initial_cash - 1) * 100
        bnh_ret = (df_res['price'].iloc[-1] / df_res['price'].iloc[0] - 1) * 100
        years = (df_res.index[-1] - df_res.index[0]).days / 365.25
        
        print(f"\n--- 1.5X LEVERAGED BTC SUMMARY (2021-2026) ---")
        print(f"Final Equity Value:    ${df_res['equity'].iloc[-1]:,.2f}")
        print(f"Total Strategy Return: {total_ret:,.2f}%")
        print(f"Buy & Hold (1x Spot):  {bnh_ret:,.2f}%")
        print(f"Total Options Income:  ${total_opt_pnl:,.2f}")
        print(f"Net Yield Alpha:       {(total_opt_pnl / initial_cash / years) * 100:.2f}% (before interest)")
        print(f"Margin Interest Drag:  {(margin_debt * margin_interest_annual / initial_cash) * 100:.2f}% per year")

if __name__ == "__main__":
    run_leveraged_btc_backtest()
