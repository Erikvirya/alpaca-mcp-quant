import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import timedelta
import os

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

def run_ibit_databento_backtest():
    file_path = "market_data/universe/IBIT_1m.parquet"
    print(f"Loading IBIT data from {file_path}...")
    df = pd.read_parquet(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to Daily
    df_daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    df_daily['returns'] = np.log(df_daily['close'] / df_daily['close'].shift(1))
    # For ETF, 252 trading days is standard
    df_daily['RV_20'] = df_daily['returns'].rolling(20).std() * np.sqrt(252)
    # ETFs tracking BTC still have high IV
    df_daily['Simulated_IV'] = (df_daily['RV_20'] + 0.10).clip(lower=0.40)
    df_daily = df_daily.dropna()

    # --- IBKR CONFIG ---
    initial_cash = 100000
    leverage = 1.5
    margin_interest_annual = 0.065 # 6.5% IBKR Pro
    opt_commission = 0.65
    slippage_pct = 0.0125 # 1.25% slippage
    
    total_notional = initial_cash * leverage
    margin_debt = total_notional - initial_cash
    shares_owned = total_notional / df_daily['close'].iloc[0]
    shares_covered = shares_owned * 0.50
    
    active_call, fiat_pnl = None, 0.0
    total_opt_pnl, trade_log = 0.0, []
    daily_equity = []

    for date in df_daily.index:
        price, high, iv = float(df_daily.loc[date, 'close']), float(df_daily.loc[date, 'high']), float(df_daily.loc[date, 'Simulated_IV'])
        
        # IBKR Margin Interest (Daily)
        fiat_pnl -= margin_debt * (margin_interest_annual / 365.0)
        
        if active_call:
            T = max((active_call['exp'] - date).days, 1) / 365.0
            h_delta = call_delta(high, active_call['strike'], T, 0.04, iv)
            reason = None
            if h_delta >= 0.80: reason, exec_p = 'Defense', high
            else:
                cur_mid = black_scholes_call(price, active_call['strike'], T, 0.04, iv)
                # Adding 50% TP as requested in previous "refined" context
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
    
    print(f"\n--- IBIT DATABENTO BACKTEST (1 YEAR, 1.5x LEVERAGE) ---")
    print(f"Period:                {df_daily.index[0].date()} to {df_daily.index[-1].date()}")
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
    run_ibit_databento_backtest()
