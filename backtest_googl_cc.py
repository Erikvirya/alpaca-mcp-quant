import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from datetime import timedelta, date
import os

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call

def call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1, 0.0, 1.0)

def find_strike_for_delta(S, T, r, sigma, target_delta=0.20):
    low = S * 0.5
    high = S * 3.0
    for _ in range(30):
        mid = (low + high) / 2
        d = call_delta(S, mid, T, r, sigma)
        if d > target_delta:
            low = mid
        else:
            high = mid
    return round(mid, 2)

from edgar import Company, set_identity
set_identity('trader@multistrat.com')

def fetch_earnings_dates(symbol):
    try:
        company = Company(symbol)
        filings = company.get_filings(form=["10-Q", "10-K"])
        if filings:
            df = filings.to_pandas()
            return pd.to_datetime(df['filing_date']).tolist()
    except Exception as e:
        print(f"Error fetching earnings for {symbol}: {e}")
    return []

def run_googl_cc_backtest():
    ticker_symbol = 'GOOGL'
    print(f"Downloading {ticker_symbol} historical data...")
    ticker = yf.Ticker(ticker_symbol)
    df_stock = ticker.history(start='2016-03-13', end='2026-03-13', interval='1d')
    df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
    
    print("Fetching Earnings Dates from SEC EDGAR...")
    earnings_dates = fetch_earnings_dates(ticker_symbol)
    if not earnings_dates:
        print("Warning: No earnings dates found from SEC. Check your identity/connection.")
    else:
        print(f"Found {len(earnings_dates)} SEC filing dates.")

    # Calculate RV (21-day for equities)
    df_stock['returns'] = np.log(df_stock['Close'] / df_stock['Close'].shift(1))
    df_stock['RV_21'] = df_stock['returns'].rolling(21).std() * np.sqrt(252)
    
    # Simple IV proxy (Equities usually have RV < IV slightly, or specific skew)
    df_stock['Simulated_IV'] = df_stock['RV_21'] + 0.05
    df_stock['Simulated_IV'] = df_stock['Simulated_IV'].clip(lower=0.15) # Floor for GOOGL
    df_stock = df_stock.dropna()
    
    starting_capital = 100000
    current_date = df_stock.index[0]
    initial_stock_price = df_stock.loc[current_date, 'Close']
    shares_owned = starting_capital / initial_stock_price
    shares_covered = shares_owned * 0.50 # User specified 50% coverage
    
    target_dte_days = 30
    target_delta = 0.20 # User specified
    risk_free_rate = 0.04
    
    take_profit_pct = 0.50 # User specified
    delta_defense_threshold = 0.45 # User specified
    time_stop_dte_days = 7 # Standard time stop for equities
    
    # Friction
    opt_commission = 0.65 # $0.65 per contract
    slippage_pct = 0.0125 # 1.25% slippage from mid (Matches a 2.5% Total Bid/Ask Spread)
    
    active_call = None
    fiat_pnl = 0.0
    total_slippage_paid = 0.0
    
    print(f"Starting capital: ${starting_capital}")
    print(f"Purchased {shares_owned:.2f} {ticker_symbol} at ${initial_stock_price:.2f}")
    print(f"Coverage: {shares_covered:.2f} shares")
    print(f"Slippage Model: 1.25% per fill (2.5% Total Spread)\n")
    
    trade_log = []
    daily_equity = []
    
    for date_idx in df_stock.index:
        stock_price = float(df_stock.loc[date_idx, 'Close'])
        high_price = float(df_stock.loc[date_idx, 'High'])
        iv = float(df_stock.loc[date_idx, 'Simulated_IV'])
        
        # 1. Manage Active Position
        if active_call is not None:
            days_to_expiry = (active_call['expiration'] - date_idx).days
            T = max(days_to_expiry, 1) / 365.0
            
            high_delta = call_delta(high_price, active_call['strike'], T, risk_free_rate, iv)
            
            reason = None
            exec_price = stock_price
            
            if high_delta >= delta_defense_threshold:
                reason = 'Delta Defense 0.45'
                exec_price = high_price
            else:
                cur_price = black_scholes_call(stock_price, active_call['strike'], T, risk_free_rate, iv)
                if cur_price <= active_call['entry_price_mid'] * take_profit_pct:
                    reason = 'Take Profit 50%'
                elif days_to_expiry <= 0:
                    reason = 'Expiration'
                elif days_to_expiry <= time_stop_dte_days:
                    reason = 'Time Stop'
            
            if reason:
                exit_val_mid = black_scholes_call(exec_price, active_call['strike'], T, risk_free_rate, iv)
                # EXIT SLIPPAGE: We must pay the Ask to buy back (Mid + 3%)
                slip = exit_val_mid * slippage_pct
                exit_val_ask = exit_val_mid + slip
                total_slippage_paid += (slip * shares_covered)
                
                contracts = shares_covered / 100.0
                trade_pnl_per_share = active_call['entry_price_net'] - exit_val_ask
                total_pnl = (trade_pnl_per_share * shares_covered) - (opt_commission * 2 * contracts)
                
                fiat_pnl += total_pnl
                trade_log.append({
                    'entry_date': active_call['entry_date'],
                    'exit_date': date_idx,
                    'strike': active_call['strike'],
                    'reason': reason,
                    'pnl': total_pnl
                })
                active_call = None

        # 2. Enter New Position
        if active_call is None:
            is_near_earnings = False
            for e_date in earnings_dates:
                diff_days = (e_date - date_idx).days
                if 0 <= diff_days <= 35:
                    is_near_earnings = True
                    break
            
            if not is_near_earnings:
                T_entry = target_dte_days / 365.0
                strike = find_strike_for_delta(stock_price, T_entry, risk_free_rate, iv, target_delta)
                
                if strike > stock_price:
                    entry_val_mid = black_scholes_call(stock_price, strike, T_entry, risk_free_rate, iv)
                    # ENTRY SLIPPAGE: We must sell at the Bid (Mid - 3%)
                    slip = entry_val_mid * slippage_pct
                    entry_val_bid = entry_val_mid - slip
                    total_slippage_paid += (slip * shares_covered)
                    
                    active_call = {
                        'entry_date': date_idx,
                        'expiration': date_idx + timedelta(days=target_dte_days),
                        'strike': strike,
                        'entry_price_mid': entry_val_mid,
                        'entry_price_net': entry_val_bid
                    }
        
        # Equity Tracking
        unrealized_stock_val = shares_owned * stock_price
        opt_liability = 0
        if active_call:
            days_to_expiry = max((active_call['expiration'] - date_idx).days, 1)
            T_val = days_to_expiry / 365.0
            cur_val = black_scholes_call(stock_price, active_call['strike'], T_val, risk_free_rate, iv)
            opt_liability = (cur_val - active_call['entry_price_net']) * shares_covered
            
        total_equity = unrealized_stock_val + fiat_pnl - opt_liability
        daily_equity.append({
            'date': date_idx,
            'equity': total_equity,
            'stock_price': stock_price
        })

    df_res = pd.DataFrame(daily_equity).set_index('date')
    df_res['strat_ret'] = df_res['equity'].pct_change()
    df_res['bnh_ret'] = df_res['stock_price'].pct_change()
    
    # Normalize for comparison
    df_res['Strategy_Equity'] = (df_res['equity'] / starting_capital) * 100
    df_res['Buy_and_Hold_Equity'] = (df_res['stock_price'] / df_res['stock_price'].iloc[0]) * 100

    print("\n--- RESULTS ---")
    print(f"Total Return: {(df_res['equity'].iloc[-1]/starting_capital - 1)*100:.2f}%")
    print(f"B&H Return: {(df_res['stock_price'].iloc[-1]/df_res['stock_price'].iloc[0] - 1)*100:.2f}%")
    print(f"Strategy Sharpe: {np.sqrt(252) * (df_res['strat_ret'].mean()/df_res['strat_ret'].std()):.2f}")
    print(f"B&H Sharpe: {np.sqrt(252) * (df_res['bnh_ret'].mean()/df_res['bnh_ret'].std()):.2f}")
    
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        print("\nExit Reasons:")
        print(df_trades['reason'].value_counts())
        print(f"\nTotal Option PnL: ${df_trades['pnl'].sum():,.2f}")

        # Monthly Cash Flow Summary
        df_trades['exit_month'] = df_trades['exit_date'].dt.to_period('M')
        monthly_cash_flow = df_trades.groupby('exit_month')['pnl'].sum()
        
        # Fill missing months with 0 (months we avoided due to earnings or no trades)
        all_months = pd.period_range(start=df_res.index.min(), end=df_res.index.max(), freq='M')
        monthly_cash_flow = monthly_cash_flow.reindex(all_months, fill_value=0)

        print("\n--- MONTHLY CASH FLOW SUMMARY ---")
        print(f"Average Monthly Cash Flow:  ${monthly_cash_flow.mean():.2f}")
        print(f"Median Monthly Cash Flow:   ${monthly_cash_flow.median():.2f}")
        print(f"Best Month:                 ${monthly_cash_flow.max():.2f} ({monthly_cash_flow.idxmax()})")
        print(f"Worst Month:                ${monthly_cash_flow.min():.2f} ({monthly_cash_flow.idxmin()})")
        print(f"Profitable Months:          {(monthly_cash_flow > 0).sum()} / {len(monthly_cash_flow)} ({((monthly_cash_flow > 0).sum()/len(monthly_cash_flow))*100:.1f}%)")
        print(f"Total Slippage Paid:        ${total_slippage_paid:,.2f}")

    # Plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(df_res.index, df_res['Strategy_Equity'], label='Covered Call Strategy (0.2 Delta / 0.45 Stop)', color='blue')
        plt.plot(df_res.index, df_res['Buy_and_Hold_Equity'], label='Buy & Hold GOOGL', color='gray', alpha=0.6)
        plt.title(f'GOOGL: Covered Call Strategy vs Buy & Hold (10-Year Backtest)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Equity (Start = 100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = 'googl_cc_vs_bnh.png'
        plt.savefig(plot_path)
        print(f"\nEquity graph saved to: {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

    # Save data
    os.makedirs('market_data', exist_ok=True)
    df_res[['Strategy_Equity', 'Buy_and_Hold_Equity']].to_csv('market_data/googl_cc_comparison.csv')
    print("Equity data saved to: market_data/googl_cc_comparison.csv")

if __name__ == "__main__":
    run_googl_cc_backtest()
