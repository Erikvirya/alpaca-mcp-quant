import pandas as pd
import vectorbt as vbt
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import time
import os

def run_skeptical_hybrid_test():
    universe = {
        'ES_M (S&P 500)': 'SPY',
        'NQ_M (Nasdaq 100)': 'QQQ',
        'YM_M (Dow Jones)': 'DIA',
        'RTY_M (Russell 2k)': 'IWM',
        'FDAX_M (DAX)': 'EWG',
        'FESX_M (EuroStoxx)': 'FEZ',
        '6E_M (Euro)': 'FXE',
        '6B_M (GBP)': 'FXB',
        '6J_M (JPY)': 'FXY',
        '6A_M (AUD)': 'FXA',
        'GC_M (Gold)': 'GLD',
        'SI_N (Silver)': 'SLV',
        'CL_M (Crude Oil)': 'USO',
        'NG_M (Nat Gas)': 'UNG',
        'ZN_M (10yr Note)': 'IEF'
    }
    
    LONG_SHORT_ASSETS = ['GC_M (Gold)', 'FDAX_M (DAX)', 'FESX_M (EuroStoxx)', 'RTY_M (Russell 2k)', '6E_M (Euro)', '6B_M (GBP)', '6A_M (AUD)', '6J_M (JPY)', 'YM_M (Dow Jones)']

    print("="*80)
    print(" SKEPTICAL QUANT REVIEW: HYBRID LONG/SHORT PORTFOLIO ")
    print("="*80)
    print("Constraints:")
    print("1. 10-Day Target Horizon.")
    print("2. Thresholds: Long > 0.60 | Short < 0.40")
    print("3. Look-Ahead Bias Removal (1-day GDELT lag).")
    print("4. Transaction Costs (2 bps / 0.02%).")
    print("5. Data Corruption (5% drop).")
    print("6. Portfolio Risk Management (Cash Sharing, Normalized Sizing).")
    print("="*80)

    try:
        df_gdelt = pd.read_csv('gdelt_vol_tone_final.csv', index_col='date')
        df_gdelt.index = pd.to_datetime(df_gdelt.index).tz_localize(None).normalize()
        if 'recession_tone' not in df_gdelt.columns:
            df_gdelt['recession_tone'] = 0.0
            
        np.random.seed(42)
        drop_indices = np.random.choice(df_gdelt.index, size=int(len(df_gdelt)*0.05), replace=False)
        df_gdelt.loc[drop_indices] = np.nan
        df_gdelt = df_gdelt.ffill() 
    except Exception as e:
        print(f"Error loading GDELT: {e}")
        return

    all_train_data = []
    asset_meta = {}

    for name, ticker in universe.items():
        try:
            time.sleep(0.5)
            data = vbt.YFData.download(ticker, start='2023-08-23').get()
            close = data['Close']
            close.index = close.index.tz_localize(None).normalize()
            
            feats = pd.DataFrame(index=close.index)
            for col in df_gdelt.columns:
                val = df_gdelt[col].reindex(close.index).ffill().shift(1) 
                feats[f'{col}_raw'] = val
                rolling_mean = val.rolling(60).mean()
                rolling_std = val.rolling(60).std().replace(0, 0.001)
                feats[f'{col}_z'] = (val - rolling_mean) / rolling_std
                if 'tone' in col:
                    feats[f'{col}_lag2'] = val.shift(2)
            
            feats['rsi'] = vbt.RSI.run(close, window=14).rsi
            feats['vol'] = close.pct_change().rolling(20).std()
            
            target = (close.pct_change(10).shift(-10) > 0).astype(int)
            combined = pd.concat([feats, target.rename('target')], axis=1).dropna()
            
            if not combined.empty:
                split = int(len(combined) * 0.8)
                all_train_data.append(combined.iloc[:split])
                asset_meta[name] = {'price': close, 'feats': combined.drop('target', axis=1), 'split_idx': split}
        except Exception as e:
            pass

    if not all_train_data:
        return
        
    full_train = pd.concat(all_train_data)
    print(f"Training Skeptical Model on {len(full_train)} historical samples...")
    model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=5, eval_metric='AUC', random_seed=42, verbose=False)
    model.fit(full_train.drop('target', axis=1), full_train['target'])

    price_df = pd.DataFrame()
    target_weights_df = pd.DataFrame()

    for name, info in asset_meta.items():
        split = info['split_idx']
        test_feats = info['feats'].iloc[split:]
        test_price = info['price'].reindex(test_feats.index)
        
        probs = model.predict_proba(test_feats)[:, 1]
        
        weights = pd.Series(0.0, index=test_price.index)
        
        if name in LONG_SHORT_ASSETS:
            weights[probs > 0.60] = 1.0
            weights[probs < 0.40] = -1.0
        else:
            weights[probs > 0.60] = 1.0
            
        hist_vol = test_price.pct_change().rolling(20).std() * np.sqrt(252)
        target_vol = 0.10
        vol_scaler = (target_vol / hist_vol).ffill().fillna(1.0)
        
        weights = weights * vol_scaler
        
        price_df[name] = test_price
        target_weights_df[name] = weights

    normalized_weights_df = target_weights_df / len(universe)
    
    fees = 0.0002 
    
    pf_portfolio = vbt.Portfolio.from_orders(
        price_df,
        size=normalized_weights_df,
        size_type='targetpercent',
        freq='1D',
        init_cash=100000,
        cash_sharing=True,
        group_by=True,
        fees=fees 
    )
    
    print("\n--- SKEPTICAL HYBRID OOS RESULTS ---")
    stats = pf_portfolio.stats()
    print(stats[['Start', 'End', 'Total Return [%]', 'Max Drawdown [%]', 'Total Trades', 'Win Rate [%]', 'Profit Factor', 'Sharpe Ratio']])
    
    print("\n--- Individual Asset Contributions ---")
    pf_indiv = vbt.Portfolio.from_orders(
        price_df,
        size=normalized_weights_df,
        size_type='targetpercent',
        freq='1D',
        init_cash=100000,
        fees=fees
    )
    results = []
    for name in price_df.columns:
        sub_pf = pf_indiv[name]
        results.append({
            'Asset': name,
            'Type': 'L/S' if name in LONG_SHORT_ASSETS else 'L-Only',
            'Return': f"{sub_pf.total_return()*100:.1f}%",
            'Win%': f"{sub_pf.trades.win_rate()*100:.1f}%",
            'Sharpe': f"{sub_pf.sharpe_ratio():.2f}",
            'Trades': sub_pf.trades.count(),
        })
    print(pd.DataFrame(results).to_string(index=False))
    
    print("\n--- MONTE CARLO ROBUSTNESS (1000 Shuffles of Trade Returns) ---")
    trades = pf_portfolio.trades.records_readable
    if len(trades) > 0:
        returns = trades['Return'].values
        mc_totals = []
        for _ in range(1000):
            shuffled = np.random.choice(returns, size=len(returns), replace=True)
            mc_totals.append((1 + shuffled).prod() - 1)
            
        mc_totals = np.sort(np.array(mc_totals))
        p5 = mc_totals[int(0.05 * 1000)] * 100
        p50 = mc_totals[int(0.50 * 1000)] * 100
        p95 = mc_totals[int(0.95 * 1000)] * 100
        
        print(f"5th Percentile (Worst Case):  {p5:.2f}%")
        print(f"50th Percentile (Median):     {p50:.2f}%")
        print(f"95th Percentile (Best Case):  {p95:.2f}%")
        if p5 < 0:
            print("WARNING: Strategy edge breaks down under bootstrap resampling.")
        else:
            print("PASS: Strategy maintains positive edge even in worst 5% of simulated sequences.")
    else:
        print("Not enough trades for Monte Carlo.")

if __name__ == "__main__":
    run_skeptical_hybrid_test()
