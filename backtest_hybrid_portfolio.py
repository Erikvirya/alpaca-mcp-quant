import pandas as pd
import vectorbt as vbt
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import time

def run_hybrid_portfolio_backtest():
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
    
    # Hybrid Logic Definition
    LONG_SHORT_ASSETS = ['GC_M (Gold)', 'FDAX_M (DAX)', 'FESX_M (EuroStoxx)', 'RTY_M (Russell 2k)', '6E_M (Euro)', '6B_M (GBP)', '6A_M (AUD)', '6J_M (JPY)', 'YM_M (Dow Jones)']
    LONG_ONLY_ASSETS = ['ES_M (S&P 500)', 'NQ_M (Nasdaq 100)', 'SI_N (Silver)', 'CL_M (Crude Oil)', 'NG_M (Nat Gas)', 'ZN_M (10yr Note)']

    try:
        df_gdelt = pd.read_csv('gdelt_vol_tone_final.csv', index_col='date')
        df_gdelt.index = pd.to_datetime(df_gdelt.index).tz_localize(None).normalize()
        if 'recession_tone' not in df_gdelt.columns:
            df_gdelt['recession_tone'] = 0.0
    except Exception as e:
        print(f"Error loading GDELT: {e}")
        return

    all_train_data = []
    asset_meta = {}

    print("Building Global Narrative Model for Hybrid Portfolio...")
    for name, ticker in universe.items():
        try:
            time.sleep(0.5)
            data = vbt.YFData.download(ticker, start='2023-08-23').get()
            close = data['Close']
            close.index = close.index.tz_localize(None).normalize()
            
            feats = pd.DataFrame(index=close.index)
            for col in df_gdelt.columns:
                val = df_gdelt[col].reindex(close.index).ffill()
                feats[f'{col}_raw'] = val
                rolling_mean = val.rolling(60).mean()
                rolling_std = val.rolling(60).std().replace(0, 0.001)
                feats[f'{col}_z'] = (val - rolling_mean) / rolling_std
                if 'tone' in col:
                    feats[f'{col}_lag2'] = val.shift(2)
            
            feats['rsi'] = vbt.RSI.run(close, window=14).rsi
            feats['vol'] = close.pct_change().rolling(20).std()
            
            target = (close.pct_change(2).shift(-2) > 0).astype(int)
            combined = pd.concat([feats, target.rename('target')], axis=1).dropna()
            
            if not combined.empty:
                split = int(len(combined) * 0.8)
                all_train_data.append(combined.iloc[:split])
                asset_meta[name] = {'price': close, 'feats': combined.drop('target', axis=1), 'split_idx': split}
                
        except Exception as e:
            print(f"  - Skipping {name}: {e}")

    if not all_train_data:
        print("No data collected.")
        return
        
    full_train = pd.concat(all_train_data)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=5, eval_metric='AUC', random_seed=42, verbose=False)
    model.fit(full_train.drop('target', axis=1), full_train['target'])

    # Assemble Portfolio Dataframes for OOS
    price_df = pd.DataFrame()
    target_weights_df = pd.DataFrame()

    for name, info in asset_meta.items():
        split = info['split_idx']
        test_feats = info['feats'].iloc[split:]
        test_price = info['price'].reindex(test_feats.index)
        
        probs = model.predict_proba(test_feats)[:, 1]
        
        # Target Weights
        weights = pd.Series(0.0, index=test_price.index)
        if name in LONG_SHORT_ASSETS:
            weights[probs > 0.51] = 1.0
            weights[probs < 0.49] = -1.0
        else:
            weights[probs > 0.51] = 1.0
            # < 0.49 goes to 0 (flat)
            
        # Volatility Targeting Size
        hist_vol = test_price.pct_change().rolling(20).std() * np.sqrt(252)
        target_vol = 0.10
        vol_scaler = (target_vol / hist_vol).ffill().fillna(1.0)
        
        # Apply volatility scaling
        weights = weights * vol_scaler
        
        price_df[name] = test_price
        target_weights_df[name] = weights

    # Portfolio Level Risk Management
    num_assets = len(universe)
    normalized_weights_df = target_weights_df / num_assets
    
    # 1. Independent Asset Breakdown (No Cash Sharing)
    pf_indiv = vbt.Portfolio.from_orders(
        price_df,
        size=normalized_weights_df,
        size_type='targetpercent',
        freq='1D',
        init_cash=100000
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
        
    print("\n--- Individual Asset Contributions (Allocated Risk) ---")
    print(pd.DataFrame(results).to_string(index=False))
    
    # 2. Aggregate Portfolio Backtest (Shared Cash & Grouped)
    pf_portfolio = vbt.Portfolio.from_orders(
        price_df,
        size=normalized_weights_df,
        size_type='targetpercent',
        freq='1D',
        init_cash=100000,
        cash_sharing=True,
        group_by=True
    )
    
    print("\n" + "="*85)
    print(" HYBRID LONG/SHORT + SHARED CASH PORTFOLIO OOS BACKTEST ")
    print("="*85)
    print(pf_portfolio.stats())

if __name__ == "__main__":
    run_hybrid_portfolio_backtest()
