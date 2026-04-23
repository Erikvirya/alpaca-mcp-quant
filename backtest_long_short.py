import pandas as pd
import vectorbt as vbt
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import time

def run_long_short_backtest():
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

    print("Building Global Narrative Model for Futures Universe...")
    for name, ticker in universe.items():
        try:
            time.sleep(1)
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
                asset_meta[name] = {'price': close, 'test': combined.iloc[split:]}
                print(f"  - Processed {name}")
                
        except Exception as e:
            print(f"  - Skipping {name}: {e}")

    if not all_train_data:
        print("No data collected.")
        return
        
    full_train = pd.concat(all_train_data)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=5, eval_metric='AUC', random_seed=42, verbose=False)
    model.fit(full_train.drop('target', axis=1), full_train['target'])

    results = []
    for name, info in asset_meta.items():
        test_df = info['test']
        X_test = test_df.drop('target', axis=1)
        price_test = info['price'].reindex(test_df.index)
        
        probs = model.predict_proba(X_test)[:, 1]
        
        long_entries = (probs > 0.51)
        long_exits = (probs < 0.49)
        
        short_entries = (probs < 0.49)
        short_exits = (probs > 0.51)
        
        pf_lo = vbt.Portfolio.from_signals(price_test, entries=long_entries, exits=long_exits, freq='1D')
        pf_ls = vbt.Portfolio.from_signals(price_test, entries=long_entries, exits=long_exits, short_entries=short_entries, short_exits=short_exits, freq='1D')
        
        results.append({
            'Asset': name,
            'LO_Return': f"{pf_lo.total_return()*100:.1f}%",
            'LS_Return': f"{pf_ls.total_return()*100:.1f}%",
            'LO_Sharpe': f"{pf_lo.sharpe_ratio():.2f}",
            'LS_Sharpe': f"{pf_ls.sharpe_ratio():.2f}",
            'LO_Win%': f"{pf_lo.trades.win_rate()*100:.1f}%",
            'LS_Win%': f"{pf_ls.trades.win_rate()*100:.1f}%",
            'LO_Trades': pf_lo.trades.count(),
            'LS_Trades': pf_ls.trades.count()
        })

    print("\n" + "="*85)
    print(" LONG-ONLY vs LONG-SHORT NARRATIVE STRATEGY OOS BACKTEST ")
    print("="*85)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
if __name__ == "__main__":
    run_long_short_backtest()
