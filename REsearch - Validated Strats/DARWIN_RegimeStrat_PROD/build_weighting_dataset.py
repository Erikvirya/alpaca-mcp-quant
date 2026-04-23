import os
import pandas as pd
import numpy as np
import yfinance as yf

# --- CONFIG ---
START_DATE = '2015-01-01'
END_DATE = '2026-03-13'
DATA_DIR = "REsearch - Validated Strats/DARWIN_RegimeStrat_PROD/data"
os.makedirs(DATA_DIR, exist_ok=True)

SYMBOLS = {
    'NQ': {'ticker': 'NQ=F', 'vol': '^VXN'},
    'ES': {'ticker': 'ES=F', 'vol': '^VIX'},
    'YM': {'ticker': 'YM=F', 'vol': '^VXD'}
}

def build_weighting_dataset():
    print("Fetching data for all 3 assets...")
    data = {}
    for name, config in SYMBOLS.items():
        p_df = yf.download(config['ticker'], start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
        v_df = yf.download(config['vol'], start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)['Close']
        
        p_df.index = pd.to_datetime(p_df.index).tz_localize(None)
        v_df.index = pd.to_datetime(v_df.index).tz_localize(None)
        
        common = p_df.index.intersection(v_df.index)
        p_df = p_df.loc[common]
        v_df = v_df.loc[common]
        
        df = pd.DataFrame(index=common)
        df[f'{name}_Close'] = p_df['Close']
        df[f'{name}_Ret'] = p_df['Close'].pct_change()
        df[f'{name}_Vol'] = v_df
        # Features
        df[f'{name}_RSI'] = vbt.RSI.run(p_df['Close']).rsi
        df[f'{name}_EMA50_Dist'] = (p_df['Close'] - p_df['Close'].ewm(span=50).mean()) / p_df['Close']
        df[f'{name}_Mom20'] = p_df['Close'].pct_change(20)
        
        data[name] = df

    # Merge all
    full_df = pd.concat(data.values(), axis=1).dropna()
    
    print("Calculating Targets (Next-day Leader)...")
    # We want to find which asset has the best Return/Vol tomorrow
    for name in SYMBOLS:
        # Tomorrow's Sharpe-ish return
        # Use simple return / vol proxy
        full_df[f'{name}_Next_Score'] = (full_df[f'{name}_Ret'].shift(-1) / (full_df[f'{name}_Vol'] / 100.0))
    
    score_cols = [f'{name}_Next_Score' for name in SYMBOLS]
    full_df['Target_Asset'] = full_df[score_cols].idxmax(axis=1).str.replace('_Next_Score', '')
    
    # Map to Int
    mapping = {'NQ': 0, 'ES': 1, 'YM': 2}
    full_df['Target'] = full_df['Target_Asset'].map(mapping)
    
    # Final cleanup
    full_df = full_df.dropna()
    output_path = os.path.join(DATA_DIR, "multi_asset_weighting_dataset.csv")
    full_df.to_csv(output_path)
    print(f"Weighting dataset built: {output_path} ({len(full_df)} rows)")

if __name__ == "__main__":
    import vectorbt as vbt # Needed for RSI in the script
    build_weighting_dataset()
