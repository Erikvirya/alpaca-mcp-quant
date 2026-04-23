import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIG ---
START_DATE = '2015-01-01'
END_DATE = '2026-03-13'
DATA_DIR = "REsearch - Validated Strats/DARWIN_RegimeStrat_PROD/data"
os.makedirs(DATA_DIR, exist_ok=True)

def build_dataset():
    print(f"Fetching data from {START_DATE} to {END_DATE}...")
    nq = yf.download('NQ=F', start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
    vxn = yf.download('^VXN', start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
    
    # Align
    common_idx = nq.index.intersection(vxn.index)
    nq = nq.loc[common_idx]
    vxn = vxn.loc[common_idx]
    
    df = nq.copy()
    df['VXN'] = vxn['Close']
    
    print("Calculating Features...")
    # 1. Moving Averages
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. ATR & ADX
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    
    up = df['High'] - df['High'].shift(1)
    dn = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, min_periods=14).mean() / df['ATR'])
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, min_periods=14).mean() / df['ATR'])
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    
    # 4. Bollinger Bands
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['UpperBB'] = df['MA20'] + (2.0 * df['STD20'])
    df['LowerBB'] = df['MA20'] - (2.0 * df['STD20'])
    
    # 5. Vol Targeting Scalar
    df['Returns'] = df['Close'].pct_change()
    df['RV20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    df['Vol_Scalar'] = (0.16 / df['RV20']).clip(0.5, 2.0)
    
    # 6. Distances
    df['Dist_EMA50_ATR'] = (df['Close'] - df['EMA50']) / df['ATR']
    
    print("Simulating Daily Regime Returns for Labels...")
    # Calculate next-day return for each strategy logic
    # We want to know which logic would have worked best TODAY (to label YESTERDAY)
    
    next_ret = df['Returns'].shift(-1)
    
    # Logic returns (Simplified)
    # TREND: If Price > EMA50, Long with Vol_Scalar
    trend_ret = np.where(df['Close'] > df['EMA50'], next_ret * df['Vol_Scalar'], 0)
    
    # RECOVERY: If Price > EMA20 and RSI > 50, Long with high Vol_Scalar
    rec_ret = np.where((df['Close'] > df['EMA20']) & (df['RSI'] > 50), next_ret * df['Vol_Scalar'].clip(lower=0.8), 0)
    
    # RANGE: Mean Reversion
    range_ret = np.where(df['Close'] < df['LowerBB'], next_ret * 0.5, 
                         np.where(df['Close'] > df['UpperBB'], -next_ret * 0.2, next_ret * 0.1))
    
    df['Ret_Trend'] = trend_ret
    df['Ret_Recovery'] = rec_ret
    df['Ret_Range'] = range_ret
    
    # Target: The regime that has the highest next-day return
    # 0: TREND, 1: RECOVERY, 2: RANGE
    regime_returns = pd.DataFrame({
        'TREND': trend_ret,
        'RECOVERY': rec_ret,
        'RANGE': range_ret
    }, index=df.index)
    
    df['Target_Regime_Name'] = regime_returns.idxmax(axis=1)
    # Map to Int
    mapping = {'TREND': 0, 'RECOVERY': 1, 'RANGE': 2}
    df['Target'] = df['Target_Regime_Name'].map(mapping)
    
    # Features for the model (All from TODAY to predict TOMORROW)
    # We drop the simulation returns and current target from the feature set later
    
    # Clean up
    df = df.dropna()
    # Shift target back by 1: We want TODAY's features to predict TOMORROW's best regime
    df['Target'] = df['Target'].shift(-1)
    df = df.dropna()
    
    dataset_path = os.path.join(DATA_DIR, "regime_catboost_dataset.csv")
    df.to_csv(dataset_path)
    print(f"Dataset built: {dataset_path} ({len(df)} rows)")

if __name__ == "__main__":
    build_dataset()
