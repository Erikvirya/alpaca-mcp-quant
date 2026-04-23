import os
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from catboost import CatBoostClassifier
from datetime import datetime, timedelta

# Import strategy logic from current directory
from strategy_logic import get_strategy_signals
from futures_config import NQ, COMMISSION_PER_SIDE

# --- CONFIG ---
DATA_DIR = "REsearch - Validated Strats/DARWIN_RegimeStrat_PROD/data"
DATASET_PATH = os.path.join(DATA_DIR, "regime_catboost_dataset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "regime_switcher_v1.cbm")

def run_catboost_backtest():
    print(f"Loading model from {MODEL_PATH}...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    print(f"Loading dataset features from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    features = [
        'EMA50', 'EMA20', 'RSI', 'ATR', 'ADX', 'VXN', 
        'MA20', 'STD20', 'UpperBB', 'LowerBB', 
        'RV20', 'Vol_Scalar', 'Dist_EMA50_ATR'
    ]
    
    # Predict regimes for the whole period (OOS starts at 2023-01-01)
    df['Predicted_Regime'] = model.predict(df[features])
    
    # Map back to allocation logic
    # 0: TREND, 1: RECOVERY, 2: RANGE
    
    def get_ml_allocation(row):
        pred = row['Predicted_Regime']
        if pred == 0: # TREND
            # Graduated Trend Entry (Logic from strategy_logic.py)
            dist_to_ema = row['Dist_EMA50_ATR']
            if dist_to_ema > 1.0:
                return row['Vol_Scalar']
            elif dist_to_ema > 0:
                return 0.4 + dist_to_ema * (row['Vol_Scalar'] - 0.4)
            else:
                return 0.4
        elif pred == 1: # RECOVERY
            return max(0.8, row['Vol_Scalar'])
        elif pred == 2: # RANGE
            if row['Close'] < row['LowerBB']: return 0.5
            if row['Close'] > row['UpperBB']: return -0.2
            return 0.1
        return 0.2 # Default
    
    df['ML_Alloc'] = df.apply(get_ml_allocation, axis=1)
    
    # Apply Crisis Overlay (Global protection)
    vix_series = df['VXN']
    crisis_scalar = (1.0 - ((vix_series - 25) / 25).clip(0, 1)).clip(lower=0.05)
    df['ML_Alloc_Final'] = df['ML_Alloc'] * crisis_scalar
    
    # Shift signals to prevent look-ahead
    trade_signals = df['ML_Alloc_Final'].shift(1).fillna(0)
    
    # Filter for OOS period (2023-2026) to see if ML added value
    test_start = '2023-01-01'
    price_data = df.loc[df.index >= test_start, 'Close']
    ml_signals = trade_signals.loc[df.index >= test_start]
    
    # Standard Strategy for comparison
    df['Standard_Alloc'] = get_strategy_signals(df['Close'], df['High'], df['Low'], df['VXN'])
    std_signals = df['Standard_Alloc'].shift(1).fillna(0).loc[df.index >= test_start]
    
    print(f"Running VectorBT simulation (OOS Period: {test_start} to {df.index[-1].date()})...")
    
    # ML Portfolio
    pf_ml = vbt.Portfolio.from_orders(
        price_data,
        size=ml_signals,
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='1D'
    )
    
    # Standard Portfolio
    pf_std = vbt.Portfolio.from_orders(
        price_data,
        size=std_signals,
        size_type='targetpercent',
        init_cash=1_000_000,
        fees=0.0002,
        slippage=0.0001,
        freq='1D'
    )
    
    print("\n" + "="*60)
    print(f"{'ML REGIME SWITCHER vs STANDARD (OOS)':^60}")
    print("="*60)
    
    ml_stats = pf_ml.stats()
    std_stats = pf_std.stats()
    
    print(f"{'Metric':<25} | {'Standard':<15} | {'ML Strategy':<15}")
    print("-" * 60)
    print(f"{'Total Return [%]':<25} | {std_stats['Total Return [%]']:>14.2f}% | {ml_stats['Total Return [%]']:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} | {std_stats['Sharpe Ratio']:>15.2f} | {ml_stats['Sharpe Ratio']:>15.2f}")
    print(f"{'Max Drawdown [%]':<25} | {std_stats['Max Drawdown [%]']:>14.2f}% | {ml_stats['Max Drawdown [%]']:>14.2f}%")
    print(f"{'Win Rate [%]':<25} | {std_stats['Win Rate [%]']:>14.2f}% | {ml_stats['Win Rate [%]']:>14.2f}%")
    print(f"{'Total Trades':<25} | {int(std_stats['Total Trades']):>15} | {int(ml_stats['Total Trades']):>15}")
    print("="*60)

if __name__ == "__main__":
    run_catboost_backtest()
