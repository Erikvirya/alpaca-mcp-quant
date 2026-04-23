import os
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
from catboost import CatBoostClassifier
from strategy_logic import get_strategy_signals

# --- CONFIG ---
DATA_DIR = "REsearch - Validated Strats/DARWIN_RegimeStrat_PROD/data"
MODEL_PATH = os.path.join(DATA_DIR, "asset_weighting_v1.cbm")
DATASET_PATH = os.path.join(DATA_DIR, "multi_asset_weighting_dataset.csv")

def run_ml_weighting_backtest():
    print(f"Loading weighting model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    features = []
    for name in ['NQ', 'ES', 'YM']:
        features += [f'{name}_Vol', f'{name}_RSI', f'{name}_EMA50_Dist', f'{name}_Mom20']
    
    # Get Probabilities for each asset being the "Leader"
    # mapping = {'NQ': 0, 'ES': 1, 'YM': 2}
    probs = model.predict_proba(df[features])
    
    # These probabilities will be our dynamic weights
    weights = pd.DataFrame(probs, index=df.index, columns=['NQ', 'ES', 'YM'])
    # Shift weights to avoid look-ahead
    weights = weights.shift(1).fillna(0.33)
    
    # Calculate base signals for each asset using strategy_logic
    print("Calculating base strategy signals...")
    base_signals = {}
    for name in ['NQ', 'ES', 'YM']:
        # Note: We need high/low/close for strategy_logic. 
        # For simplicity, we assume we have them or download them.
        # But we can also just use the ones already in the dataset df if we had them.
        # Let's just download them again to be safe and clean.
        ticker = 'NQ=F' if name == 'NQ' else ('ES=F' if name == 'ES' else 'YM=F')
        p_df = yf.download(ticker, start=df.index.min(), end=df.index.max(), progress=False, multi_level_index=False)
        v_df = df[f'{name}_Vol']
        
        common = p_df.index.intersection(v_df.index)
        alloc = get_strategy_signals(p_df.loc[common, 'Close'], p_df.loc[common, 'High'], p_df.loc[common, 'Low'], v_df.loc[common])
        base_signals[name] = alloc

    base_signal_matrix = pd.DataFrame(base_signals).fillna(0)
    
    # Combine: Final Signal = Base Signal * ML Weight
    # This ensures we only trade the asset if the regime is right, 
    # and we overweight the asset the model likes.
    ml_weighted_signals = base_signal_matrix * weights
    
    # Benchmark: Equal weight (alloc / 3.0)
    equal_weighted_signals = base_signal_matrix * 0.3333
    
    # Run Backtest (OOS only: 2023-01-01 to 2026-03-10)
    test_start = '2023-01-01'
    price_matrix = pd.DataFrame({n: df[f'{n}_Close'] for n in ['NQ', 'ES', 'YM']})

    # Final Alignment
    common_idx = price_matrix.index.intersection(ml_weighted_signals.index).intersection(equal_weighted_signals.index)
    price_matrix = price_matrix.loc[common_idx]
    ml_weighted_signals = ml_weighted_signals.loc[common_idx]
    equal_weighted_signals = equal_weighted_signals.loc[common_idx]

    price_test = price_matrix.loc[price_matrix.index >= test_start]
    ml_sig_test = ml_weighted_signals.loc[ml_weighted_signals.index >= test_start]
    eq_sig_test = equal_weighted_signals.loc[equal_weighted_signals.index >= test_start]

    print(f"Running Portfolio Simulations (Size: {len(price_test)} rows)...")

    
    pf_ml = vbt.Portfolio.from_orders(
        price_test, ml_sig_test, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )
    
    pf_eq = vbt.Portfolio.from_orders(
        price_test, eq_sig_test, size_type='targetpercent',
        init_cash=1_000_000, fees=0.0002, slippage=0.0001, freq='1D'
    )

    ml_stats = pf_ml.stats()
    eq_stats = pf_eq.stats()
    
    # Custom Sharpe
    ml_rets = pf_ml.returns().mean(axis=1)
    eq_rets = pf_eq.returns().mean(axis=1)
    ml_sharpe = np.sqrt(252) * (ml_rets.mean() / ml_rets.std())
    eq_sharpe = np.sqrt(252) * (eq_rets.mean() / eq_rets.std())

    print("\n" + "="*60)
    print(f"{'ML DYNAMIC WEIGHTING vs EQUAL WEIGHT (OOS)':^60}")
    print("="*60)
    print(f"{'Metric':<25} | {'Equal Weight':<15} | {'ML Weighted':<15}")
    print("-" * 60)
    print(f"{'Total Return [%]':<25} | {eq_stats['Total Return [%]']:>14.2f}% | {ml_stats['Total Return [%]']:>14.2f}%")
    print(f"{'Portfolio Sharpe':<25} | {eq_sharpe:>15.2f} | {ml_sharpe:>15.2f}")
    print(f"{'Max Drawdown [%]':<25} | {eq_stats['Max Drawdown [%]']:>14.2f}% | {ml_stats['Max Drawdown [%]']:>14.2f}%")
    print(f"{'Total Trades':<25} | {int(eq_stats['Total Trades']):>15} | {int(ml_stats['Total Trades']):>15}")
    print("="*60)

if __name__ == "__main__":
    run_ml_weighting_backtest()
