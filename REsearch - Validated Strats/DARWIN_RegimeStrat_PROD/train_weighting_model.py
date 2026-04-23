import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os

# --- CONFIG ---
DATA_DIR = "REsearch - Validated Strats/DARWIN_RegimeStrat_PROD/data"
DATASET_PATH = os.path.join(DATA_DIR, "multi_asset_weighting_dataset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "asset_weighting_v1.cbm")

def train_weighting_model():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Define Features
    features = []
    for name in ['NQ', 'ES', 'YM']:
        features += [f'{name}_Vol', f'{name}_RSI', f'{name}_EMA50_Dist', f'{name}_Mom20']
    
    target = 'Target'
    
    # Walk-forward split
    train_df = df[df.index < '2023-01-01']
    test_df = df[df.index >= '2023-01-01']
    
    X_train, y_train = train_df[features], train_df[target].astype(int)
    X_test, y_test = test_df[features], test_df[target].astype(int)
    
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("Target distribution (Train):")
    print(y_train.value_counts(normalize=True))
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=5,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100
    )
    
    print("Training Asset Weighting Model...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Best Test Accuracy: {model.get_best_score()['validation']['Accuracy']:.4f}")

if __name__ == "__main__":
    train_weighting_model()
