import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import os

# --- CONFIG ---
DATA_DIR = "REsearch - Validated Strats/DARWIN_RegimeStrat_PROD/data"
DATASET_PATH = os.path.join(DATA_DIR, "regime_catboost_dataset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "regime_switcher_v1.cbm")

def train_model():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Define Features
    features = [
        'EMA50', 'EMA20', 'RSI', 'ATR', 'ADX', 'VXN', 
        'MA20', 'STD20', 'UpperBB', 'LowerBB', 
        'RV20', 'Vol_Scalar', 'Dist_EMA50_ATR'
    ]
    
    target = 'Target'
    
    # Walk-forward split
    # Train: 2015-2022
    # Test: 2023-2026
    train_df = df[df.index < '2023-01-01']
    test_df = df[df.index >= '2023-01-01']
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set:  {len(test_df)} rows")
    
    X_train = train_df[features]
    y_train = train_df[target].astype(int)
    
    X_test = test_df[features]
    y_test = test_df[target].astype(int)
    
    # Class weights (if imbalanced)
    # mapping = {'TREND': 0, 'RECOVERY': 1, 'RANGE': 2}
    print("Target distribution in training:")
    print(y_train.value_counts(normalize=True))
    
    # Model configuration
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100
    )
    
    print("Starting training...")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    
    # Evaluate
    test_acc = model.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Feature Importance
    importances = model.get_feature_importance()
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(feat_imp)

if __name__ == "__main__":
    train_model()
