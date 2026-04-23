import pandas as pd
import numpy as np

def get_strategy_signals(close, high, low, vix_series):
    """
    Regime Switching Strategy (instrument-agnostic, long/short).
    Works on SPY, ES=F, MES=F, NQ=F, MNQ=F — any major equity index.
    For futures deployment, use futures_config.allocation_to_contracts() to convert
    the allocation signal to discrete contract counts.
    
    Regimes & Allocations:
    1. Recovery (Price > EMA20 & RSI > 50 & VIX < 30) -> Vol Target (Min 0.8)
    2. Range (ADX < 25 & VIX < 25, below trend) -> Mean Reversion:
       - BB dip: long 0.5x | BB rip: SHORT -0.2x | neutral: 0.1x
    3. Trend (Price > EMA50) -> Vol Target (Max 2.0x)
    Default: Graduated 0.2-0.4 based on ATR distance to EMA50.
    
    Timing overlays:
    - Graduated entry: smooth ramp within 1 ATR of EMA50 (improves Darwinex Os)
    - Early exit: RSI fade >15pts from peak reduces allocation 30% (improves Darwinex Cs)
    Crisis overlay: Smooth VIX-based scaling (VIX 25-50 linearly reduces allocation).
    
    Returns:
        pd.Series: Target Allocation (-0.2 to 2.0).
    """
    # --- Indicators (Pure Pandas/NumPy - No External Libs) ---
    
    # 1. Moving Averages
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean() # Fast Re-entry
    
    # 2. RSI (14) - Wilder's Smoothing (Standard)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 3. ADX (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14).mean()
    
    up = high - high.shift(1)
    dn = low.shift(1) - low
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm, index=close.index).ewm(alpha=1/14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).ewm(alpha=1/14, min_periods=14).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/14, min_periods=14).mean()
    
    # 4. Bollinger Bands (20, 2.0)
    ma_bb = close.rolling(20).mean()
    std_bb = close.rolling(20).std()
    upper_bb = ma_bb + (2.0 * std_bb)
    lower_bb = ma_bb - (2.0 * std_bb)
    
    # 5. Volatility Targeting (16% Annualized)
    realized_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_scalar = (0.16 / realized_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_scalar = vol_scalar.clip(0.5, 2.0) # Leverage Cap 2.0x
    
    # --- Regime Logic ---
    # 6. ATR distance to EMA50 (for graduated entry)
    dist_to_ema = (close - ema_50) / atr
    
    # Graduated default: smooth ramp from 0.2 (far below) to 0.4 (at/above EMA50)
    alloc = pd.Series(0.2, index=close.index)
    approaching = (dist_to_ema > -1.0) & (dist_to_ema <= 0)
    alloc.loc[approaching] = 0.2 + 0.2 * (dist_to_ema.loc[approaching] + 1.0)
    alloc.loc[close > ema_50] = 0.4
    
    # Flags
    is_crisis = vix_series > 35
    is_trend = (close > ema_50) & (~is_crisis)
    is_recovery = (close > ema_20) & (rsi > 50) & (vix_series < 30) & (~is_crisis)
    is_range = (adx < 25) & (vix_series < 25) & (~is_trend) & (~is_recovery) & (~is_crisis)
    
    # Allocations (Priority: Recovery -> Range -> Graduated Trend)
    
    # 1. Recovery (Base Bullish Floor)
    alloc.loc[is_recovery] = vol_scalar.loc[is_recovery].clip(lower=0.8)

    # 2. Bidirectional Range (Mean Reversion — long dips, short rips)
    if is_range.any():
        range_alloc = pd.Series(0.1, index=close.index)  # near-flat base
        range_alloc.loc[close < lower_bb] = 0.5           # long on BB dip
        range_alloc.loc[close > upper_bb] = -0.2          # SHORT on BB rip
        alloc.loc[is_range] = range_alloc.loc[is_range]

    # 3. Graduated Trend Entry (smooth ramp within 1 ATR above EMA50)
    just_above = (dist_to_ema > 0) & (dist_to_ema <= 1.0) & (~is_crisis)
    alloc.loc[just_above] = 0.4 + dist_to_ema.loc[just_above].clip(0, 1) * (vol_scalar.loc[just_above] - 0.4)
    full_trend = (dist_to_ema > 1.0) & (~is_crisis)
    alloc.loc[full_trend] = vol_scalar.loc[full_trend]
    
    # 4. Early Exit: RSI momentum fade (reduce 30% when RSI drops >15 from peak)
    rsi_peak = rsi.rolling(14).max()
    rsi_fade = rsi_peak - rsi
    is_fading = (rsi_fade > 15) & is_trend
    alloc.loc[is_fading] = alloc.loc[is_fading] * 0.7
    
    # 5. Smooth Crisis Overlay (applied last — scales down all allocations)
    # VIX <= 25: no effect (scalar = 1.0)
    # VIX = 35: scalar ~= 0.6
    # VIX = 50: scalar ~= 0.05 (nearly flat)
    crisis_scalar = (1.0 - ((vix_series - 25) / 25).clip(0, 1))
    crisis_scalar = crisis_scalar.clip(lower=0.05)  # never fully zero
    alloc = alloc * crisis_scalar
        
    return alloc
