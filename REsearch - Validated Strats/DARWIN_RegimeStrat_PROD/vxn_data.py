"""
VXN (Nasdaq Volatility Index) data sourcing for the Darwinex Zero pipeline.
Waterfall: CBOE CSV (primary) → yfinance (fallback) → local cache (emergency).
"""
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "vxn_cache.csv")
MAX_CACHE_STALE_DAYS = 3


def fetch_vxn(lookback_days: int = 400) -> pd.Series:
    """
    Fetch VXN close prices. Tries CBOE → yfinance → local cache.

    Returns:
        pd.Series with DatetimeIndex and VXN close values.

    Raises:
        RuntimeError: if all sources fail and cache is too stale.
    """
    # 1. Try CBOE
    try:
        vxn = _fetch_cboe_vxn(lookback_days)
        if vxn is not None and len(vxn) > 0:
            _save_cache(vxn)
            return vxn
    except Exception as e:
        logger.warning(f"primary_error=CBOE_failed error={e}")

    # 2. Try yfinance
    try:
        vxn = _fetch_yfinance_vxn(lookback_days)
        if vxn is not None and len(vxn) > 0:
            logger.warning(f"fallback_source=yfinance")
            _save_cache(vxn)
            return vxn
    except Exception as e:
        logger.warning(f"fallback_error=yfinance_failed error={e}")

    # 3. Try local cache
    vxn = _load_cached_vxn(lookback_days)
    if vxn is not None and len(vxn) > 0:
        return vxn

    raise RuntimeError("All VXN sources failed and cache is stale or missing")


def _fetch_cboe_vxn(lookback_days: int) -> pd.Series:
    """Fetch VXN from CBOE direct CSV download."""
    logger.info("source=CBOE fetching VXN...")
    resp = requests.get(CBOE_URL, timeout=30)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff]

    vxn = df["CLOSE"].dropna()
    vxn.index.name = None
    latest = vxn.index[-1].date()
    logger.info(f"source=CBOE latest_date={latest} vxn_close={vxn.iloc[-1]:.2f} bars={len(vxn)}")
    return vxn


def _fetch_yfinance_vxn(lookback_days: int) -> pd.Series:
    """Fetch VXN from yfinance as fallback."""
    import yfinance as yf

    logger.info("source=yfinance fetching VXN...")
    start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    data = yf.download("^VXN", start=start, progress=False, multi_level_index=False, timeout=30)
    if data is None or data.empty:
        return None

    vxn = data["Close"].dropna()
    latest = vxn.index[-1].date()
    logger.info(f"source=yfinance latest_date={latest} vxn_close={vxn.iloc[-1]:.2f} bars={len(vxn)}")
    return vxn


def _save_cache(vxn: pd.Series) -> None:
    """Save VXN data to local cache file."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        vxn_df = vxn.tail(500).to_frame(name="CLOSE")
        vxn_df.index.name = "DATE"
        vxn_df.to_csv(CACHE_FILE)
        logger.info(f"cache_saved=True rows={len(vxn_df)}")
    except Exception as e:
        logger.warning(f"cache_save_failed error={e}")


def _load_cached_vxn(lookback_days: int) -> pd.Series:
    """Load VXN from local cache. Warns if stale."""
    if not os.path.exists(CACHE_FILE):
        logger.error("cache_exists=False")
        return None

    try:
        df = pd.read_csv(CACHE_FILE, index_col="DATE", parse_dates=True)
        vxn = df["CLOSE"].dropna()

        latest = vxn.index[-1].date()
        today = datetime.utcnow().date()
        # Count business days stale
        stale_days = sum(
            1 for d in pd.bdate_range(latest, today) if d.date() != latest
        )

        if stale_days > MAX_CACHE_STALE_DAYS:
            logger.error(
                f"cache_stale=True cache_age_bdays={stale_days} "
                f"latest_cached_date={latest} max_allowed={MAX_CACHE_STALE_DAYS}"
            )
            return None

        logger.warning(
            f"source=cache cache_age_bdays={stale_days} "
            f"latest_cached_date={latest} vxn_close={vxn.iloc[-1]:.2f}"
        )
        cutoff = vxn.index.max() - pd.Timedelta(days=lookback_days)
        return vxn[vxn.index >= cutoff]
    except Exception as e:
        logger.error(f"cache_load_failed error={e}")
        return None
