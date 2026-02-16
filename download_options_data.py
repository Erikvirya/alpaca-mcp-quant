"""
Download options EOD data WITH Greeks from DoltHub and save locally as parquet.
No external terminal or paid subscription required — uses the free DoltHub SQL API.

Coverage: S&P 500 components + SPY + SPDR ETFs, 2019–present, ~3 short-term expirations.

Usage:
    python download_options_data.py --symbol SPY --start 2024-01-01 --end 2026-02-12
    python download_options_data.py --symbol SPY --start 2024-01-01  # end defaults to yesterday
    python download_options_data.py --symbol AAPL --start 2025-01-01 --max-dte 90

The script downloads in monthly chunks to avoid DoltHub API timeouts and saves
incrementally. If interrupted, re-run with the same args to resume —
it skips months that already have data in the output file.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

DOLTHUB_API_BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "options_cache")


def dolthub_query(sql: str, timeout: int = 120) -> list:
    """Execute a SQL query against the DoltHub options database. Returns list of row dicts."""
    resp = requests.get(DOLTHUB_API_BASE, params={"q": sql}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("query_execution_status") != "Success":
        raise RuntimeError(f"DoltHub query failed: {data.get('query_execution_message', 'unknown')}")
    return data.get("rows", [])


def _to_yyyymmdd_int(series):
    """Convert a Series of mixed date formats to YYYYMMDD int."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    s = series.astype(str).str[:10].str.replace("-", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def month_ranges(start: datetime, end: datetime):
    """Yield (start_date, end_date) tuples for each month in the range."""
    current = start.replace(day=1)
    while current <= end:
        month_start = max(current, start)
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        month_end = min(next_month - timedelta(days=1), end)
        yield month_start, month_end
        current = next_month


def fetch_month(symbol: str, start_date: str, end_date: str, max_dte: int) -> list:
    """Fetch one month of option chain data with Greeks from DoltHub.
    Returns list of normalized row dicts ready for parquet."""
    sql = (
        f"SELECT date, act_symbol, expiration, strike, call_put, "
        f"bid, ask, vol, delta, gamma, theta, vega, rho "
        f"FROM option_chain "
        f"WHERE act_symbol = '{symbol}' "
        f"AND date >= '{start_date}' AND date <= '{end_date}' "
        f"ORDER BY date, expiration, strike"
    )

    rows = dolthub_query(sql)
    normalized = []
    for row in rows:
        d_str = row.get("date", "")
        exp_str = row.get("expiration", "")
        d_int = int(d_str.replace("-", "")) if d_str else 0
        exp_int = int(exp_str.replace("-", "")) if exp_str else 0

        try:
            d_date = datetime.strptime(d_str, "%Y-%m-%d")
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - d_date).days
        except Exception:
            dte = 0

        if dte > max_dte:
            continue

        cp = row.get("call_put", "")
        right = "C" if cp.lower().startswith("c") else "P"

        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        mid = round((bid + ask) / 2, 4) if (bid > 0 or ask > 0) else 0.0

        rec = {
            "date": d_int,
            "expiration": exp_int,
            "strike": float(row.get("strike", 0)),
            "right": right,
            "bid": bid,
            "ask": ask,
            "close": mid,
            "open": mid,
            "high": ask if ask > 0 else mid,
            "low": bid if bid > 0 else mid,
            "volume": 0,
            "dte": dte,
            "iv": float(row.get("vol", 0)) if row.get("vol") is not None else None,
            "delta": float(row.get("delta", 0)) if row.get("delta") is not None else None,
            "gamma": float(row.get("gamma", 0)) if row.get("gamma") is not None else None,
            "theta": float(row.get("theta", 0)) if row.get("theta") is not None else None,
            "vega": float(row.get("vega", 0)) if row.get("vega") is not None else None,
            "rho": float(row.get("rho", 0)) if row.get("rho") is not None else None,
        }
        normalized.append(rec)

    return normalized


def download_symbol(symbol: str, start: str, end: str, max_dte: int = 90):
    """Download all option EOD data with Greeks for a symbol and save as parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"{symbol.upper()}_eod.parquet")

    # Load existing data to enable resume
    existing_months = set()
    existing_df = None
    if os.path.exists(out_path):
        existing_df = pd.read_parquet(out_path)
        if "date" in existing_df.columns:
            def _date_to_month_key(d):
                s = str(int(d))
                return f"{s[:4]}-{s[4:6]}"
            existing_months = set(existing_df["date"].apply(_date_to_month_key).unique())
        print(f"  Existing cache: {len(existing_df)} rows, {len(existing_months)} months")
        has_greeks = "delta" in existing_df.columns
        if not has_greeks:
            print("  ⚠ Existing cache lacks Greeks — will re-download all months from DoltHub")
            existing_months = set()
            existing_df = None

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    all_chunks = []
    if existing_df is not None:
        all_chunks.append(existing_df)

    total_new = 0
    months = list(month_ranges(start_dt, end_dt))

    for i, (ms, me) in enumerate(months, 1):
        month_key = ms.strftime("%Y-%m")
        if month_key in existing_months:
            print(f"  [{i}/{len(months)}] {month_key} — already cached, skipping")
            continue

        s = ms.strftime("%Y-%m-%d")
        e = me.strftime("%Y-%m-%d")
        label = f"[{i}/{len(months)}] {month_key}"

        try:
            month_rows = fetch_month(symbol.upper(), s, e, max_dte)
            print(f"  {label}: {len(month_rows)} rows")
        except requests.exceptions.HTTPError as ex:
            print(f"  {label}: HTTP error {ex}")
            month_rows = []
        except requests.exceptions.ConnectionError as ex:
            print(f"  {label}: Connection error — {ex}")
            month_rows = []
        except Exception as ex:
            print(f"  {label}: Error — {ex}")
            month_rows = []

        if month_rows:
            chunk = pd.DataFrame(month_rows)
            all_chunks.append(chunk)
            total_new += len(month_rows)

        # Rate-limit to be polite to DoltHub API
        time.sleep(0.5)

        # Save incrementally every 3 months
        if i % 3 == 0 and all_chunks:
            combined = pd.concat(all_chunks, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            print(f"  — Saved checkpoint: {len(combined)} total rows")

    # Final save
    if all_chunks:
        combined = pd.concat(all_chunks, ignore_index=True)
        # Ensure consistent int types
        for col in ["date", "expiration"]:
            if col in combined.columns:
                combined[col] = _to_yyyymmdd_int(combined[col])
        # Drop duplicates in case of overlapping resume
        dedup_cols = [c for c in ["date", "expiration", "strike", "right"] if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        combined = combined.sort_values(["date", "expiration", "strike", "right"]).reset_index(drop=True)
        combined.to_parquet(out_path, index=False)
        print(f"\nDone! Saved {len(combined)} rows to {out_path}")
        print(f"  New rows this run: {total_new}")
        print(f"  Columns: {list(combined.columns)}")
        print(f"  Greeks: {'✓ yes' if 'delta' in combined.columns else '✗ no'}")
        print(f"  File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")
    else:
        print("\nNo data downloaded. Check that the symbol is in the DoltHub S&P 500 universe.")


def main():
    parser = argparse.ArgumentParser(
        description="Download options EOD data with Greeks from DoltHub (free)")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--max-dte", type=int, default=90, help="Max DTE to include (default: 90)")
    args = parser.parse_args()

    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading {args.symbol} options EOD + Greeks from DoltHub")
    print(f"  Range: {args.start} → {args.end}")
    print(f"  Max DTE: {args.max_dte}")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Source: DoltHub (post-no-preference/options)")
    print()

    download_symbol(args.symbol, args.start, args.end, args.max_dte)


if __name__ == "__main__":
    main()
