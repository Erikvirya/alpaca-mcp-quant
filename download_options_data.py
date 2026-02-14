"""
Download options EOD data from ThetaData and save locally as parquet.
Requires Theta Terminal v3 running on http://127.0.0.1:25503

Usage:
    python download_options_data.py --symbol SPY --start 2024-01-01 --end 2026-02-12
    python download_options_data.py --symbol SPY --start 2024-01-01  # end defaults to yesterday
    python download_options_data.py --symbol AAPL --start 2025-01-01 --max-dte 90

The script downloads in monthly chunks to avoid API timeouts and saves
incrementally. If interrupted, re-run with the same args to resume —
it skips months that already have data in the output file.
"""

import argparse
import csv
import io
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

THETADATA_URL = os.environ.get("THETADATA_URL", "http://127.0.0.1:25503")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "options_cache")


def _to_yyyymmdd_int(series):
    """Convert a Series of mixed date formats to YYYYMMDD int. Handles int, str, Timestamp."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    # String-based: take first 10 chars (YYYY-MM-DD) and strip dashes
    s = series.astype(str).str[:10].str.replace("-", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def theta_get(path: str, params: dict, timeout: int = 120) -> list:
    """Fetch from local Theta Terminal, parse CSV → list of dicts."""
    url = f"{THETADATA_URL}{path}"
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        clean = {}
        for k, v in row.items():
            v = v.strip('"') if v else v
            try:
                if '.' in v:
                    clean[k] = float(v)
                else:
                    clean[k] = int(v)
            except (ValueError, TypeError):
                clean[k] = v
        rows.append(clean)
    return rows


def month_ranges(start: datetime, end: datetime):
    """Yield (start_date, end_date) tuples for each month in the range."""
    current = start.replace(day=1)
    while current <= end:
        month_start = max(current, start)
        # Last day of the month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        month_end = min(next_month - timedelta(days=1), end)
        yield month_start, month_end
        current = next_month


def download_symbol(symbol: str, start: str, end: str, max_dte: int = 90):
    """Download all option EOD data for a symbol and save as parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"{symbol.upper()}_eod.parquet")

    # Load existing data to enable resume
    existing_months = set()
    existing_df = None
    if os.path.exists(out_path):
        existing_df = pd.read_parquet(out_path)
        if "date" in existing_df.columns:
            def _date_to_month_key(d):
                s = str(int(d))  # e.g. "20240201"
                return f"{s[:4]}-{s[4:6]}"
            existing_months = set(existing_df["date"].apply(_date_to_month_key).unique())
        print(f"  Existing cache: {len(existing_df)} rows, {len(existing_months)} months")

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

        s = ms.strftime("%Y%m%d")
        e = me.strftime("%Y%m%d")
        month_rows = []

        for right in ["C", "P"]:
            label = f"[{i}/{len(months)}] {month_key} {right}"
            try:
                rows = theta_get("/v3/option/history/eod", {
                    "symbol": symbol.upper(),
                    "expiration": "*",
                    "strike": "*",
                    "right": right,
                    "start_date": s,
                    "end_date": e,
                    "max_dte": str(max_dte),
                })
                month_rows.extend(rows)
                print(f"  {label}: {len(rows)} rows")
            except requests.exceptions.HTTPError as ex:
                print(f"  {label}: HTTP error {ex}")
            except requests.exceptions.ConnectionError:
                print(f"  {label}: Cannot connect to Theta Terminal at {THETADATA_URL}")
                print("  Make sure Theta Terminal v3 is running.")
                sys.exit(1)
            except Exception as ex:
                print(f"  {label}: Error — {ex}")

            # Rate-limit: free tier allows 1 concurrent request
            time.sleep(1)

        if month_rows:
            chunk = pd.DataFrame(month_rows)
            # Normalize columns to match tool expectations
            if "created" in chunk.columns and "date" not in chunk.columns:
                chunk["date"] = chunk["created"].astype(str).str[:10].str.replace("-", "", regex=False).astype(int)
            if "right" in chunk.columns:
                chunk["right"] = chunk["right"].replace({"CALL": "C", "PUT": "P", "call": "C", "put": "P"})
            if "expiration" in chunk.columns:
                # Ensure expiration is YYYYMMDD int
                chunk["expiration"] = _to_yyyymmdd_int(chunk["expiration"])
            # Drop non-essential columns to save space
            drop_cols = [c for c in ["created", "last_trade", "bid_size", "bid_exchange",
                                     "bid_condition", "ask_size", "ask_exchange", "ask_condition",
                                     "symbol", "count"] if c in chunk.columns]
            chunk = chunk.drop(columns=drop_cols)
            all_chunks.append(chunk)
            total_new += len(month_rows)

        # Save incrementally every 3 months
        if i % 3 == 0 and all_chunks:
            combined = pd.concat(all_chunks, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            print(f"  — Saved checkpoint: {len(combined)} total rows")

    # Final save
    if all_chunks:
        combined = pd.concat(all_chunks, ignore_index=True)
        # Ensure consistent int types after concat of mixed chunks
        for col in ["date", "expiration"]:
            if col in combined.columns:
                combined[col] = _to_yyyymmdd_int(combined[col])
        # Drop duplicates in case of overlapping resume
        dedup_cols = [c for c in ["date", "expiration", "strike", "right"] if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        combined.to_parquet(out_path, index=False)
        print(f"\nDone! Saved {len(combined)} rows to {out_path}")
        print(f"  New rows this run: {total_new}")
        print(f"  File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")
    else:
        print("\nNo data downloaded.")


def main():
    parser = argparse.ArgumentParser(description="Download options EOD data from ThetaData")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--max-dte", type=int, default=90, help="Max DTE to include (default: 90)")
    args = parser.parse_args()

    if args.end is None:
        args.end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading {args.symbol} options EOD data")
    print(f"  Range: {args.start} → {args.end}")
    print(f"  Max DTE: {args.max_dte}")
    print(f"  Cache dir: {CACHE_DIR}")
    print()

    download_symbol(args.symbol, args.start, args.end, args.max_dte)


if __name__ == "__main__":
    main()
