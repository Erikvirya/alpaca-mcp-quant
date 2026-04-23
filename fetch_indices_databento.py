import databento as db
from datetime import datetime, timedelta
import os
import pandas as pd

def load_api_key():
    with open("databento.env", "r") as f:
        content = f.read().strip()
        for line in content.split('\n'):
            if "API Key:" in line:
                return line.split("API Key:")[1].strip()
    return None

api_key = load_api_key()
client = db.Historical(api_key)

symbols = ["ES", "YM"]
# End date: March 11, 2026
end_date = datetime(2026, 3, 11)
start_date = end_date - timedelta(days=365)

for symbol in symbols:
    output_path = f"market_data/{symbol}_1m_1yr.parquet"
    print(f"Fetching {symbol} 1-minute data from {start_date.date()} to {end_date.date()}...")
    try:
        data = client.timeseries.get_range(
            dataset="DBEQ.BASIC",
            symbols=symbol,
            schema="ohlcv-1m",
            start=start_date,
            end=end_date
        )
        df = data.to_df()
        if not df.empty:
            df = df.reset_index()
            df = df.rename(columns={'ts_event': 'timestamp'})
            final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            final_df.to_parquet(output_path, index=False)
            print(f"Successfully saved {symbol} data to {output_path} ({len(final_df)} rows)")
        else:
            print(f"No data returned for {symbol}.")
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
