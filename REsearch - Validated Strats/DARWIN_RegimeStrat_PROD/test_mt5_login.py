#!/usr/bin/env python3
"""
Quick MT5 login test.
"""
import sys
import os

# Load config
config = {}
config_path = os.path.join(os.path.dirname(__file__), "config.env")
with open(config_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()

import MetaTrader5 as mt5

print("Connecting to MT5...")
ok = mt5.initialize(
    path=config.get("MT5_PATH", ""),
    login=int(config.get("MT5_LOGIN", 0)),
    password=config.get("MT5_PASSWORD", ""),
    server=config.get("MT5_SERVER", ""),
)

if not ok:
    err = mt5.last_error()
    print(f"FAILED: {err}")
    sys.exit(1)

info = mt5.account_info()
print("LOGIN OK")
print(f"  Login:    {info.login}")
print(f"  Name:     {info.name}")
print(f"  Server:   {info.server}")
print(f"  Balance:  {info.balance:,.2f}")
print(f"  Equity:   {info.equity:,.2f}")
print(f"  Margin:   {info.margin:,.2f}")
print(f"  Free Mgn: {info.margin_free:,.2f}")
print(f"  Currency: {info.currency}")

# Check NQ symbol
sym = mt5.symbol_info("NQ_H")
if sym:
    tick = mt5.symbol_info_tick("NQ_H")
    print(f"  NQ_H bid: {tick.bid}  ask: {tick.ask}  spread: {tick.ask - tick.bid:.2f}pts")
else:
    print("  NQ_H: symbol not found (MT5 may need market watch refresh)")

mt5.shutdown()
print("============================================")
print("All checks passed.")
