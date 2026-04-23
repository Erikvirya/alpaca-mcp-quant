#!/usr/bin/env python3
"""
Daily recap email for the DARWIN RegimeStrat pipeline.
Reads trade_log.csv and sends a performance + slippage summary via Gmail SMTP.

Schedule via Task Scheduler at 22:30 UTC Mon-Fri (after execute_trade finishes).
Can also be run manually: python daily_recap.py [YYYY-MM-DD]
"""
import csv
import os
import sys
from datetime import datetime, date

import notify
from logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG  = os.path.join(SCRIPT_DIR, "trade_log.csv")


def load_config() -> dict:
    config = {}
    with open(os.path.join(SCRIPT_DIR, "config.env"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


def load_trade_log() -> list:
    if not os.path.exists(TRADE_LOG):
        return []
    with open(TRADE_LOG, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_date(row: dict) -> date:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(row.get("date", ""), fmt).date()
        except ValueError:
            continue
    return date.min


def sf(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def avg(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def slippage_block(label: str, trades: list) -> str:
    if not trades:
        return f"{label}\n{'─'*44}\n  No trades in this window.\n"

    total_pts  = [sf(r.get("slippage_total_pts"))    for r in trades]
    spread_pts = [sf(r.get("slippage_spread_pts"))   for r in trades]
    wb_pts     = [sf(r.get("slippage_walkback_pts")) for r in trades]
    total_bps  = [sf(r.get("slippage_total_bps"))    for r in trades]
    attempts   = [sf(r.get("fill_attempts", 1))      for r in trades]
    spreads    = [sf(r.get("spread_at_order"))        for r in trades]

    passive_fills  = sum(1 for a in attempts if a <= 1)
    walkback_fills = sum(1 for a in attempts if a > 1)
    emergency      = sum(1 for r in trades if r.get("order_type","") == "EMERGENCY")
    worst_slip     = max(total_pts) if total_pts else 0.0
    best_slip      = min(total_pts) if total_pts else 0.0

    rows_str = ""
    for r in trades[-10:]:  # show last 10 trades in window
        d      = r.get("date", "")[:10]
        sym    = r.get("contract_symbol", "")
        delta  = int(sf(r.get("delta_contracts", 0)))
        dirn   = "BUY " if delta > 0 else "SELL"
        slip   = sf(r.get("slippage_total_pts"))
        att    = int(sf(r.get("fill_attempts", 1)))
        otype  = r.get("order_type", "SMART")[:5]
        rows_str += f"  {d}  {dirn} {abs(delta):>2} {sym:<6}  slip={slip:+.2f}pts  att={att}  {otype}\n"

    return (
        f"{label}  ({len(trades)} trades)\n"
        f"{'─'*44}\n"
        f"  Avg Slip Total:      {avg(total_pts):+.3f} pts  ({avg(total_bps):+.1f} bps)\n"
        f"  Avg Slip Spread:     {avg(spread_pts):+.3f} pts\n"
        f"  Avg Slip Walkback:   {avg(wb_pts):+.3f} pts\n"
        f"  Best / Worst:        {best_slip:+.2f} / {worst_slip:+.2f} pts\n"
        f"  Avg Spread @ Order:  {avg(spreads):.3f} pts\n"
        f"  Avg Fill Attempts:   {avg(attempts):.2f}\n"
        f"  Passive fills:       {passive_fills} / {len(trades)}\n"
        f"  Walkback fills:      {walkback_fills} / {len(trades)}\n"
        f"  Emergency orders:    {emergency}\n\n"
        f"  Recent trades:\n"
        f"{rows_str}"
    )


def build_recap(rows: list, today: date) -> tuple:
    rows_7d   = [r for r in rows if (today - parse_date(r)).days <= 7]
    rows_30d  = [r for r in rows if (today - parse_date(r)).days <= 30]
    today_rows = [r for r in rows if parse_date(r) == today]

    trades_7d  = [r for r in rows_7d  if int(sf(r.get("delta_contracts", 0))) != 0]
    trades_30d = [r for r in rows_30d if int(sf(r.get("delta_contracts", 0))) != 0]

    # ── Today ────────────────────────────────────────────────────────
    if today_rows:
        t = today_rows[-1]
        delta  = int(sf(t.get("delta_contracts", 0)))
        dirn   = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"
        today_section = (
            f"TODAY  {today}  |  {dirn} {abs(delta)} x {t.get('contract_symbol','')}\n"
            f"{'─'*44}\n"
            f"  Regime:           {t.get('regime','')}\n"
            f"  Allocation:       {sf(t.get('target_alloc')):.2f}x\n"
            f"  Position:         {t.get('prev_contracts','')} -> {t.get('target_contracts','')}\n"
            f"  Signal Close:     {t.get('signal_close_price','')}\n"
            f"  Fill Price:       {t.get('fill_price','')}\n"
            f"  Order Type:       {t.get('order_type','')}\n"
            f"  Fill Attempts:    {t.get('fill_attempts','')}\n"
            f"  Spread @ Order:   {sf(t.get('spread_at_order')):.2f} pts\n"
            f"\n"
            f"  SLIPPAGE BREAKDOWN\n"
            f"    Total:          {sf(t.get('slippage_total_pts')):+.2f} pts  "
            f"({sf(t.get('slippage_total_bps')):+.1f} bps)\n"
            f"    Spread cost:    {sf(t.get('slippage_spread_pts')):+.2f} pts\n"
            f"    Walkback drift: {sf(t.get('slippage_walkback_pts')):+.2f} pts\n"
            f"\n"
            f"  Equity:           ${sf(t.get('equity')):,.0f}\n"
            f"  VXN:              {sf(t.get('vxn_level')):.2f}\n"
            f"  Commission:       ${sf(t.get('commission')):.2f}\n"
        )
        subject = (
            f"Recap {today} | {dirn} {abs(delta)} {t.get('contract_symbol','')} "
            f"| slip={sf(t.get('slippage_total_pts')):+.2f}pts "
            f"| {t.get('regime','')}"
        )
    else:
        today_section = (
            f"TODAY  {today}\n"
            f"{'─'*44}\n"
            f"  No trade executed today.\n"
        )
        subject = f"Recap {today} | No Trade | Systems OK"

    # ── Current position ─────────────────────────────────────────────
    if rows:
        last = rows[-1]
        pos_section = (
            f"CURRENT POSITION\n"
            f"{'─'*44}\n"
            f"  Contracts:  {last.get('target_contracts','?')} x {last.get('contract_symbol','')}\n"
            f"  Equity:     ${sf(last.get('equity')):,.0f}\n"
            f"  As of:      {last.get('date','?')}\n"
        )
    else:
        pos_section = f"CURRENT POSITION\n{'─'*44}\n  No trade history found.\n"

    # ── Regime breakdown ─────────────────────────────────────────────
    def regime_counts(rws):
        counts: dict = {}
        for r in rws:
            reg = r.get("regime", "UNKNOWN")
            counts[reg] = counts.get(reg, 0) + 1
        return "  " + "  |  ".join(f"{k}: {v}d" for k, v in sorted(counts.items()))

    regime_section = (
        f"REGIME DISTRIBUTION\n"
        f"{'─'*44}\n"
        f"  7d:  {regime_counts(rows_7d)}\n"
        f"  30d: {regime_counts(rows_30d)}\n"
    )

    # ── Slippage stats ───────────────────────────────────────────────
    slip_7d  = slippage_block("SLIPPAGE — LAST 7 DAYS",  trades_7d)
    slip_30d = slippage_block("SLIPPAGE — LAST 30 DAYS", trades_30d)

    body = (
        f"DARWIN RegimeStrat — Daily Recap\n"
        f"{'='*44}\n\n"
        f"{today_section}\n"
        f"{pos_section}\n"
        f"{regime_section}\n"
        f"{slip_7d}\n"
        f"{slip_30d}\n"
        f"{'='*44}\n"
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )

    return subject, body


def run_recap(recap_date: date = None) -> None:
    setup_logging("daily_recap")
    if recap_date is None:
        recap_date = datetime.utcnow().date()

    config = load_config()
    rows   = load_trade_log()

    subject, body = build_recap(rows, recap_date)
    logger.info(f"sending_recap date={recap_date} subject={subject}")

    ok = notify.send_email(subject, body, config)
    if ok:
        logger.info(f"recap_sent=True date={recap_date}")
    else:
        logger.error(f"recap_sent=False date={recap_date}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            recap_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print("Usage: python daily_recap.py [YYYY-MM-DD]")
            sys.exit(1)
    else:
        recap_date = None

    run_recap(recap_date)
