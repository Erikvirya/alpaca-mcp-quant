"""
Slippage tracking and reporting for the Darwinex Zero NQ pipeline.
Reads trade_log.csv and generates per-trade, weekly, and monthly reports.

CLI usage:
    python slippage.py --weekly
    python slippage.py --monthly
    python slippage.py --all
"""
import argparse
import csv
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from logging_config import setup_logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(SCRIPT_DIR, "trade_log.csv")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")


def load_trade_log() -> pd.DataFrame:
    """Load trade_log.csv into a DataFrame."""
    if not os.path.exists(TRADE_LOG):
        raise FileNotFoundError(f"Trade log not found: {TRADE_LOG}")

    df = pd.read_csv(TRADE_LOG, parse_dates=["date"])
    # Only rows where a trade actually happened
    df = df[df["delta_contracts"] != 0].copy()
    numeric_cols = [
        "slippage_total_pts", "slippage_total_bps",
        "slippage_spread_pts", "slippage_walkback_pts",
        "spread_at_order", "fill_attempts",
        "commission", "signal_close_price", "initial_mid",
        "fill_price", "target_alloc", "equity", "vxn_level",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def per_trade_report() -> pd.DataFrame:
    """Return per-trade slippage details with decomposition."""
    df = load_trade_log()
    cols = ["date", "regime", "contract_symbol", "delta_contracts",
            "signal_close_price", "initial_mid", "fill_price",
            "slippage_total_pts", "slippage_spread_pts", "slippage_walkback_pts",
            "spread_at_order", "fill_attempts", "commission"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def _summarize(df: pd.DataFrame, label: str) -> dict:
    """Compute summary stats for a slice of trades with slippage decomposition."""
    if len(df) == 0:
        return {"label": label, "trades": 0}

    commissions = df["commission"]
    point_value = 20.0  # NQ
    df = df.copy()
    abs_cts = df["delta_contracts"].abs()

    # Total slippage (fill vs signal close)
    slip_total = df.get("slippage_total_pts", pd.Series(0, index=df.index))
    slip_total_bps = df.get("slippage_total_bps", pd.Series(0, index=df.index))
    # Spread component (fill vs initial mid)
    slip_spread = df.get("slippage_spread_pts", pd.Series(0, index=df.index))
    # Walkback component (initial mid vs signal close = market drift during retries)
    slip_walkback = df.get("slippage_walkback_pts", pd.Series(0, index=df.index))
    # Spread at time of order
    spread_at = df.get("spread_at_order", pd.Series(0, index=df.index))
    # Fill attempts
    attempts = df.get("fill_attempts", pd.Series(1, index=df.index))

    # USD costs
    df["slip_total_usd"] = slip_total * abs_cts * point_value
    df["slip_spread_usd"] = slip_spread * abs_cts * point_value
    df["slip_walkback_usd"] = slip_walkback * abs_cts * point_value

    return {
        "label": label,
        "trades": len(df),
        # Total slippage
        "total_slip_pts": slip_total.sum(),
        "avg_slip_pts": slip_total.mean(),
        "median_slip_pts": slip_total.median(),
        "worst_slip_pts": slip_total.abs().max(),
        "avg_slip_bps": slip_total_bps.mean(),
        "total_slip_usd": df["slip_total_usd"].sum(),
        # Spread component
        "avg_spread_slip_pts": slip_spread.mean(),
        "total_spread_slip_usd": df["slip_spread_usd"].sum(),
        # Walkback component
        "avg_walkback_slip_pts": slip_walkback.mean(),
        "total_walkback_slip_usd": df["slip_walkback_usd"].sum(),
        # Execution quality
        "avg_spread_at_order": spread_at.mean(),
        "avg_fill_attempts": attempts.mean(),
        "max_fill_attempts": attempts.max(),
        # Commission
        "total_commission": commissions.sum(),
        "total_cost_usd": df["slip_total_usd"].sum() + commissions.sum(),
    }


def weekly_report(week_ending: datetime = None) -> str:
    """Generate a weekly slippage summary."""
    df = load_trade_log()
    if week_ending is None:
        week_ending = datetime.utcnow()

    week_start = week_ending - timedelta(days=7)
    mask = (df["date"] >= week_start) & (df["date"] <= week_ending)
    week_df = df[mask]
    cumulative = df[df["date"] <= week_ending]

    w = _summarize(week_df, f"Week ending {week_ending.strftime('%Y-%m-%d')}")
    c = _summarize(cumulative, "Cumulative")

    lines = [
        "=" * 60,
        f"  WEEKLY SLIPPAGE REPORT — {w['label']}",
        "=" * 60,
    ]

    if w["trades"] == 0:
        lines.append("  No trades this week.")
    else:
        lines.extend([
            f"  Trades this week:       {w['trades']}",
            f"  Avg fill attempts:      {w.get('avg_fill_attempts', 1):.1f}  (max {w.get('max_fill_attempts', 1):.0f})",
            f"  Avg spread at order:    {w.get('avg_spread_at_order', 0):.2f} pts",
            "",
            f"  --- Slippage Decomposition ---",
            f"  Total (fill vs close):  {w['total_slip_pts']:+.2f} pts  ${w['total_slip_usd']:+,.0f}",
            f"    Spread component:     {w.get('avg_spread_slip_pts', 0):+.2f} avg pts  ${w.get('total_spread_slip_usd', 0):+,.0f}",
            f"    Walkback component:   {w.get('avg_walkback_slip_pts', 0):+.2f} avg pts  ${w.get('total_walkback_slip_usd', 0):+,.0f}",
            f"  Avg slippage (pts):     {w['avg_slip_pts']:+.2f}",
            f"  Median slippage (pts):  {w['median_slip_pts']:+.2f}",
            f"  Worst slippage (pts):   {w['worst_slip_pts']:.2f}",
            f"  Avg slippage (bps):     {w['avg_slip_bps']:+.1f}",
            "",
            f"  Total commission:       ${w['total_commission']:,.0f}",
            f"  Total cost (USD):       ${w['total_cost_usd']:+,.0f}",
        ])

    lines.extend([
        "-" * 60,
        f"  CUMULATIVE (inception to date)",
        f"  Total trades:         {c.get('trades', 0)}",
        f"  Total slip cost:      ${c.get('total_slip_usd', 0):+,.0f}",
        f"  Total commission:     ${c.get('total_commission', 0):,.0f}",
        f"  Total drag (USD):     ${c.get('total_cost_usd', 0):+,.0f}",
        "=" * 60,
    ])

    report = "\n".join(lines)
    _save_report(report, f"weekly_{week_ending.strftime('%Y-%m-%d')}.txt")
    return report


def monthly_report(month_ending: datetime = None) -> str:
    """Generate a monthly slippage summary for Darwinex D-Score tracking."""
    df = load_trade_log()
    if month_ending is None:
        month_ending = datetime.utcnow()

    month_start = month_ending.replace(day=1)
    mask = (df["date"] >= month_start) & (df["date"] <= month_ending)
    month_df = df[mask]

    m = _summarize(month_df, f"Month: {month_ending.strftime('%Y-%m')}")

    lines = [
        "=" * 60,
        f"  MONTHLY SLIPPAGE REPORT — {m['label']}",
        "=" * 60,
    ]

    if m["trades"] == 0:
        lines.append("  No trades this month.")
    else:
        lines.extend([
            f"  Trades:                 {m['trades']}",
            f"  Avg fill attempts:      {m.get('avg_fill_attempts', 1):.1f}  (max {m.get('max_fill_attempts', 1):.0f})",
            f"  Avg spread at order:    {m.get('avg_spread_at_order', 0):.2f} pts",
            "",
            f"  --- Slippage Decomposition ---",
            f"  Total (fill vs close):  {m['total_slip_pts']:+.2f} pts  ${m['total_slip_usd']:+,.0f}",
            f"    Spread component:     {m.get('avg_spread_slip_pts', 0):+.2f} avg pts  ${m.get('total_spread_slip_usd', 0):+,.0f}",
            f"    Walkback component:   {m.get('avg_walkback_slip_pts', 0):+.2f} avg pts  ${m.get('total_walkback_slip_usd', 0):+,.0f}",
            f"  Avg slippage (pts):     {m['avg_slip_pts']:+.2f}",
            f"  Avg slippage (bps):     {m['avg_slip_bps']:+.1f}",
            "",
            f"  Total commission:       ${m['total_commission']:,.0f}",
            f"  Total drag:             ${m['total_cost_usd']:+,.0f}",
        ])

    lines.append("=" * 60)

    report = "\n".join(lines)
    _save_report(report, f"monthly_{month_ending.strftime('%Y-%m')}.txt")
    return report


def _save_report(report: str, filename: str) -> None:
    """Save report to reports/ directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        f.write(report)
    logger.info(f"report_saved={path}")


if __name__ == "__main__":
    setup_logging("slippage")

    parser = argparse.ArgumentParser(description="Slippage tracking reports")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly report")
    parser.add_argument("--monthly", action="store_true", help="Generate monthly report")
    parser.add_argument("--all", action="store_true", help="Print all per-trade slippage")
    args = parser.parse_args()

    if args.all:
        df = per_trade_report()
        print(df.to_string(index=False))
    if args.weekly:
        print(weekly_report())
    if args.monthly:
        print(monthly_report())
    if not (args.weekly or args.monthly or args.all):
        print("Usage: python slippage.py --weekly | --monthly | --all")
