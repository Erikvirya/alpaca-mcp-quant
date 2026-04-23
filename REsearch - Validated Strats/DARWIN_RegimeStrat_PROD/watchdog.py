"""
Watchdog monitor for the Darwinex Zero NQ pipeline.
Runs every 5 min from 21:10-21:30 UTC Mon-Fri via Task Scheduler.
Covers both DST seasons (summer: trade done ~21:03, winter: ~22:03).

Checks:
1. Today's trade logged in trade_log.csv?
2. MT5 terminal process running?
3. Disk space > 1 GB free?

Sends email alert on any failure.
"""
import csv
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime

from logging_config import setup_logging
import notify

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(SCRIPT_DIR, "trade_log.csv")
MIN_DISK_GB = 1.0


def load_config() -> dict:
    """Load config.env as a dict."""
    config = {}
    config_path = os.path.join(SCRIPT_DIR, "config.env")
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


def check_trade_logged() -> bool:
    """Check if today's date appears in trade_log.csv."""
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if not os.path.exists(TRADE_LOG):
        return False
    try:
        with open(TRADE_LOG, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date", "").startswith(today_str):
                    return True
    except Exception:
        pass
    return False


def check_mt5_running() -> bool:
    """Check if MT5 terminal process is running (Windows)."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            capture_output=True, text=True, timeout=10
        )
        return "terminal64.exe" in result.stdout
    except Exception as e:
        logger.error(f"mt5_process_check_failed error={e}")
        return False


def check_disk_space() -> tuple:
    """Check free disk space. Returns (ok: bool, free_gb: float)."""
    try:
        usage = shutil.disk_usage(SCRIPT_DIR)
        free_gb = usage.free / (1024 ** 3)
        return free_gb >= MIN_DISK_GB, free_gb
    except Exception as e:
        logger.error(f"disk_check_failed error={e}")
        return False, 0.0


def run() -> None:
    """Main watchdog check."""
    setup_logging("watchdog")
    config = load_config()

    now = datetime.utcnow()
    logger.info(f"watchdog_start time={now.strftime('%H:%M:%S')}")

    # Skip weekends
    if now.weekday() >= 5:
        logger.info("reason=weekend skipping")
        return

    alerts = []

    # 1. Trade logged?
    trade_ok = check_trade_logged()
    if not trade_ok:
        minutes_since_trigger = (now.hour - 21) * 60 + now.minute - 2
        if minutes_since_trigger > 40:  # 30 min bar poll + 10 min buffer
            alert = f"MISSED TRADE: No entry in trade_log.csv for today ({now.strftime('%Y-%m-%d')}). Minutes since trigger: {minutes_since_trigger}"
            alerts.append(alert)
            logger.error(f"trade_logged=False minutes_since_trigger={minutes_since_trigger}")
        else:
            logger.info(f"trade_logged=False minutes_since_trigger={minutes_since_trigger} waiting")

    # 2. MT5 running?
    mt5_ok = check_mt5_running()
    if not mt5_ok:
        alert = "MT5 TERMINAL DOWN: terminal64.exe not found in process list"
        alerts.append(alert)
        logger.error("mt5_running=False process_running=False")
    else:
        logger.info("mt5_running=True")

    # 3. Disk space
    disk_ok, free_gb = check_disk_space()
    if not disk_ok:
        alert = f"LOW DISK SPACE: {free_gb:.1f} GB free (minimum: {MIN_DISK_GB} GB)"
        alerts.append(alert)
        logger.error(f"disk_ok=False free_gb={free_gb:.1f}")
    else:
        logger.info(f"disk_ok=True free_gb={free_gb:.1f}")

    # Send alerts
    if alerts:
        details = "\n\n".join(alerts)
        subj, body = notify.format_alert_msg("WATCHDOG ALERT", details)
        notify.send_email(subj, body, config)
    else:
        logger.info(f"trade_logged={trade_ok} mt5_running={mt5_ok} disk_ok={disk_ok}")

    logger.info("watchdog_end")


if __name__ == "__main__":
    run()
