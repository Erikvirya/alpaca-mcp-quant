"""
Email notification helper for the Darwinex Zero MT5 execution pipeline.
Uses Python smtplib + email.mime — no extra dependencies.
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str, config: dict) -> bool:
    """
    Send an email via SMTP.

    Args:
        subject: Email subject line.
        body: Plain-text email body.
        config: dict with keys: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO

    Returns:
        True if sent successfully, False otherwise.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = config["SMTP_USER"]
        msg["To"] = config["EMAIL_TO"]
        msg["Subject"] = f"[DarwinexZero] {subject}"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(config["SMTP_HOST"], int(config["SMTP_PORT"]), timeout=30) as server:
            server.starttls()
            server.login(config["SMTP_USER"], config["SMTP_PASSWORD"])
            server.send_message(msg)

        logger.info(f"email_sent=True subject={subject}")
        return True
    except Exception as e:
        logger.error(f"email_sent=False subject={subject} error={e}")
        return False


def format_trade_msg(
    date: str, regime: str, alloc: float, target_cts: int, prev_cts: int,
    delta: int, fill_price: float, signal_close: float, slippage_pts: float,
    slippage_bps: float, equity: float, vxn: float, contract_symbol: str,
) -> tuple:
    """Format a trade execution email. Returns (subject, body)."""
    action = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"
    subject = f"{action} {abs(delta)} {contract_symbol} | {regime} | alloc={alloc:.2f}"
    body = (
        f"Date:            {date}\n"
        f"Regime:          {regime}\n"
        f"Allocation:      {alloc:.2f}x\n"
        f"Target Cts:      {target_cts}\n"
        f"Previous Cts:    {prev_cts}\n"
        f"Delta:           {delta:+d}\n"
        f"Contract:        {contract_symbol}\n"
        f"Fill Price:      {fill_price:.2f}\n"
        f"Signal Close:    {signal_close:.2f}\n"
        f"Slippage (pts):  {slippage_pts:+.2f}\n"
        f"Slippage (bps):  {slippage_bps:+.1f}\n"
        f"Equity:          ${equity:,.0f}\n"
        f"VXN:             {vxn:.2f}\n"
    )
    return subject, body


def format_alert_msg(alert_type: str, details: str) -> tuple:
    """Format an alert/error email. Returns (subject, body)."""
    subject = f"ALERT: {alert_type}"
    body = f"Alert Type: {alert_type}\n\nDetails:\n{details}"
    return subject, body


def format_heartbeat_msg(
    date: str, position: int, contract_symbol: str, equity: float, vxn: float,
) -> tuple:
    """Format a daily heartbeat email (no position change). Returns (subject, body)."""
    subject = f"HEARTBEAT OK | pos={position} {contract_symbol}"
    body = (
        f"Date:            {date}\n"
        f"Position:        {position} {contract_symbol}\n"
        f"Equity:          ${equity:,.0f}\n"
        f"VXN:             {vxn:.2f}\n"
        f"Status:          No position change — all systems OK\n"
    )
    return subject, body
