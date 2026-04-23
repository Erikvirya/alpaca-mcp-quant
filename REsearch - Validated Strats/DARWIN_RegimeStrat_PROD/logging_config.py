"""
Shared logging configuration for the Darwinex Zero MT5 execution pipeline.
All modules use: logger = logging.getLogger(__name__)
Log format: key=value for AI supervisor parsing.
Daily rotation, 90-day retention.
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FORMAT = "%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: str = "execution", level: int = logging.INFO) -> None:
    """
    Configure root logger with console + daily rotating file output.
    Call once at the start of each entry-point script.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"{name}_{today}.log")

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler — daily rotation, keep 90 days
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=90, utc=True
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on re-init
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
