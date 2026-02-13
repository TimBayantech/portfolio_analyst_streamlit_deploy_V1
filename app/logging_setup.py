from __future__ import annotations
import logging

def get_logger() -> logging.Logger:
    logger = logging.getLogger("portfolio_analyst")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("trigger_events.log")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
