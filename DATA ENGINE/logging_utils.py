from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} | {message}"
)


def configure_logging(program_name: str, log_root: Path) -> Path:
    """Configure loguru for console and file logging.

    Args:
        program_name: Logical program identifier used in log filename.
        log_root: Directory where log files are written.

    Returns:
        Full path of the active log file.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    file_path = log_root / f"{datetime.now():%Y%m%d}_{program_name}.log"

    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level="INFO", colorize=False)
    logger.add(
        file_path,
        format=LOG_FORMAT,
        level="INFO",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    return file_path
