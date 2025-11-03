"""Simple JSON logging for AudioChimp API."""

import logging
import sys
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str, log_level: str = "INFO") -> None:
    """
    Configure JSON-based logging with console output.

    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Simple formatter - just the message
    simple_formatter = logging.Formatter('%(message)s')

    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(simple_formatter)

    # Console handler - also simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)

    # Get AudioChimp logger (not root to avoid interfering with uvicorn)
    app_logger = logging.getLogger('auditchimp')
    app_logger.setLevel(level)
    app_logger.handlers.clear()
    app_logger.addHandler(file_handler)
    app_logger.addHandler(console_handler)
    app_logger.propagate = False  # Don't propagate to root logger

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('multipart').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('nemo').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def log_event(logger, status: str, event: str, message: str, **kwargs):
    """
    Log a JSON-formatted event.

    Args:
        logger: Logger instance
        status: success, error, warning, info
        event: Event name (e.g., "diarization_started", "transcription_completed")
        message: Human-readable message
        **kwargs: Additional data to include in payload
    """
    log_data = {
        "status": status,
        "event": event,
        "message": message
    }

    if kwargs:
        log_data["payload"] = kwargs

    log_message = json.dumps(log_data)

    if status == "error":
        logger.error(log_message)
    elif status == "warning":
        logger.warning(log_message)
    else:
        logger.info(log_message)
