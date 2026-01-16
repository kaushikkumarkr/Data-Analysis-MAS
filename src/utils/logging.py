"""Structured logging for DataVault."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional

from src.utils.config import get_config


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as readable text.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {record.levelname:8} | {record.name} | {record.getMessage()}"

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


class DataVaultLogger(logging.Logger):
    """Custom logger with structured data support."""

    def _log_with_data(
        self,
        level: int,
        msg: str,
        data: Optional[dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log message with optional structured data.

        Args:
            level: Log level.
            msg: Log message.
            data: Optional structured data to include.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        extra = kwargs.get("extra", {})
        if data:
            extra["extra_data"] = data
        kwargs["extra"] = extra
        super().log(level, msg, *args, **kwargs)

    def info_with_data(
        self, msg: str, data: Optional[dict[str, Any]] = None, *args: Any, **kwargs: Any
    ) -> None:
        """Log info message with optional data.

        Args:
            msg: Log message.
            data: Optional structured data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._log_with_data(logging.INFO, msg, data, *args, **kwargs)

    def error_with_data(
        self, msg: str, data: Optional[dict[str, Any]] = None, *args: Any, **kwargs: Any
    ) -> None:
        """Log error message with optional data.

        Args:
            msg: Log message.
            data: Optional structured data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._log_with_data(logging.ERROR, msg, data, *args, **kwargs)

    def debug_with_data(
        self, msg: str, data: Optional[dict[str, Any]] = None, *args: Any, **kwargs: Any
    ) -> None:
        """Log debug message with optional data.

        Args:
            msg: Log message.
            data: Optional structured data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._log_with_data(logging.DEBUG, msg, data, *args, **kwargs)


def setup_logging(
    name: str = "datavault",
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> DataVaultLogger:
    """Set up and return a configured logger.

    Args:
        name: Logger name.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format type ('json' or 'text').

    Returns:
        Configured DataVaultLogger instance.
    """
    config = get_config()
    level = level or config.log_level
    log_format = log_format or config.log_format

    # Register custom logger class
    logging.setLoggerClass(DataVaultLogger)

    logger = logging.getLogger(name)
    logger.__class__ = DataVaultLogger

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)

    # Set formatter based on format type
    if log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# Create default logger
logger = setup_logging()


def get_logger(name: str) -> DataVaultLogger:
    """Get a child logger with the given name.

    Args:
        name: Child logger name.

    Returns:
        Configured DataVaultLogger instance.
    """
    return setup_logging(f"datavault.{name}")
