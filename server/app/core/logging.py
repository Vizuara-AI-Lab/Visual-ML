"""
Structured logging configuration using loguru.
Provides request tracing and production-ready logging.
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict
from contextvars import ContextVar
from loguru import logger
from app.core.config import settings

# Context variable for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context."""
    request_id_var.set(request_id)


def serialize_log_record(record: Dict[str, Any]) -> str:
    """
    Serialize log record to JSON format.
    """
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "request_id": get_request_id(),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add exception info if present
    if record["exception"]:
        subset["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
        }

    # Add extra fields
    if record.get("extra"):
        subset["extra"] = record["extra"]

    return json.dumps(subset)


def format_log_record(record: Dict[str, Any]) -> str:
    """
    Format log record based on settings.
    """
    if settings.LOG_FORMAT == "json":
        return serialize_log_record(record) + "\n"
    else:
        # Text format with request ID
        request_id = get_request_id()
        rid_prefix = f"[{request_id}] " if request_id else ""
        return f"<green>{record['time']:YYYY-MM-DD HH:mm:ss}</green> | <level>{record['level']:.4}</level> | {rid_prefix}<cyan>{record['name']}</cyan>:<cyan>{record['function']}</cyan>:<cyan>{record['line']}</cyan> - <level>{record['message']}</level>\n"


def setup_logging() -> None:
    """
    Configure logging for the application.
    """
    # Remove default handler
    logger.remove()

    # Use simple format for stdout
    if settings.LOG_FORMAT == "json":
        # Simple JSON format
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=settings.LOG_LEVEL,
            colorize=False,
        )
    else:
        # Text format
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:.4}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True,
        )

    # Add file handler for production
    if settings.ENVIRONMENT == "production":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logger.add(
            log_dir / "app_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="00:00",  # Rotate at midnight
            retention="30 days",
            compression="zip",
        )

    logger.info(f"Logging configured - Level: {settings.LOG_LEVEL}, Format: {settings.LOG_FORMAT}")


# Initialize logging
setup_logging()


def log_ml_operation(operation: str, details: Dict[str, Any], level: str = "info") -> None:
    """
    Log ML-specific operations with structured data.

    Args:
        operation: Name of the operation (e.g., "model_training", "prediction")
        details: Dictionary of operation details
        level: Log level (info, warning, error)
    """
    log_func = getattr(logger, level.lower())
    log_func(f"ML Operation: {operation}", extra=details)
