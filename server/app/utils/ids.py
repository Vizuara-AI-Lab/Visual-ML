"""
Utility functions for ID generation.
Generates unique identifiers with timestamps and prefixes.
"""

import uuid
from datetime import datetime


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique ID string

    Examples:
        >>> generate_id("dataset")
        'dataset_20260119_a3b2c1d4'
        >>> generate_id()
        'e5f6g7h8'
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]

    if prefix:
        return f"{prefix}_{timestamp}_{unique_suffix}"
    else:
        return f"{timestamp}_{unique_suffix}"


def generate_request_id() -> str:
    """
    Generate a request ID for tracking.

    Returns:
        Request ID string
    """
    return f"req_{uuid.uuid4().hex[:12]}"
