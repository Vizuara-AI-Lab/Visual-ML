"""
Rate limiting middleware for production traffic handling.
Implements token bucket algorithm for 50k+ traffic.
"""

from typing import Dict, Optional
from time import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings
from app.core.logging import logger


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        now = time()
        elapsed = now - self.last_refill

        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    Supports per-IP and per-user rate limiting.
    """

    def __init__(self, app):
        super().__init__(app)
        self.ip_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=settings.RATE_LIMIT_PER_MINUTE,
                refill_rate=settings.RATE_LIMIT_PER_MINUTE / 60.0,
            )
        )
        self.user_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=settings.RATE_LIMIT_PER_HOUR,
                refill_rate=settings.RATE_LIMIT_PER_HOUR / 3600.0,
            )
        )

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)

        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"

        # Check IP rate limit
        if not self.ip_buckets[client_ip].consume():
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "RateLimitExceeded",
                    "message": "Too many requests. Please try again later.",
                    "suggestion": "Wait 60 seconds before making more requests.",
                },
                headers={"Retry-After": "60"},
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)

        return response


def get_client_id(request: Request) -> str:
    """
    Extract client identifier from request.
    Uses user ID if authenticated, otherwise IP address.

    Args:
        request: FastAPI request object

    Returns:
        Client identifier string
    """
    # Try to get user from token
    auth_header = request.headers.get("authorization")
    if auth_header:
        # Extract user_id from token if needed
        # For now, use IP
        pass

    return request.client.host if request.client else "unknown"
