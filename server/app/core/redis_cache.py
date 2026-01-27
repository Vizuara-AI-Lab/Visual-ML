"""
Redis cache manager for user state and session management.
Provides high-performance caching to avoid database queries on every request.
"""

import json
from typing import Optional, Dict, Any
from datetime import timedelta
import redis.asyncio as redis
from app.core.config import settings
from app.core.logging import logger


class RedisCache:
    """
    Redis cache manager with automatic serialization and connection pooling.
    Handles user state caching and session management.
    """

    def __init__(self):
        """Initialize Redis connection pool."""
        self._redis: Optional[redis.Redis] = None
        self._enabled = settings.ENABLE_CACHE

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client with connection pooling."""
        if not self._enabled:
            raise RuntimeError("Redis caching is disabled")

        if self._redis is None:
            try:
                self._redis = await redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=50,  # Connection pool size
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                # Test connection
                await self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Redis connection failed: {str(e)}")
                raise

        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("Redis connection closed")

    # ========== User Caching ==========

    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user data from cache.

        Args:
            user_id: User ID

        Returns:
            User data dict or None if not found
        """
        if not self._enabled:
            return None

        try:
            client = await self._get_client()
            key = f"user:{user_id}"
            data = await client.get(key)

            if data:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(data)

            logger.debug(f"Cache MISS: {key}")
            return None

        except Exception as e:
            logger.warning(f"Redis get_user failed: {str(e)}, falling back to DB")
            return None

    async def set_user(self, user_id: int, user_data: Dict[str, Any], ttl: int = None) -> bool:
        """
        Cache user data.

        Args:
            user_id: User ID
            user_data: User data dictionary
            ttl: Time to live in seconds (default: settings.CACHE_TTL)

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            key = f"user:{user_id}"
            ttl = ttl or settings.CACHE_TTL

            # Serialize user data
            serialized = json.dumps(user_data, default=str)

            await client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Redis set_user failed: {str(e)}")
            return False

    async def delete_user(self, user_id: int) -> bool:
        """
        Invalidate user cache.

        Args:
            user_id: User ID

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            key = f"user:{user_id}"
            await client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True

        except Exception as e:
            logger.warning(f"Redis delete_user failed: {str(e)}")
            return False

    # ========== Session Management ==========

    async def create_session(
        self,
        jti: str,
        user_id: int,
        role: str,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None,
        ttl: int = None,
    ) -> bool:
        """
        Create session in Redis.

        Args:
            jti: JWT ID (unique token identifier)
            user_id: User ID
            role: User role (STUDENT/ADMIN)
            device_info: Device information
            ip_address: IP address
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            key = f"session:{jti}"
            ttl = ttl or (settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400)

            session_data = {
                "userId": user_id,
                "role": role,
                "deviceInfo": device_info,
                "ipAddress": ip_address,
                "createdAt": str(timedelta()),
            }

            serialized = json.dumps(session_data, default=str)
            await client.setex(key, ttl, serialized)
            logger.debug(f"Session created: {key}")
            return True

        except Exception as e:
            logger.warning(f"Redis create_session failed: {str(e)}")
            return False

    async def get_session(self, jti: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.

        Args:
            jti: JWT ID

        Returns:
            Session data dict or None if not found
        """
        if not self._enabled:
            return None

        try:
            client = await self._get_client()
            key = f"session:{jti}"
            data = await client.get(key)

            if data:
                return json.loads(data)

            return None

        except Exception as e:
            logger.warning(f"Redis get_session failed: {str(e)}")
            return None

    async def delete_session(self, jti: str) -> bool:
        """
        Delete session (logout).

        Args:
            jti: JWT ID

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            key = f"session:{jti}"
            await client.delete(key)
            logger.debug(f"Session deleted: {key}")
            return True

        except Exception as e:
            logger.warning(f"Redis delete_session failed: {str(e)}")
            return False

    async def delete_all_user_sessions(self, user_id: int) -> bool:
        """
        Delete all sessions for a user (logout all devices).

        Args:
            user_id: User ID

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            pattern = f"session:*"

            # Find all sessions
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)

                for key in keys:
                    session_data = await client.get(key)
                    if session_data:
                        session = json.loads(session_data)
                        if session.get("userId") == user_id:
                            await client.delete(key)
                            deleted_count += 1

                if cursor == 0:
                    break

            logger.info(f"Deleted {deleted_count} sessions for user {user_id}")
            return True

        except Exception as e:
            logger.warning(f"Redis delete_all_user_sessions failed: {str(e)}")
            return False

    # ========== Cache Statistics ==========

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dict with cache statistics
        """
        if not self._enabled:
            return {"enabled": False}

        try:
            client = await self._get_client()
            info = await client.info("stats")

            return {
                "enabled": True,
                "total_keys": await client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                    * 100
                ),
            }

        except Exception as e:
            logger.warning(f"Redis get_cache_stats failed: {str(e)}")
            return {"enabled": True, "error": str(e)}

    # ========== Generic Data Caching ==========

    async def get(self, key: str) -> Optional[Any]:
        """
        Get generic data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        if not self._enabled:
            return None

        try:
            client = await self._get_client()
            data = await client.get(key)

            if data:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(data)

            logger.debug(f"Cache MISS: {key}")
            return None

        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set generic data in cache.

        Args:
            key: Cache key
            value: Data to cache (will be JSON serialized)
            ttl: Time to live in seconds (default: settings.CACHE_TTL)

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            ttl = ttl or settings.CACHE_TTL

            # Serialize data
            serialized = json.dumps(value, default=str)

            await client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Redis set failed for {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete data from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            await client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True

        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {str(e)}")
            return False

    async def invalidate_pattern(self, pattern: str) -> bool:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "projects:student:*")

        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            cursor = 0
            deleted_count = 0

            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)

                if keys:
                    await client.delete(*keys)
                    deleted_count += len(keys)

                if cursor == 0:
                    break

            logger.info(f"Invalidated {deleted_count} keys matching pattern: {pattern}")
            return True

        except Exception as e:
            logger.warning(f"Redis invalidate_pattern failed for {pattern}: {str(e)}")
            return False


# Global Redis cache instance
redis_cache = RedisCache()
