"""
Production-grade security utilities for authentication and authorization.
Implements HTTP-only cookie-based auth with Redis caching for 50k+ users.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request, Response
from sqlalchemy.orm import Session
import secrets
import hashlib
import time
from cryptography.fernet import Fernet
from app.core.config import settings
from app.core.logging import logger
from app.core.redis_cache import redis_cache
from app.db.session import get_db

# Password hashing
# Password hashing - Argon2 for new, Bcrypt for legacy
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
)


# ========== Password Hashing ==========


def hash_password(password: str) -> str:
    """Hash a password using bcrypt with optimized rounds."""
    if isinstance(password, str):
        password_bytes = password.encode("utf-8")[:72]
        password = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if isinstance(plain_password, str):
        password_bytes = plain_password.encode("utf-8")[:72]
        plain_password = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(plain_password, hashed_password)


# ========== Token Creation ==========


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token for HTTP-only cookie.

    Args:
        data: Payload data (sub, role, email)
        expires_delta: Optional custom expiration

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_urlsafe(16),  # Unique token ID
        }
    )

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: int, role: str) -> tuple[str, str]:
    """
    Create JWT refresh token with unique JTI for session tracking.

    Args:
        user_id: User ID
        role: User role (STUDENT/ADMIN)

    Returns:
        Tuple of (refresh_token, jti)
    """
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    jti = secrets.token_urlsafe(32)  # Unique token ID for revocation

    to_encode = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": jti,
    }

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt, jti


def create_reset_token() -> str:
    """Create secure password reset token."""
    return secrets.token_urlsafe(32)


# ========== Token Validation ==========


def decode_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string
        token_type: Expected token type ('access' or 'refresh')

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid, expired, or wrong type
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        # Verify token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}",
            )

        return payload
    except JWTError as e:
        logger.warning(f"Invalid {token_type} token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


def get_token_from_request(request: Request) -> str:
    """
    Extract token from HTTP-only cookie or Authorization header.
    Supports both browser (cookie) and mobile (header) clients.

    Args:
        request: FastAPI request object

    Returns:
        JWT token string

    Raises:
        HTTPException: If no token found
    """
    # Try cookie first (browser)
    token = request.cookies.get("access_token")
    if token:
        logger.debug("Token extracted from cookie")
        return token

    # Fallback to Authorization header (mobile apps)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        logger.debug("Token extracted from Authorization header")
        return token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. No token found in cookie or header.",
    )


async def verify_refresh_token(token: str, db: Session) -> Dict[str, Any]:
    """
    Verify refresh token and check if it's revoked in database.

    Args:
        token: Refresh token string
        db: Database session

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or revoked
    """
    from app.models.user import RefreshToken

    payload = decode_token(token, token_type="refresh")
    jti = payload.get("jti")

    # Check if token is revoked in database
    db_token = db.query(RefreshToken).filter(RefreshToken.token == jti).first()
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not found"
        )

    if db_token.isRevoked:
        logger.warning(f"Attempted use of revoked refresh token: {jti}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has been revoked"
        )

    if db_token.expiresAt < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has expired"
        )

    return payload


# ========== Cookie Management ==========


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
):
    """
    Set secure HTTP-only cookies for authentication.

    Args:
        response: FastAPI response object
        access_token: JWT access token
        refresh_token: JWT refresh token
    """
    is_production = settings.ENVIRONMENT == "production"
    cookie_domain = getattr(settings, "COOKIE_DOMAIN", None)
    # Use 'none' for cross-origin requests in production, 'lax' for same-origin in dev
    samesite_policy = "none" if is_production else "lax"

    # Access token cookie (short-lived)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,  # Prevent JavaScript access (XSS protection)
        secure=is_production,  # HTTPS only in production
        samesite=samesite_policy,  # 'none' for cross-origin in production
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        domain=cookie_domain,
    )

    # Refresh token cookie (long-lived)
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=is_production,
        samesite=samesite_policy,  # 'none' for cross-origin in production
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/v1/auth",  # Only sent to auth endpoints
        domain=cookie_domain,
    )

    logger.debug("Auth cookies set successfully")


def clear_auth_cookies(response: Response):
    """
    Clear authentication cookies (logout).

    Args:
        response: FastAPI response object
    """
    cookie_domain = getattr(settings, "COOKIE_DOMAIN", None)

    response.delete_cookie(
        key="access_token",
        path="/",
        domain=cookie_domain,
    )

    response.delete_cookie(
        key="refresh_token",
        path="/api/v1/auth",
        domain=cookie_domain,
    )

    logger.debug("Auth cookies cleared")


# ========== RBAC Dependencies with Redis Caching ==========


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from token.
    Extracts token from cookie or Authorization header.

    Args:
        request: FastAPI request object

    Returns:
        User data from token payload
    """
    start_time = time.time()

    token = get_token_from_request(request)
    payload = decode_token(token, token_type="access")

    user_id = payload.get("sub")
    role = payload.get("role")

    if not user_id or not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    # Add user_id and role to payload
    payload["user_id"] = int(user_id)
    payload["user_role"] = role

    duration_ms = (time.time() - start_time) * 1000
    logger.debug(f"JWT verification took {duration_ms:.2f}ms")

    return payload


async def get_current_student(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Dependency to get current student from JWT claims (NO DATABASE QUERY).
    Ensures user is a STUDENT and account is active.

    Performance: ~2-5ms (JWT decode only, no DB/Redis)

    Args:
        request: FastAPI request object
        current_user: Current user from token
        db: Database session (not used, kept for compatibility)

    Returns:
        Student-like object with data from JWT

    Raises:
        HTTPException: If not a student or account inactive
    """
    from app.models.user import UserRole

    start_time = time.time()

    if current_user.get("role") != UserRole.STUDENT.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student access required")

    # Check if account is active
    if not current_user.get("isActive", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact administrator.",
        )

    # Create a lightweight Student object from JWT claims (NO DATABASE QUERY!)
    class JWTStudent:
        """Lightweight student object from JWT claims"""

        def __init__(self, claims: Dict[str, Any]):
            self.id = int(claims.get("sub"))
            self.emailId = claims.get("email")
            self.fullName = claims.get("fullName", "")
            self.role = UserRole.STUDENT
            self.isActive = claims.get("isActive", True)
            self.isPremium = claims.get("isPremium", False)
            self.isEmailVerified = claims.get("isEmailVerified", False)

    student = JWTStudent(current_user)

    duration_ms = (time.time() - start_time) * 1000
    logger.debug(f"get_current_student (JWT ONLY) took {duration_ms:.2f}ms")

    return student


async def get_optional_current_student(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[Any]:
    """
    Dependency to optionally get current student (allows anonymous access).
    Returns student if authenticated, None if not authenticated.

    Used for endpoints that allow both authenticated and anonymous access.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Student-like object from JWT if authenticated, None otherwise
    """
    try:
        # Try to get current user from token
        current_user = await get_current_user(request)
        # Then validate student access
        return await get_current_student(request, current_user, db)
    except (HTTPException, JWTError):
        # Anonymous access allowed
        return None


async def get_current_admin(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Dependency to get current admin with Redis caching.
    Ensures user is an ADMIN and account is active.

    Args:
        request: FastAPI request object
        current_user: Current user from token
        db: Database session

    Returns:
        Admin model instance

    Raises:
        HTTPException: If not an admin or account inactive
    """
    from app.models.user import Admin, UserRole

    start_time = time.time()

    if current_user.get("user_role") != UserRole.ADMIN.value:
        logger.warning(f"Non-admin user attempted admin action: {current_user.get('user_id')}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    user_id = current_user["user_id"]

    # Try Redis cache first
    cached_user = await redis_cache.get_user(user_id)

    if cached_user:
        admin = Admin(**cached_user)

        if not admin.isActive:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin account is inactive"
            )

        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"get_current_admin (CACHE HIT) took {duration_ms:.2f}ms")

        return admin

    # Cache miss - query database
    admin = db.query(Admin).filter(Admin.id == user_id).first()
    if not admin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")

    if not admin.isActive:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin account is inactive"
        )

    # Cache admin data
    admin_dict = {
        "id": admin.id,
        "email": admin.email,
        "name": admin.name,
        "role": admin.role.value,
        "isActive": admin.isActive,
        "createdAt": str(admin.createdAt),
        "lastLogin": str(admin.lastLogin) if admin.lastLogin else None,
    }

    await redis_cache.set_user(user_id, admin_dict, ttl=settings.CACHE_TTL)

    duration_ms = (time.time() - start_time) * 1000
    logger.debug(f"get_current_admin (CACHE MISS) took {duration_ms:.2f}ms")

    return admin


async def require_premium_student(student=Depends(get_current_student)):
    """
    Dependency to ensure student has premium access.

    Args:
        student: Current student

    Returns:
        Student model instance

    Raises:
        HTTPException: If student is not premium
    """
    if not student.isPremium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Premium access required for this feature"
        )

    return student


# ========== API Key Encryption (unchanged) ==========


def _get_encryption_key() -> bytes:
    """Get or generate Fernet encryption key."""
    if hasattr(settings, "ENCRYPTION_KEY") and settings.ENCRYPTION_KEY:
        return settings.ENCRYPTION_KEY.encode()

    default_key = Fernet.generate_key()
    logger.warning("Using generated encryption key - NOT SECURE FOR PRODUCTION!")
    return default_key


_cipher_suite = Fernet(_get_encryption_key())


def encrypt_api_key(api_key: str) -> str:
    """Encrypt API key using Fernet symmetric encryption."""
    if not api_key:
        raise ValueError("API key cannot be empty")

    encrypted = _cipher_suite.encrypt(api_key.encode())
    return encrypted.decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt API key."""
    if not encrypted_key:
        raise ValueError("Encrypted key cannot be empty")

    try:
        decrypted = _cipher_suite.decrypt(encrypted_key.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {str(e)}")
        raise ValueError("Invalid encrypted key")


def hash_api_key(api_key: str) -> str:
    """Create SHA-256 hash of API key for deduplication."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """Mask API key for display (show first/last N chars)."""
    if len(api_key) <= visible_chars * 2:
        return "*" * len(api_key)

    start = api_key[:visible_chars]
    end = api_key[-visible_chars:]
    return f"{start}...{end}"
