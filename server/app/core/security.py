"""
Security utilities for authentication and authorization.
Handles JWT tokens, password hashing, refresh tokens, RBAC, and API key encryption.
Production-ready with refresh token rotation and Fernet encryption.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import secrets
import hashlib
from cryptography.fernet import Fernet
from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token
security = HTTPBearer()


# ========== Password Hashing ==========


def hash_password(password: str) -> str:
    """Hash a password using bcrypt. Truncates password to 72 bytes."""
    # Bcrypt has a 72-byte limit, truncate if necessary
    if isinstance(password, str):
        password_bytes = password.encode("utf-8")[:72]
        password = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash. Truncates password to 72 bytes."""
    # Bcrypt has a 72-byte limit, truncate if necessary
    if isinstance(plain_password, str):
        password_bytes = plain_password.encode("utf-8")[:72]
        plain_password = password_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(plain_password, hashed_password)


# ========== Token Creation ==========


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload data to encode (should include 'sub', 'role', etc.)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: int, role: str) -> str:
    """
    Create JWT refresh token with longer expiration.

    Args:
        user_id: User ID
        role: User role (STUDENT/ADMIN)

    Returns:
        Encoded JWT refresh token
    """
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32),  # Unique token ID for revocation
    }

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_reset_token() -> str:
    """
    Create secure password reset token.

    Returns:
        URL-safe token string
    """
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
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload
    except JWTError as e:
        logger.warning(f"Invalid {token_type} token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_refresh_token(token: str, db: Session) -> Dict[str, Any]:
    """
    Verify refresh token and check if it's revoked.

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

    # Check if token is revoked
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


# ========== RBAC Dependencies ==========


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from access token.

    Args:
        credentials: HTTP Authorization credentials
        db: Database session

    Returns:
        User data from token
    """
    token = credentials.credentials
    payload = decode_token(token, token_type="access")

    user_id = payload.get("sub")
    role = payload.get("role")

    if not user_id or not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    # Add user_id and role to payload for easy access
    payload["user_id"] = int(user_id)
    payload["user_role"] = role

    return payload


async def get_current_student(
    current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Dependency to get current student from database.
    Ensures user is a STUDENT and account is active.

    Args:
        current_user: Current user from token
        db: Database session

    Returns:
        Student model instance

    Raises:
        HTTPException: If not a student or account inactive
    """
    from app.models.user import Student, UserRole

    if current_user.get("user_role") != UserRole.STUDENT.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student access required")

    student = db.query(Student).filter(Student.id == current_user["user_id"]).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    if not student.isActive:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact administrator.",
        )

    return student


async def get_current_admin(
    current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Dependency to get current admin from database.
    Ensures user is an ADMIN and account is active.

    Args:
        current_user: Current user from token
        db: Database session

    Returns:
        Admin model instance

    Raises:
        HTTPException: If not an admin or account inactive
    """
    from app.models.user import Admin, UserRole

    if current_user.get("user_role") != UserRole.ADMIN.value:
        logger.warning(f"Non-admin user attempted admin action: {current_user.get('user_id')}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    admin = db.query(Admin).filter(Admin.id == current_user["user_id"]).first()
    if not admin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")

    if not admin.isActive:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin account is inactive"
        )

    return admin


async def require_premium_student(student=Depends(get_current_student)):
    """
    Dependency to ensure student has premium access.
    Use this to protect premium-only features.

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


# ========== API Key Verification ==========


def verify_api_key(api_key: Optional[str]) -> bool:
    """
    Verify API key for background workers or internal services.

    Args:
        api_key: API key to verify

    Returns:
        True if valid
    """
    # In production, use proper API key management
    # For now, simple check against configured key
    return api_key == settings.API_KEY if hasattr(settings, "API_KEY") else True


# ========== API Key Encryption ==========


# Generate or load encryption key
# In production, store this in environment variable or secret manager
def _get_encryption_key() -> bytes:
    """Get or generate Fernet encryption key."""
    # Try to get from settings/env
    if hasattr(settings, "ENCRYPTION_KEY") and settings.ENCRYPTION_KEY:
        return settings.ENCRYPTION_KEY.encode()

    # Generate a default key (NOT for production!)
    # In production, use: Fernet.generate_key() and store securely
    # Valid Fernet key: 32 url-safe base64-encoded bytes
    default_key = Fernet.generate_key()
    logger.warning("Using generated encryption key - NOT SECURE FOR PRODUCTION!")
    return default_key


_cipher_suite = Fernet(_get_encryption_key())


def encrypt_api_key(api_key: str) -> str:
    """
    Encrypt API key using Fernet symmetric encryption.

    Args:
        api_key: Plain text API key

    Returns:
        Encrypted API key (base64 encoded)
    """
    if not api_key:
        raise ValueError("API key cannot be empty")

    encrypted = _cipher_suite.encrypt(api_key.encode())
    return encrypted.decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """
    Decrypt API key.

    Args:
        encrypted_key: Encrypted API key (base64 encoded)

    Returns:
        Plain text API key
    """
    if not encrypted_key:
        raise ValueError("Encrypted key cannot be empty")

    try:
        decrypted = _cipher_suite.decrypt(encrypted_key.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {str(e)}")
        raise ValueError("Invalid encrypted key")


def hash_api_key(api_key: str) -> str:
    """
    Create SHA-256 hash of API key for deduplication.

    Args:
        api_key: Plain text API key

    Returns:
        Hex digest of SHA-256 hash
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """
    Mask API key for display (show first/last N chars).

    Args:
        api_key: Plain text API key
        visible_chars: Number of chars to show at start/end

    Returns:
        Masked key (e.g., "sk-ab...xyz")
    """
    if len(api_key) <= visible_chars * 2:
        return "*" * len(api_key)

    start = api_key[:visible_chars]
    end = api_key[-visible_chars:]
    return f"{start}...{end}"
    return api_key == settings.SECRET_KEY
