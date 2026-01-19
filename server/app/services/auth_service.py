"""
Authentication service with complete business logic.
Handles student/admin registration, login, Google OAuth, and token management.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import secrets

from app.models.user import Student, Admin, RefreshToken, UserRole, AuthProvider
from app.schemas.auth import (
    StudentRegister,
    StudentLogin,
    StudentGoogleAuth,
    AdminLogin,
    TokenResponse,
    StudentResponse,
    AdminResponse,
)
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    create_reset_token,
)
from app.core.config import settings
from app.core.logging import logger


# ========== Student Authentication ==========


def register_student(db: Session, data: StudentRegister) -> Tuple[Student, TokenResponse]:
    """
    Register a new student with email/password.

    Args:
        db: Database session
        data: Student registration data

    Returns:
        Tuple of (Student model, TokenResponse)

    Raises:
        HTTPException: If email already exists
    """
    # Check if email already exists
    existing = db.query(Student).filter(Student.emailId == data.emailId).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create student
    student = Student(
        emailId=data.emailId,
        password=hash_password(data.password),
        role=UserRole.STUDENT,
        authProvider=AuthProvider.LOCAL,
        collegeOrSchool=data.collegeOrSchool,
        contactNo=data.contactNo,
        isPremium=False,
        isActive=True,
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    logger.info(f"New student registered: {student.emailId}")

    # Generate tokens
    tokens = _create_tokens_for_student(db, student)

    return student, tokens


def login_student(
    db: Session,
    data: StudentLogin,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Tuple[Student, TokenResponse]:
    """
    Login student with email/password.

    Args:
        db: Database session
        data: Student login credentials
        device_info: Optional device information
        ip_address: Optional IP address

    Returns:
        Tuple of (Student model, TokenResponse)

    Raises:
        HTTPException: If credentials invalid or account inactive
    """
    # Find student
    student = db.query(Student).filter(Student.emailId == data.emailId).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Check account status
    if not student.isActive:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact administrator.",
        )

    # Verify password (only for LOCAL auth)
    if student.authProvider == AuthProvider.LOCAL:
        if not student.password or not verify_password(data.password, student.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This account uses {student.authProvider.value} authentication",
        )

    # Update last login
    student.lastLogin = datetime.utcnow()
    db.commit()

    logger.info(f"Student logged in: {student.emailId}")

    # Generate tokens
    tokens = _create_tokens_for_student(db, student, device_info, ip_address)

    return student, tokens


def google_auth_student(
    db: Session,
    data: StudentGoogleAuth,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Tuple[Student, TokenResponse]:
    """
    Authenticate student with Google OAuth.
    Creates new student if doesn't exist.

    Args:
        db: Database session
        data: Google auth data with ID token
        device_info: Optional device information
        ip_address: Optional IP address

    Returns:
        Tuple of (Student model, TokenResponse)

    Raises:
        HTTPException: If Google token invalid
    """
    try:
        # Verify Google ID token
        id_info = id_token.verify_oauth2_token(
            data.idToken, google_requests.Request(), settings.GOOGLE_CLIENT_ID
        )

        google_id = id_info["sub"]
        email = id_info["email"]
        profile_pic = data.profilePic or id_info.get("picture")

        # Check if student exists with this Google ID
        student = db.query(Student).filter(Student.googleId == google_id).first()

        if student:
            # Update existing student
            student.lastLogin = datetime.utcnow()
            if profile_pic:
                student.profilePic = profile_pic
            db.commit()
            logger.info(f"Student Google login: {student.emailId}")
        else:
            # Check if email exists with different auth provider
            existing_email = db.query(Student).filter(Student.emailId == email).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered with different authentication method",
                )

            # Create new student
            student = Student(
                emailId=email,
                googleId=google_id,
                authProvider=AuthProvider.GOOGLE,
                role=UserRole.STUDENT,
                profilePic=profile_pic,
                isPremium=False,
                isActive=True,
            )
            db.add(student)
            db.commit()
            db.refresh(student)
            logger.info(f"New student registered via Google: {student.emailId}")

        # Generate tokens
        tokens = _create_tokens_for_student(db, student, device_info, ip_address)

        return student, tokens

    except ValueError as e:
        logger.error(f"Google token verification failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token")


# ========== Admin Authentication ==========


def login_admin(
    db: Session,
    data: AdminLogin,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Tuple[Admin, TokenResponse]:
    """
    Login admin with email/password.

    Args:
        db: Database session
        data: Admin login credentials
        device_info: Optional device information
        ip_address: Optional IP address

    Returns:
        Tuple of (Admin model, TokenResponse)

    Raises:
        HTTPException: If credentials invalid or account inactive
    """
    # Find admin
    admin = db.query(Admin).filter(Admin.email == data.email).first()
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Check account status
    if not admin.isActive:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin account is inactive"
        )

    # Verify password
    if not verify_password(data.password, admin.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Update last login
    admin.lastLogin = datetime.utcnow()
    db.commit()

    logger.info(f"Admin logged in: {admin.email}")

    # Generate tokens
    tokens = _create_tokens_for_admin(db, admin, device_info, ip_address)

    return admin, tokens


# ========== Token Management ==========


def refresh_access_token(
    db: Session,
    refresh_token_str: str,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    Implements token rotation for security.

    Args:
        db: Database session
        refresh_token_str: Refresh token JWT
        device_info: Optional device information
        ip_address: Optional IP address

    Returns:
        New TokenResponse with rotated tokens

    Raises:
        HTTPException: If refresh token invalid or revoked
    """
    from app.core.security import verify_refresh_token

    # Verify refresh token
    payload = verify_refresh_token(refresh_token_str, db)

    user_id = int(payload.get("sub"))
    role = payload.get("role")
    jti = payload.get("jti")

    # Revoke old refresh token (token rotation)
    old_token = db.query(RefreshToken).filter(RefreshToken.token == jti).first()
    if old_token:
        old_token.isRevoked = True
        db.commit()

    # Create new tokens
    if role == UserRole.STUDENT.value:
        student = db.query(Student).filter(Student.id == user_id).first()
        if not student or not student.isActive:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student account not found or inactive",
            )
        return _create_tokens_for_student(db, student, device_info, ip_address)
    else:
        admin = db.query(Admin).filter(Admin.id == user_id).first()
        if not admin or not admin.isActive:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin account not found or inactive"
            )
        return _create_tokens_for_admin(db, admin, device_info, ip_address)


def logout_user(db: Session, refresh_token_str: str) -> None:
    """
    Logout user by revoking refresh token.

    Args:
        db: Database session
        refresh_token_str: Refresh token JWT
    """
    from app.core.security import decode_token

    try:
        payload = decode_token(refresh_token_str, token_type="refresh")
        jti = payload.get("jti")

        # Revoke token
        token = db.query(RefreshToken).filter(RefreshToken.token == jti).first()
        if token:
            token.isRevoked = True
            db.commit()
            logger.info(f"User logged out, token revoked: {jti}")
    except Exception as e:
        logger.warning(f"Logout failed: {str(e)}")
        # Don't raise exception, just log


# ========== Password Management ==========


def initiate_password_reset(db: Session, email: str) -> str:
    """
    Initiate password reset for student.

    Args:
        db: Database session
        email: Student email

    Returns:
        Reset token (to be sent via email in production)

    Raises:
        HTTPException: If student not found
    """
    student = db.query(Student).filter(Student.emailId == email).first()
    if not student:
        # Don't reveal if email exists
        raise HTTPException(
            status_code=status.HTTP_200_OK, detail="If email exists, reset link has been sent"
        )

    # Only allow reset for LOCAL auth
    if student.authProvider != AuthProvider.LOCAL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password reset not available for {student.authProvider.value} accounts",
        )

    # Generate reset token
    reset_token = create_reset_token()
    student.resetToken = reset_token
    student.resetTokenExpiry = datetime.utcnow() + timedelta(
        minutes=settings.RESET_TOKEN_EXPIRE_MINUTES
    )

    db.commit()

    logger.info(f"Password reset initiated for: {email}")

    # In production, send email here
    # For now, return token (remove in production!)
    return reset_token


def reset_password(db: Session, token: str, new_password: str) -> None:
    """
    Reset student password using reset token.

    Args:
        db: Database session
        token: Reset token
        new_password: New password

    Raises:
        HTTPException: If token invalid or expired
    """
    student = db.query(Student).filter(Student.resetToken == token).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token"
        )

    # Check expiry
    if student.resetTokenExpiry < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Reset token has expired"
        )

    # Update password
    student.password = hash_password(new_password)
    student.resetToken = None
    student.resetTokenExpiry = None

    # Revoke all refresh tokens for security
    db.query(RefreshToken).filter(RefreshToken.studentId == student.id).update({"isRevoked": True})

    db.commit()

    logger.info(f"Password reset successful for: {student.emailId}")


def change_password(db: Session, student: Student, old_password: str, new_password: str) -> None:
    """
    Change student password.

    Args:
        db: Database session
        student: Student model
        old_password: Current password
        new_password: New password

    Raises:
        HTTPException: If old password incorrect
    """
    # Only for LOCAL auth
    if student.authProvider != AuthProvider.LOCAL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password change not available for OAuth accounts",
        )

    # Verify old password
    if not student.password or not verify_password(old_password, student.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
        )

    # Update password
    student.password = hash_password(new_password)

    # Revoke all refresh tokens for security
    db.query(RefreshToken).filter(RefreshToken.studentId == student.id).update({"isRevoked": True})

    db.commit()

    logger.info(f"Password changed for: {student.emailId}")


# ========== Helper Functions ==========


def _create_tokens_for_student(
    db: Session,
    student: Student,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> TokenResponse:
    """Create access and refresh tokens for student."""
    # Create access token
    access_token = create_access_token(
        {"sub": str(student.id), "role": UserRole.STUDENT.value, "email": student.emailId}
    )

    # Create refresh token
    refresh_token = create_refresh_token(student.id, UserRole.STUDENT.value)

    # Decode to get JTI and expiry
    from jose import jwt

    refresh_payload = jwt.decode(
        refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
    )

    # Store refresh token in database
    db_refresh_token = RefreshToken(
        token=refresh_payload["jti"],
        studentId=student.id,
        expiresAt=datetime.fromtimestamp(refresh_payload["exp"]),
        deviceInfo=device_info,
        ipAddress=ip_address,
    )
    db.add(db_refresh_token)
    db.commit()

    return TokenResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        tokenType="bearer",
        expiresIn=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def _create_tokens_for_admin(
    db: Session, admin: Admin, device_info: Optional[str] = None, ip_address: Optional[str] = None
) -> TokenResponse:
    """Create access and refresh tokens for admin."""
    # Create access token
    access_token = create_access_token(
        {"sub": str(admin.id), "role": UserRole.ADMIN.value, "email": admin.email}
    )

    # Create refresh token
    refresh_token = create_refresh_token(admin.id, UserRole.ADMIN.value)

    # Decode to get JTI and expiry
    from jose import jwt

    refresh_payload = jwt.decode(
        refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
    )

    # Store refresh token in database
    db_refresh_token = RefreshToken(
        token=refresh_payload["jti"],
        adminId=admin.id,
        expiresAt=datetime.fromtimestamp(refresh_payload["exp"]),
        deviceInfo=device_info,
        ipAddress=ip_address,
    )
    db.add(db_refresh_token)
    db.commit()

    return TokenResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        tokenType="bearer",
        expiresIn=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
