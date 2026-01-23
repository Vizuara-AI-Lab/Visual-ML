"""
OTP verification functions for email verification.
"""

from datetime import datetime, timedelta
from typing import Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import random

from app.models.user import Student, AuthProvider
from app.schemas.auth import TokenResponse
from app.core.logging import logger


async def verify_email_otp(db: Session, email: str, otp: str) -> Tuple[Student, str, str]:
    """
    Verify student email with OTP and activate account.
    
    Args:
        db: Database session
        email: Student email
        otp: 6-digit OTP code
        
    Returns:
        Tuple of (Student model, access_token, refresh_token)
        
    Raises:
        HTTPException: If OTP invalid or expired
    """
    from app.services.auth_service import _create_tokens_for_student
    
    # Find student
    student = db.query(Student).filter(Student.emailId == email).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    # Check if already verified
    if student.isEmailVerified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified. You can login now."
        )
    
    # Check OTP
    if not student.verificationOTP or student.verificationOTP != otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP code"
        )
    
    # Check expiry
    if not student.otpExpiresAt or student.otpExpiresAt < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired. Please request a new one."
        )
    
    # Verify account
    student.isEmailVerified = True
    student.verificationOTP = None
    student.otpExpiresAt = None
    student.lastLogin = datetime.utcnow()
    
    db.commit()
    db.refresh(student)
    
    logger.info(f"Email verified for: {student.emailId}")
    
    # Send welcome email and schedule survey (async via Celery)
    try:
        from app.tasks.email_tasks import send_welcome_email_task, send_experience_survey_task
        
        # Send welcome email immediately
        send_welcome_email_task.delay(student.emailId, student.fullName)
        
        # Schedule experience survey for 10 days later
        send_experience_survey_task.apply_async(
            args=[student.emailId, student.fullName],
            countdown=10 * 24 * 60 * 60  # 10 days in seconds
        )
        
        logger.info(f"Welcome email and survey scheduled for: {student.emailId}")
    except Exception as e:
        logger.error(f"Failed to schedule emails: {str(e)}")
        # Don't fail verification if email scheduling fails
    
    # Generate tokens and create Redis session
    access_token, refresh_token, jti = await _create_tokens_for_student(db, student)
    
    return student, access_token, refresh_token


def resend_verification_otp(db: Session, email: str) -> None:
    """
    Resend OTP verification email.
    
    Args:
        db: Database session
        email: Student email
        
    Raises:
        HTTPException: If student not found or already verified
    """
    # Find student
    student = db.query(Student).filter(Student.emailId == email).first()
    if not student:
        # Don't reveal if email exists
        return
    
    # Check if already verified
    if student.isEmailVerified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified. You can login now."
        )
    
    # Generate new OTP
    otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)
    
    student.verificationOTP = otp
    student.otpExpiresAt = otp_expiry
    
    db.commit()
    
    logger.info(f"New OTP generated for: {student.emailId}")
    
    # Send OTP email
    from app.services.email_service import email_service
    try:
        email_service.send_verification_otp(student.emailId, student.fullName, otp)
        logger.info(f"OTP resent to {student.emailId}")
    except Exception as e:
        logger.error(f"Failed to resend OTP email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email. Please try again later."
        )
