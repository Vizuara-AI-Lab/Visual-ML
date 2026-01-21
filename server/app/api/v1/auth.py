"""
Authentication API endpoints for students and admins.
Implements complete RBAC authentication with Google OAuth.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.models.user import Student, Admin
from app.schemas.auth import (
    StudentRegister,
    StudentLogin,
    StudentGoogleAuth,
    AdminLogin,
    AdminRegister,
    TokenResponse,
    RefreshTokenRequest,
    StudentResponse,
    AdminResponse,
    StudentUpdate,
    ChangePassword,
    ForgotPassword,
    ResetPassword,
    AuthResponse,
    StudentListItem,
    AdminUpdateStudent,
)
from app.services import auth_service
from app.core.security import get_current_student, get_current_admin
from app.core.logging import logger


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ========== Student Registration & Login ==========


@router.post("/student/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_student(data: StudentRegister, request: Request, db: Session = Depends(get_db)):
    """
    Register a new student with email and password.

    - **emailId**: Unique email address
    
    - **password**: Min 8 chars with uppercase, lowercase, digit
        - **Name**: compulsory full name

    - **collegeOrSchool**: Optional college/school name
    - **contactNo**: Optional contact number

    Returns access token and refresh token.
    """
    student, tokens = auth_service.register_student(db, data)

    return AuthResponse(
        user=StudentResponse.model_validate(student),
        tokens=tokens,
        message="Student registered successfully",
    )


@router.post("/student/login", response_model=AuthResponse)
async def login_student(data: StudentLogin, request: Request, db: Session = Depends(get_db)):
    """
    Login student with email and password.

    Returns access token and refresh token.
    """
    device_info = request.headers.get("User-Agent")
    ip_address = request.client.host if request.client else None

    student, tokens = auth_service.login_student(db, data, device_info, ip_address)

    return AuthResponse(
        user=StudentResponse.model_validate(student), tokens=tokens, message="Login successful"
    )


@router.post("/student/google", response_model=AuthResponse)
async def google_auth_student(
    data: StudentGoogleAuth, request: Request, db: Session = Depends(get_db)
):
    """
    Authenticate student with Google OAuth.
    Creates new account if doesn't exist.

    - **idToken**: Google ID token from frontend (Google Sign-In)
    - **profilePic**: Optional profile picture URL

    Returns access token and refresh token.
    """
    device_info = request.headers.get("User-Agent")
    ip_address = request.client.host if request.client else None

    student, tokens = auth_service.google_auth_student(db, data, device_info, ip_address)

    return AuthResponse(
        user=StudentResponse.model_validate(student),
        tokens=tokens,
        message="Google authentication successful",
    )


# ========== Admin Login ==========


@router.post("/admin/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_admin(data: AdminRegister, request: Request, db: Session = Depends(get_db)):
    """
    Register a new admin account.

    - **email**: Unique admin email
    - **password**: Min 8 chars with uppercase, lowercase, digit
    - **name**: Optional admin name

    Returns access token and refresh token.
    """
    admin, tokens = auth_service.register_admin(db, data)

    return AuthResponse(
        user=AdminResponse.model_validate(admin),
        tokens=tokens,
        message="Admin registered successfully",
    )


@router.post("/admin/login", response_model=AuthResponse)
async def login_admin(data: AdminLogin, request: Request, db: Session = Depends(get_db)):
    """
    Login admin with email and password.

    Returns access token and refresh token.
    """
    device_info = request.headers.get("User-Agent")
    ip_address = request.client.host if request.client else None

    admin, tokens = auth_service.login_admin(db, data, device_info, ip_address)

    return AuthResponse(
        user=AdminResponse.model_validate(admin), tokens=tokens, message="Admin login successful"
    )


# ========== Token Management ==========


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(data: RefreshTokenRequest, request: Request, db: Session = Depends(get_db)):
    """
    Refresh access token using refresh token.
    Implements token rotation - old refresh token is revoked.

    Returns new access token and refresh token.
    """
    device_info = request.headers.get("User-Agent")
    ip_address = request.client.host if request.client else None

    tokens = auth_service.refresh_access_token(db, data.refreshToken, device_info, ip_address)

    return tokens


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(data: RefreshTokenRequest, db: Session = Depends(get_db)):
    """
    Logout user by revoking refresh token.
    """
    auth_service.logout_user(db, data.refreshToken)


# ========== Student Profile Management ==========


@router.get("/student/me", response_model=StudentResponse)
async def get_student_profile(student: Student = Depends(get_current_student)):
    """
    Get current student profile.
    Requires valid access token.
    """
    return StudentResponse.model_validate(student)


@router.patch("/student/me", response_model=StudentResponse)
async def update_student_profile(
    data: StudentUpdate,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Update current student profile.
    Cannot change emailId (it's the unique identifier).

    - **collegeOrSchool**: College or school name
    - **contactNo**: Contact number
    - **recentProject**: Recent project description
    - **profilePic**: Profile picture URL
    """
    # Update fields
    if data.collegeOrSchool is not None:
        student.collegeOrSchool = data.collegeOrSchool
    if data.contactNo is not None:
        student.contactNo = data.contactNo
    if data.recentProject is not None:
        student.recentProject = data.recentProject
    if data.profilePic is not None:
        student.profilePic = data.profilePic

    db.commit()
    db.refresh(student)

    logger.info(f"Student profile updated: {student.emailId}")

    return StudentResponse.model_validate(student)


@router.post("/student/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    data: ChangePassword,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Change student password.
    Only for students with LOCAL authentication.
    Revokes all refresh tokens for security.
    """
    auth_service.change_password(db, student, data.oldPassword, data.newPassword)


@router.post("/student/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(data: ForgotPassword, db: Session = Depends(get_db)):
    """
    Initiate password reset.
    Sends reset token (in production, via email).

    Returns success message even if email doesn't exist (security).
    """
    reset_token = auth_service.initiate_password_reset(db, data.emailId)

    # In production, send email here and don't return token
    # For development, return token
    return {
        "message": "If email exists, reset link has been sent",
        "resetToken": reset_token,  # Remove in production!
    }


@router.post("/student/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(data: ResetPassword, db: Session = Depends(get_db)):
    """
    Reset password using reset token.
    Revokes all refresh tokens for security.
    """
    auth_service.reset_password(db, data.token, data.newPassword)


# ========== Admin Profile & Student Management ==========


@router.get("/admin/me", response_model=AdminResponse)
async def get_admin_profile(admin: Admin = Depends(get_current_admin)):
    """
    Get current admin profile.
    Requires admin access token.
    """
    return AdminResponse.model_validate(admin)


@router.get("/admin/students", response_model=List[StudentListItem])
async def list_students(
    skip: int = 0,
    limit: int = 100,
    search: str = None,
    isPremium: bool = None,
    isActive: bool = None,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    List all students (admin only).

    Query parameters:
    - **skip**: Pagination offset (default: 0)
    - **limit**: Max results (default: 100)
    - **search**: Search by email or college
    - **isPremium**: Filter by premium status
    - **isActive**: Filter by active status
    """
    query = db.query(Student)

    # Apply filters
    if search:
        query = query.filter(
            (Student.emailId.ilike(f"%{search}%")) | (Student.collegeOrSchool.ilike(f"%{search}%"))
        )

    if isPremium is not None:
        query = query.filter(Student.isPremium == isPremium)

    if isActive is not None:
        query = query.filter(Student.isActive == isActive)

    # Pagination
    students = query.offset(skip).limit(limit).all()

    return [StudentListItem.model_validate(s) for s in students]


@router.get("/admin/students/{student_id}", response_model=StudentResponse)
async def get_student_detail(
    student_id: int, admin: Admin = Depends(get_current_admin), db: Session = Depends(get_db)
):
    """
    Get student details by ID (admin only).
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    return StudentResponse.model_validate(student)


@router.patch("/admin/students/{student_id}", response_model=StudentResponse)
async def update_student(
    student_id: int,
    data: AdminUpdateStudent,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Update student (admin only).
    Can modify isPremium and isActive status.

    - **isPremium**: Grant/revoke premium access
    - **isActive**: Activate/deactivate account
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Update fields
    if data.isPremium is not None:
        student.isPremium = data.isPremium
        logger.info(
            f"Admin {admin.email} updated student {student.emailId} premium status: {data.isPremium}"
        )

    if data.isActive is not None:
        student.isActive = data.isActive
        logger.info(
            f"Admin {admin.email} updated student {student.emailId} active status: {data.isActive}"
        )

        # If deactivating, revoke all refresh tokens
        if not data.isActive:
            from app.models.user import RefreshToken

            db.query(RefreshToken).filter(RefreshToken.studentId == student.id).update(
                {"isRevoked": True}
            )

    db.commit()
    db.refresh(student)

    return StudentResponse.model_validate(student)
