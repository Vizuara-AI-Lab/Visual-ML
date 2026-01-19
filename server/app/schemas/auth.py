"""
Authentication Pydantic schemas for request/response validation.
Implements comprehensive auth schemas for Student and Admin.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum


class AuthProvider(str, Enum):
    """Authentication provider enum."""

    LOCAL = "LOCAL"
    GOOGLE = "GOOGLE"


class UserRole(str, Enum):
    """User role enum."""

    STUDENT = "STUDENT"
    ADMIN = "ADMIN"


# ========== Student Registration & Login ==========


class StudentRegister(BaseModel):
    """Student registration request."""

    emailId: EmailStr = Field(..., description="Student email (unique identifier)")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    collegeOrSchool: Optional[str] = Field(None, description="College or school name")
    contactNo: Optional[str] = Field(None, max_length=20, description="Contact number")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    @field_validator("contactNo")
    @classmethod
    def validate_contact(cls, v: Optional[str]) -> Optional[str]:
        """Validate contact number."""
        if v and not v.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            raise ValueError("Contact number must contain only digits, +, -, and spaces")
        return v


class StudentLogin(BaseModel):
    """Student login request."""

    emailId: EmailStr = Field(..., description="Student email")
    password: str = Field(..., description="Password")


class StudentGoogleAuth(BaseModel):
    """Student Google authentication request."""

    idToken: str = Field(..., description="Google ID token from frontend")
    profilePic: Optional[str] = Field(None, description="Google profile picture URL")


# ========== Admin Login ==========


class AdminLogin(BaseModel):
    """Admin login request."""

    email: EmailStr = Field(..., description="Admin email")
    password: str = Field(..., description="Password")


# ========== Token Schemas ==========


class TokenResponse(BaseModel):
    """Token response with access and refresh tokens."""

    accessToken: str = Field(..., description="JWT access token")
    refreshToken: str = Field(..., description="JWT refresh token")
    tokenType: str = Field(default="bearer", description="Token type")
    expiresIn: int = Field(..., description="Access token expiry in seconds")


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refreshToken: str = Field(..., description="Refresh token")


# ========== Student Profile Schemas ==========


class StudentResponse(BaseModel):
    """Student profile response."""

    id: int
    emailId: str
    role: UserRole
    authProvider: AuthProvider
    collegeOrSchool: Optional[str] = None
    contactNo: Optional[str] = None
    recentProject: Optional[str] = None
    profilePic: Optional[str] = None
    isPremium: bool
    isActive: bool
    createdAt: datetime
    lastLogin: Optional[datetime] = None

    class Config:
        from_attributes = True


class StudentUpdate(BaseModel):
    """Student profile update request (cannot change emailId)."""

    collegeOrSchool: Optional[str] = Field(None, max_length=255)
    contactNo: Optional[str] = Field(None, max_length=20)
    recentProject: Optional[str] = None
    profilePic: Optional[str] = Field(None, max_length=500)

    @field_validator("contactNo")
    @classmethod
    def validate_contact(cls, v: Optional[str]) -> Optional[str]:
        """Validate contact number."""
        if v and not v.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            raise ValueError("Contact number must contain only digits, +, -, and spaces")
        return v


class ChangePassword(BaseModel):
    """Change password request."""

    oldPassword: str = Field(..., description="Current password")
    newPassword: str = Field(..., min_length=8, description="New password")

    @field_validator("newPassword")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class ForgotPassword(BaseModel):
    """Forgot password request."""

    emailId: EmailStr = Field(..., description="Student email")


class ResetPassword(BaseModel):
    """Reset password request."""

    token: str = Field(..., description="Reset token from email")
    newPassword: str = Field(..., min_length=8, description="New password")

    @field_validator("newPassword")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ========== Admin Profile Schemas ==========


class AdminResponse(BaseModel):
    """Admin profile response."""

    id: int
    email: str
    role: UserRole
    name: Optional[str] = None
    isActive: bool
    createdAt: datetime
    lastLogin: Optional[datetime] = None

    class Config:
        from_attributes = True


# ========== Admin Student Management ==========


class StudentListItem(BaseModel):
    """Student list item for admin."""

    id: int
    emailId: str
    authProvider: AuthProvider
    collegeOrSchool: Optional[str] = None
    isPremium: bool
    isActive: bool
    createdAt: datetime
    lastLogin: Optional[datetime] = None

    class Config:
        from_attributes = True


class AdminUpdateStudent(BaseModel):
    """Admin update student request."""

    isPremium: Optional[bool] = None
    isActive: Optional[bool] = None


# ========== Auth Response ==========


class AuthResponse(BaseModel):
    """Complete authentication response."""

    user: StudentResponse | AdminResponse
    tokens: TokenResponse
    message: str = "Authentication successful"
