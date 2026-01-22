"""
User models: Student, Admin, and RefreshToken.
Implements RBAC with comprehensive student and admin schemas.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.orm import relationship
import enum
from app.db.session import Base


class UserRole(str, enum.Enum):
    """User roles for RBAC."""

    STUDENT = "STUDENT"
    ADMIN = "ADMIN"


class AuthProvider(str, enum.Enum):
    """Authentication providers."""

    LOCAL = "LOCAL"
    GOOGLE = "GOOGLE"


class Student(Base):
    """
    Student model with comprehensive fields.

    Supports both local (email/password) and Google OAuth authentication.
    """

    __tablename__ = "students"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Authentication Fields
    emailId = Column(String(255), unique=True, index=True, nullable=False)
    fullName = Column(String(255), nullable=False)
    password = Column(String(255), nullable=True)  # Nullable for Google OAuth users
    role = Column(SQLEnum(UserRole), default=UserRole.STUDENT, nullable=False)
    authProvider = Column(SQLEnum(AuthProvider), default=AuthProvider.LOCAL, nullable=False)
    googleId = Column(String(255), unique=True, nullable=True, index=True)

    # Student Profile Fields
    collegeOrSchool = Column(String(255), nullable=True)
    contactNo = Column(String(20), nullable=True)
    recentProject = Column(Text, nullable=True)
    profilePic = Column(String(500), nullable=True)

    # Premium and Status
    isPremium = Column(Boolean, default=False, nullable=False)
    isActive = Column(Boolean, default=True, nullable=False)

    # Email Verification
    isEmailVerified = Column(Boolean, default=False, nullable=False)
    verificationOTP = Column(String(6), nullable=True)
    otpExpiresAt = Column(DateTime, nullable=True)
    verificationToken = Column(String(255), nullable=True)

    # Password Reset
    resetToken = Column(String(255), nullable=True)
    resetTokenExpiry = Column(DateTime, nullable=True)

    # Timestamps
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    lastLogin = Column(DateTime, nullable=True)

    # Relationships
    refresh_tokens = relationship(
        "RefreshToken", back_populates="student", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Student(id={self.id}, email={self.emailId}, provider={self.authProvider})>"


class Admin(Base):
    """
    Admin model for administrative users.

    Admins have elevated privileges to manage students and system.
    """

    __tablename__ = "admins"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Authentication Fields
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.ADMIN, nullable=False)

    # Admin Profile
    name = Column(String(255), nullable=True)
    isActive = Column(Boolean, default=True, nullable=False)

    # Timestamps
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    lastLogin = Column(DateTime, nullable=True)

    # Relationships
    refresh_tokens = relationship(
        "RefreshToken", back_populates="admin", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Admin(id={self.id}, email={self.email})>"


class RefreshToken(Base):
    """
    Refresh token model for secure token rotation.

    Supports token revocation and prevents token reuse attacks.
    """

    __tablename__ = "refresh_tokens"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Token Fields
    token = Column(String(500), unique=True, index=True, nullable=False)
    expiresAt = Column(DateTime, nullable=False)
    isRevoked = Column(Boolean, default=False, nullable=False)

    # User Reference (polymorphic - can be student or admin)
    studentId = Column(Integer, ForeignKey("students.id"), nullable=True)
    adminId = Column(Integer, ForeignKey("admins.id"), nullable=True)

    # Metadata
    deviceInfo = Column(String(255), nullable=True)
    ipAddress = Column(String(45), nullable=True)

    # Timestamps
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    usedAt = Column(DateTime, nullable=True)

    # Relationships
    student = relationship("Student", back_populates="refresh_tokens")
    admin = relationship("Admin", back_populates="refresh_tokens")

    def __repr__(self):
        user_type = "student" if self.studentId else "admin"
        user_id = self.studentId or self.adminId
        return f"<RefreshToken({user_type}={user_id}, revoked={self.isRevoked})>"
