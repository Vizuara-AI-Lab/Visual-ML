"""Initial auth schema with Student, Admin, RefreshToken

Revision ID: 001
Revises:
Create Date: 2024-01-19 00:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create students table
    op.create_table(
        "students",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("emailId", sa.String(length=255), nullable=False),
        sa.Column("password", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("authProvider", sa.String(length=50), nullable=False),
        sa.Column("googleId", sa.String(length=255), nullable=True),
        sa.Column("collegeOrSchool", sa.String(length=255), nullable=True),
        sa.Column("contactNo", sa.String(length=20), nullable=True),
        sa.Column("recentProject", sa.Text(), nullable=True),
        sa.Column("profilePic", sa.String(length=500), nullable=True),
        sa.Column("isPremium", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("isActive", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("resetToken", sa.String(length=255), nullable=True),
        sa.Column("resetTokenExpiry", sa.DateTime(), nullable=True),
        sa.Column(
            "createdAt", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updatedAt", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("lastLogin", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_students_emailId", "students", ["emailId"], unique=True)
    op.create_index("idx_students_googleId", "students", ["googleId"], unique=True)
    op.create_index("idx_students_resetToken", "students", ["resetToken"], unique=False)

    # Create admins table
    op.create_table(
        "admins",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("isActive", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "createdAt", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updatedAt", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column("lastLogin", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_admins_email", "admins", ["email"], unique=True)

    # Create refresh_tokens table
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("token", sa.String(length=500), nullable=False),
        sa.Column("studentId", sa.Integer(), nullable=True),
        sa.Column("adminId", sa.Integer(), nullable=True),
        sa.Column("expiresAt", sa.DateTime(), nullable=False),
        sa.Column("isRevoked", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("deviceInfo", sa.String(length=500), nullable=True),
        sa.Column("ipAddress", sa.String(length=50), nullable=True),
        sa.Column(
            "createdAt", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.Column(
            "updatedAt", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")
        ),
        sa.ForeignKeyConstraint(["studentId"], ["students.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["adminId"], ["admins.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_refresh_tokens_token", "refresh_tokens", ["token"], unique=True)
    op.create_index("idx_refresh_tokens_studentId", "refresh_tokens", ["studentId"], unique=False)
    op.create_index("idx_refresh_tokens_adminId", "refresh_tokens", ["adminId"], unique=False)


def downgrade() -> None:
    op.drop_table("refresh_tokens")
    op.drop_table("admins")
    op.drop_table("students")
