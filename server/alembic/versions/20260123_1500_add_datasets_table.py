"""
Add datasets table for S3 dataset management

Revision ID: 20260123_1500
Revises: 6c828d0aa2fd
Create Date: 2026-01-23 15:00:00.000000

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260123_1500_add_datasets_table"
down_revision: Union[str, None] = "6c828d0aa2fd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create datasets table for S3-backed dataset storage."""

    # Create datasets table
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.Integer(), nullable=True),
        # File metadata
        sa.Column("filename", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("file_extension", sa.String(length=20), nullable=False),
        # Storage location
        sa.Column("storage_backend", sa.String(length=20), nullable=False),
        sa.Column("s3_bucket", sa.String(length=255), nullable=True),
        sa.Column("s3_key", sa.String(length=1000), nullable=True),
        sa.Column("s3_region", sa.String(length=50), nullable=True),
        sa.Column("local_path", sa.String(length=1000), nullable=True),
        # Dataset characteristics
        sa.Column("n_rows", sa.Integer(), nullable=False),
        sa.Column("n_columns", sa.Integer(), nullable=False),
        sa.Column("columns", sa.JSON(), nullable=False),
        sa.Column("dtypes", sa.JSON(), nullable=False),
        sa.Column("memory_usage_mb", sa.Float(), nullable=False),
        # Preview data
        sa.Column("preview_data", sa.JSON(), nullable=True),
        # Ownership & Access
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("is_public", sa.Boolean(), nullable=True, default=False),
        # Lifecycle
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=True, default=False),
        # Validation
        sa.Column("is_validated", sa.Boolean(), nullable=True, default=False),
        sa.Column("validation_errors", sa.JSON(), nullable=True),
        # Primary key
        sa.PrimaryKeyConstraint("id"),
        # Foreign keys
        sa.ForeignKeyConstraint(["project_id"], ["genai_pipelines.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["genai_nodes.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["students.id"], ondelete="CASCADE"),
    )

    # Create indexes
    op.create_index("ix_datasets_id", "datasets", ["id"])
    op.create_index("ix_datasets_dataset_id", "datasets", ["dataset_id"], unique=True)
    op.create_index("ix_datasets_project_id", "datasets", ["project_id"])
    op.create_index("ix_datasets_node_id", "datasets", ["node_id"])
    op.create_index("ix_datasets_user_id", "datasets", ["user_id"])
    op.create_index("ix_datasets_created_at", "datasets", ["created_at"])
    op.create_index("ix_datasets_is_deleted", "datasets", ["is_deleted"])

    # Create composite indexes for common queries
    op.create_index("idx_project_user", "datasets", ["project_id", "user_id"])
    op.create_index("idx_user_created", "datasets", ["user_id", "created_at"])
    op.create_index("idx_storage_deleted", "datasets", ["storage_backend", "is_deleted"])


def downgrade() -> None:
    """Drop datasets table."""

    # Drop indexes
    op.drop_index("idx_storage_deleted", table_name="datasets")
    op.drop_index("idx_user_created", table_name="datasets")
    op.drop_index("idx_project_user", table_name="datasets")
    op.drop_index("ix_datasets_is_deleted", table_name="datasets")
    op.drop_index("ix_datasets_created_at", table_name="datasets")
    op.drop_index("ix_datasets_user_id", table_name="datasets")
    op.drop_index("ix_datasets_node_id", table_name="datasets")
    op.drop_index("ix_datasets_project_id", table_name="datasets")
    op.drop_index("ix_datasets_dataset_id", table_name="datasets")
    op.drop_index("ix_datasets_id", table_name="datasets")

    # Drop table
    op.drop_table("datasets")
