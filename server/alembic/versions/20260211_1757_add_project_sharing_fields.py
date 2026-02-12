"""add_project_sharing_fields

Revision ID: 20260211_1757
Revises: 20260124_performance_indexes
Create Date: 2026-02-11 17:57:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260211_1757"
down_revision: Union[str, None] = "20260124_performance_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add sharing fields to genai_pipelines table"""
    # Add is_public column
    op.add_column(
        "genai_pipelines",
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default="false"),
    )

    # Add share_token column (unique, indexed)
    op.add_column("genai_pipelines", sa.Column("share_token", sa.String(255), nullable=True))
    op.create_index("idx_share_token", "genai_pipelines", ["share_token"], unique=True)

    # Add allow_cloning column
    op.add_column(
        "genai_pipelines",
        sa.Column("allow_cloning", sa.Boolean(), nullable=False, server_default="true"),
    )

    # Add view_count column
    op.add_column(
        "genai_pipelines", sa.Column("view_count", sa.Integer(), nullable=False, server_default="0")
    )

    # Add clone_count column
    op.add_column(
        "genai_pipelines",
        sa.Column("clone_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    """Remove sharing fields from genai_pipelines table"""
    op.drop_index("idx_share_token", table_name="genai_pipelines")
    op.drop_column("genai_pipelines", "clone_count")
    op.drop_column("genai_pipelines", "view_count")
    op.drop_column("genai_pipelines", "allow_cloning")
    op.drop_column("genai_pipelines", "share_token")
    op.drop_column("genai_pipelines", "is_public")
