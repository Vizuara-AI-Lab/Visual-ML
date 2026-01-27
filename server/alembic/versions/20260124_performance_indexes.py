"""Add performance indexes

Revision ID: 20260124_performance_indexes
Revises: 20260123_1500_add_datasets_table
Create Date: 2026-01-24 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260124_performance_indexes"
down_revision: Union[str, None] = "20260123_1500_add_datasets_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add composite and single-column indexes for performance optimization."""

    # GenAI Pipelines - for list_projects() query
    op.create_index(
        "idx_pipelines_student_updated",
        "genai_pipelines",
        ["studentId", sa.text('"updatedAt" DESC')],
        unique=False,
    )

    # Datasets - for list_project_datasets() query
    op.create_index(
        "idx_datasets_project_deleted", "datasets", ["project_id", "is_deleted"], unique=False
    )

    # Datasets - for user's datasets query
    op.create_index(
        "idx_datasets_user_created",
        "datasets",
        ["user_id", sa.text("created_at DESC")],
        unique=False,
    )

    # GenAI Nodes - for get_pipeline_nodes() query
    op.create_index(
        "idx_nodes_pipeline_type", "genai_nodes", ["pipelineId", "nodeType"], unique=False
    )

    # GenAI Edges - for get_pipeline_edges() query
    op.create_index("idx_edges_pipeline", "genai_edges", ["pipelineId"], unique=False)

    # Students - for login and search queries
    op.create_index("idx_students_email_active", "students", ["emailId", "isActive"], unique=False)

    # Students - for admin search
    op.create_index("idx_students_college", "students", ["collegeOrSchool"], unique=False)

    # Pipeline Runs - for list_pipeline_runs() query
    op.create_index(
        "idx_runs_pipeline_created",
        "genai_pipeline_runs",
        ["pipelineId", sa.text('"createdAt" DESC')],
        unique=False,
    )


def downgrade() -> None:
    """Remove performance indexes."""

    op.drop_index("idx_runs_pipeline_created", table_name="genai_pipeline_runs")
    op.drop_index("idx_students_college", table_name="students")
    op.drop_index("idx_students_email_active", table_name="students")
    op.drop_index("idx_edges_pipeline", table_name="genai_edges")
    op.drop_index("idx_nodes_pipeline_type", table_name="genai_nodes")
    op.drop_index("idx_datasets_user_created", table_name="datasets")
    op.drop_index("idx_datasets_project_deleted", table_name="datasets")
    op.drop_index("idx_pipelines_student_updated", table_name="genai_pipelines")
