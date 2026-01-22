"""add_fullname_to_students

Revision ID: 284a03fdedc4
Revises: b5d726af7c5c
Create Date: 2026-01-22 09:45:50.852490

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '284a03fdedc4'
down_revision: Union[str, None] = 'b5d726af7c5c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add fullName column to students table
    # First add as nullable
    op.add_column('students', sa.Column('fullName', sa.String(length=255), nullable=True))
    
    # Update existing rows with a default value (email prefix)
    op.execute("UPDATE students SET \"fullName\" = split_part(\"emailId\", '@', 1) WHERE \"fullName\" IS NULL")
    
    # Now make it non-nullable
    op.alter_column('students', 'fullName', nullable=False)


def downgrade() -> None:
    # Remove fullName column from students table
    op.drop_column('students', 'fullName')
