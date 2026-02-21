"""add_xp_and_level_to_students

Revision ID: 2d012f632569
Revises: 
Create Date: 2026-02-19 14:22:19.522126

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2d012f632569'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('students', sa.Column('xp', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('students', sa.Column('level', sa.Integer(), nullable=False, server_default='1'))


def downgrade() -> None:
    op.drop_column('students', 'level')
    op.drop_column('students', 'xp')
