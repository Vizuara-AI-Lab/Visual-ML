"""
Custom App Database Model.
Stores student-built UI pages that wrap ML pipelines.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from datetime import datetime
from app.db.session import Base


class CustomApp(Base):
    __tablename__ = "custom_apps"

    id = Column(Integer, primary_key=True, index=True)

    # Owner and pipeline link
    studentId = Column(
        Integer,
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pipelineId = Column(
        Integer,
        ForeignKey("genai_pipelines.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # App identity
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    # UI definition
    blocks = Column(JSON, nullable=False, default=list)
    theme = Column(JSON, nullable=True)

    # Publishing state
    is_published = Column(Boolean, nullable=False, default=False)
    published_at = Column(DateTime, nullable=True)

    # Analytics
    view_count = Column(Integer, nullable=False, default=0)
    execution_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
