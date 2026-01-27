"""
Dataset Model - Stores metadata for uploaded datasets with S3 support.
Each project can have multiple datasets (3-4 upload nodes).
"""

from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    BigInteger,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.session import Base


class Dataset(Base):
    """
    Dataset model for storing uploaded dataset metadata.
    Actual file content is stored in S3 or local storage.
    """

    __tablename__ = "datasets"

    # Primary identifiers
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(
        String(255), unique=True, nullable=False, index=True
    )  # e.g., "dataset_abc123"

    # Project/Pipeline association
    project_id = Column(
        Integer, ForeignKey("genai_pipelines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    node_id = Column(
        Integer, ForeignKey("genai_nodes.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # File metadata
    filename = Column(String(500), nullable=False)  # Original filename (e.g., "sales_data.csv")
    content_type = Column(String(100), default="text/csv")  # MIME type
    file_size = Column(BigInteger, nullable=False)  # Size in bytes
    file_extension = Column(String(20), nullable=False)  # ".csv", ".txt", ".json"

    # Storage location (S3 or local)
    storage_backend = Column(String(20), default="s3", nullable=False)  # "s3" or "local"
    s3_bucket = Column(String(255), nullable=True)  # S3 bucket name
    s3_key = Column(String(1000), nullable=True)  # S3 object key (path)
    s3_region = Column(String(50), nullable=True)  # S3 region
    local_path = Column(String(1000), nullable=True)  # Fallback for local storage

    # Dataset characteristics (CSV metadata)
    n_rows = Column(Integer, nullable=False)
    n_columns = Column(Integer, nullable=False)
    columns = Column(JSON, nullable=False)  # ["col1", "col2", ...]
    dtypes = Column(JSON, nullable=False)  # {"col1": "int64", "col2": "float64"}
    memory_usage_mb = Column(Float, nullable=False)

    # Preview data (cached for quick access without downloading from S3)
    preview_data = Column(JSON, nullable=True)  # First 10 rows as list of dicts

    # Ownership & Access
    user_id = Column(
        Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )
    is_public = Column(Boolean, default=False)

    # Lifecycle & Tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_accessed_at = Column(DateTime, nullable=True)  # Track usage for caching
    expires_at = Column(DateTime, nullable=True)  # Optional TTL for cleanup
    is_deleted = Column(Boolean, default=False, index=True)  # Soft delete

    # Validation status
    is_validated = Column(Boolean, default=False)  # CSV structure validated
    validation_errors = Column(JSON, nullable=True)  # Any validation issues

    # Relationships
    project = relationship("GenAIPipeline", backref="datasets")
    node = relationship("GenAINode", backref="dataset")
    user = relationship("Student", backref="datasets")

    # Composite indexes for performance
    __table_args__ = (
        Index("idx_project_user", "project_id", "user_id"),
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_storage_deleted", "storage_backend", "is_deleted"),
    )

    def __repr__(self):
        return (
            f"<Dataset(id={self.id}, dataset_id='{self.dataset_id}', filename='{self.filename}')>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "storage_backend": self.storage_backend,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "memory_usage_mb": self.memory_usage_mb,
            "preview_data": self.preview_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def s3_url(self) -> Optional[str]:
        """Generate S3 URL."""
        if self.storage_backend == "s3" and self.s3_bucket and self.s3_key:
            return f"s3://{self.s3_bucket}/{self.s3_key}"
        return None

    @property
    def file_path_str(self) -> str:
        """Get file path as string (S3 key or local path)."""
        if self.storage_backend == "s3":
            return str(self.s3_key) if self.s3_key else ""
        return str(self.local_path) if self.local_path else ""
