"""
Import all models here for Alembic migrations.
"""

from app.db.session import Base
from app.models.user import Student, Admin, RefreshToken
from app.models.dataset import Dataset
from app.models.genai import (
    GenAIPipeline,
    GenAINode,
    GenAIEdge,
    GenAIPipelineRun,
    GenAINodeExecution,
    KnowledgeBase,
    KnowledgeBaseDocument,
    DocumentChunk,
    APISecret,
    ConversationMemory,
)
from app.models.custom_app import CustomApp
from app.models.gamification import StudentBadge

__all__ = ["Base", "Student", "Admin", "RefreshToken", "Dataset", "CustomApp", "StudentBadge"]
