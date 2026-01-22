"""
Project schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ProjectCreate(BaseModel):
    """Create new project request."""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")


class ProjectUpdate(BaseModel):
    """Update project request."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class ProjectState(BaseModel):
    """Playground state to save/load."""
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="React Flow nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="React Flow edges")
    datasetMetadata: Optional[Dict[str, Any]] = Field(None, description="Dataset metadata")
    executionResult: Optional[Dict[str, Any]] = Field(None, description="Last execution result")


class ProjectResponse(BaseModel):
    """Project response."""
    id: int
    name: str
    description: Optional[str]
    studentId: int
    createdAt: datetime
    updatedAt: datetime
    lastRunAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ProjectListItem(BaseModel):
    """Project list item for dashboard."""
    id: int
    name: str
    description: Optional[str]
    createdAt: datetime
    updatedAt: datetime
    
    class Config:
        from_attributes = True


class ProjectStateResponse(BaseModel):
    """Project state response."""
    projectId: int
    state: ProjectState
    updatedAt: datetime
