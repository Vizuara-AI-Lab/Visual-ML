"""
Custom App schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class CustomAppCreate(BaseModel):
    """Create a new custom app linked to a pipeline."""
    pipeline_id: int = Field(..., description="Pipeline ID to wrap")
    name: str = Field(..., min_length=1, max_length=255, description="App name")


class CustomAppUpdate(BaseModel):
    """Update custom app fields."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    slug: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    blocks: Optional[List[Dict[str, Any]]] = None
    theme: Optional[Dict[str, Any]] = None


class CustomAppPublish(BaseModel):
    """Publish or unpublish an app."""
    is_published: bool
    slug: Optional[str] = Field(None, min_length=1, max_length=255)


class CustomAppResponse(BaseModel):
    """Full custom app response (owner view)."""
    id: int
    studentId: int
    pipelineId: int
    name: str
    slug: str
    description: Optional[str]
    blocks: List[Dict[str, Any]]
    theme: Optional[Dict[str, Any]]
    is_published: bool
    published_at: Optional[datetime]
    view_count: int
    execution_count: int
    createdAt: datetime
    updatedAt: Optional[datetime]

    class Config:
        from_attributes = True


class PublicAppResponse(BaseModel):
    """Public-facing app response (no internal details)."""
    name: str
    description: Optional[str]
    blocks: List[Dict[str, Any]]
    theme: Optional[Dict[str, Any]]
    owner_name: str


class PublicAppExecuteRequest(BaseModel):
    """Request to execute a pipeline through a public app."""
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Form field values")
    file_data: Optional[str] = Field(None, description="Base64-encoded CSV file content")
    node_inputs: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Node-mapped inputs: { nodeId: { configKey: value } }"
    )
    file_node_id: Optional[str] = Field(
        None, description="nodeId of the upload_file node to receive the file"
    )


class PublicAppExecuteResponse(BaseModel):
    """Pipeline execution result from a public app."""
    success: bool
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class SuggestedBlocksResponse(BaseModel):
    """Response from suggest-blocks endpoint."""
    blocks: List[Dict[str, Any]]
    pipeline_name: str
    node_count: int
