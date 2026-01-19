"""
Pydantic schemas for ML pipeline API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# Node Execution Schemas


class NodeExecuteRequest(BaseModel):
    """Request to execute a single node."""

    node_type: str = Field(..., description="Type of node to execute")
    input_data: Dict[str, Any] = Field(..., description="Node input data")
    node_id: Optional[str] = Field(None, description="Optional custom node ID")
    dry_run: bool = Field(False, description="Validate without executing")


class NodeExecuteResponse(BaseModel):
    """Response from node execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    node_type: str = Field(..., description="Type of node executed")
    result: Optional[Dict[str, Any]] = Field(None, description="Node output")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")


# Pipeline Execution Schemas


class PipelineNodeConfig(BaseModel):
    """Configuration for a single node in a pipeline."""

    node_type: str = Field(..., description="Type of node")
    input: Dict[str, Any] = Field(..., description="Node input data")
    node_id: Optional[str] = Field(None, description="Optional custom node ID")


class PipelineExecuteRequest(BaseModel):
    """Request to execute a complete pipeline."""

    pipeline: List[PipelineNodeConfig] = Field(..., description="List of nodes to execute")
    dry_run: bool = Field(False, description="Validate without executing")
    pipeline_name: Optional[str] = Field(None, description="Optional pipeline name")


class PipelineExecuteResponse(BaseModel):
    """Response from pipeline execution."""

    success: bool = Field(..., description="Whether pipeline succeeded")
    pipeline_name: Optional[str] = Field(None, description="Pipeline name")
    results: List[Dict[str, Any]] = Field(..., description="Results from each node")
    total_execution_time_seconds: float = Field(..., description="Total execution time")


# Training Schemas


class TrainModelRequest(BaseModel):
    """Request to train a model."""

    dataset_path: str = Field(..., description="Path to training dataset")
    target_column: str = Field(..., description="Name of target column")
    algorithm: str = Field(
        ..., description="Algorithm: 'linear_regression' or 'logistic_regression'"
    )
    task_type: str = Field(..., description="Task type: 'regression' or 'classification'")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    train_test_split: float = Field(0.2, description="Test set ratio")
    validation_split: Optional[float] = Field(None, description="Validation set ratio")
    model_name: Optional[str] = Field(None, description="Optional model name")


class TrainModelResponse(BaseModel):
    """Response from model training."""

    success: bool = Field(..., description="Whether training succeeded")
    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to saved model")
    model_version: str = Field(..., description="Model version")
    training_metrics: Dict[str, float] = Field(..., description="Training metrics")
    test_metrics: Optional[Dict[str, Any]] = Field(None, description="Test metrics")
    metadata: Dict[str, Any] = Field(..., description="Training metadata")


# Prediction Schemas


class PredictRequest(BaseModel):
    """Request for prediction."""

    model_path: str = Field(..., description="Path to trained model")
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")


class PredictBatchRequest(BaseModel):
    """Request for batch prediction."""

    model_path: str = Field(..., description="Path to trained model")
    features: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")


class PredictResponse(BaseModel):
    """Response from prediction."""

    prediction: Any = Field(..., description="Model prediction")
    probability: Optional[List[float]] = Field(
        None, description="Class probabilities (classification)"
    )
    model_version: str = Field(..., description="Model version used")


class PredictBatchResponse(BaseModel):
    """Response from batch prediction."""

    predictions: List[Any] = Field(..., description="List of predictions")
    probabilities: Optional[List[List[float]]] = Field(
        None, description="List of class probabilities"
    )
    model_version: str = Field(..., description="Model version used")
    count: int = Field(..., description="Number of predictions")


# Model Management Schemas


class ModelListItem(BaseModel):
    """Model item in list."""

    model_id: str
    algorithm: str
    model_path: str
    model_version: str
    created_at: datetime
    size_mb: float
    metadata: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    """Response with list of models."""

    models: List[ModelListItem]
    total_count: int


class ModelReloadRequest(BaseModel):
    """Request to reload a model."""

    model_path: str = Field(..., description="Path to model to reload")


class ModelReloadResponse(BaseModel):
    """Response from model reload."""

    success: bool
    model_path: str
    message: str
