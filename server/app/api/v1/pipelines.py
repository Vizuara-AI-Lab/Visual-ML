"""
ML Pipeline API endpoints.
Production-ready with rate limiting, caching, and admin protection.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import JSONResponse
from app.schemas.pipeline import (
    NodeExecuteRequest,
    NodeExecuteResponse,
    PipelineExecuteRequest,
    PipelineExecuteResponse,
    TrainModelRequest,
    TrainModelResponse,
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    ModelListResponse,
    ModelListItem,
    ModelReloadRequest,
    ModelReloadResponse,
)
from app.services.pipeline_service import ml_service
from app.core.security import get_current_admin, get_current_user
from app.core.logging import logger, set_request_id
from app.core.exceptions import BaseMLException
from app.utils.ids import generate_request_id
from datetime import datetime

router = APIRouter(prefix="/ml", tags=["ML Pipeline"])


@router.post("/nodes/run", response_model=NodeExecuteResponse)
async def execute_node(
    request: NodeExecuteRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a single ML pipeline node.

    **Authentication Required**

    Available nodes:
    - upload_file: Upload and validate dataset
    - preprocess: Clean data and extract features
    - split: Split dataset into train/val/test
    - train: Train ML model
    - evaluate: Evaluate model performance

    Example:
    ```json
    {
        "node_type": "upload_file",
        "input_data": {
            "file_content": "...",
            "filename": "data.csv"
        }
    }
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    try:
        logger.info(f"Node execution request: {request.node_type}")

        result = await ml_service.execute_node(
            node_type=request.node_type, input_data=request.input_data, dry_run=request.dry_run
        )

        return NodeExecuteResponse(
            success=True, node_type=request.node_type, result=result, error=None
        )

    except BaseMLException as e:
        logger.error(f"Node execution failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=NodeExecuteResponse(
                success=False, node_type=request.node_type, error=e.to_dict(), result=None
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/pipeline/run", response_model=PipelineExecuteResponse)
async def execute_pipeline(
    request: PipelineExecuteRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a complete ML pipeline.

    **Authentication Required**

    Pipelines are sequences of nodes that process data step-by-step.

    Example:
    ```json
    {
        "pipeline": [
            {
                "node_type": "split",
                "input": {"dataset_path": "...", "target_column": "price"}
            },
            {
                "node_type": "train",
                "input": {"algorithm": "linear_regression", ...}
            }
        ]
    }
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    try:
        start_time = datetime.utcnow()

        logger.info(f"Pipeline execution request: {len(request.pipeline)} nodes")

        # Convert to dict format
        pipeline_config = [node.model_dump() for node in request.pipeline]

        results = await ml_service.execute_pipeline(
            pipeline=pipeline_config, dry_run=request.dry_run
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Check if any node failed
        success = all(r.get("success", True) for r in results)

        return PipelineExecuteResponse(
            success=success,
            pipeline_name=request.pipeline_name,
            results=results,
            total_execution_time_seconds=execution_time,
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/train/regression", response_model=TrainModelResponse)
async def train_regression_model(
    request: TrainModelRequest, admin_user: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Train a regression model (Linear Regression).

    **Admin Only**

    This endpoint trains a complete regression model including:
    - Data splitting
    - Model training
    - Evaluation on test set

    Example:
    ```json
    {
        "dataset_path": "./uploads/dataset_123.csv",
        "target_column": "price",
        "algorithm": "linear_regression",
        "task_type": "regression",
        "hyperparameters": {"fit_intercept": true}
    }
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    try:
        logger.info(f"Training regression model: {request.algorithm}")

        result = await ml_service.train_model(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            algorithm=request.algorithm,
            task_type="regression",
            hyperparameters=request.hyperparameters,
            test_ratio=request.train_test_split,
            val_ratio=request.validation_split,
        )

        return TrainModelResponse(**result)

    except BaseMLException as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/train/classification", response_model=TrainModelResponse)
async def train_classification_model(
    request: TrainModelRequest, admin_user: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Train a classification model (Logistic Regression).

    **Admin Only**

    This endpoint trains a complete classification model including:
    - Data splitting
    - Model training
    - Evaluation on test set with confusion matrix

    Example:
    ```json
    {
        "dataset_path": "./uploads/dataset_123.csv",
        "target_column": "category",
        "algorithm": "logistic_regression",
        "task_type": "classification",
        "hyperparameters": {"C": 1.0, "max_iter": 1000}
    }
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    try:
        logger.info(f"Training classification model: {request.algorithm}")

        result = await ml_service.train_model(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            algorithm=request.algorithm,
            task_type="classification",
            hyperparameters=request.hyperparameters,
            test_ratio=request.train_test_split,
            val_ratio=request.validation_split,
        )

        return TrainModelResponse(**result)

    except BaseMLException as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/predict/regression", response_model=PredictResponse)
async def predict_regression(
    request: PredictRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Make predictions with a regression model.

    **Authentication Required**

    Results are cached for performance.

    Example:
    ```json
    {
        "model_path": "./models/regression/model_123_v20260119.joblib",
        "features": {
            "feature1": 10.5,
            "feature2": 20.3,
            "feature3": 5.0
        }
    }
    ```
    """
    try:
        result = await ml_service.predict(
            model_path=request.model_path, features=request.features, task_type="regression"
        )

        return PredictResponse(**result)

    except BaseMLException as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.to_dict())


@router.post("/predict/classification", response_model=PredictResponse)
async def predict_classification(
    request: PredictRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Make predictions with a classification model.

    **Authentication Required**

    Returns predicted class and probabilities.
    Results are cached for performance.

    Example:
    ```json
    {
        "model_path": "./models/classification/model_123_v20260119.joblib",
        "features": {
            "feature1": 10.5,
            "feature2": 20.3
        }
    }
    ```
    """
    try:
        result = await ml_service.predict(
            model_path=request.model_path, features=request.features, task_type="classification"
        )

        return PredictResponse(**result)

    except BaseMLException as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.to_dict())


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    algorithm: Optional[str] = None, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List all available trained models.

    **Authentication Required**

    Optionally filter by algorithm.

    Query Parameters:
    - algorithm: Filter by algorithm (linear_regression, logistic_regression)
    """
    try:
        models = ml_service.list_models(algorithm=algorithm)

        model_items = [
            ModelListItem(
                model_id=m["model_id"],
                algorithm=m["algorithm"],
                model_path=m["model_path"],
                model_version=m.get("model_id", "unknown"),
                created_at=datetime.fromisoformat(m["created_at"]),
                size_mb=m["size_mb"],
                metadata=None,
            )
            for m in models
        ]

        return ModelListResponse(models=model_items, total_count=len(model_items))

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/reload", response_model=ModelReloadResponse)
async def reload_model(
    request: ModelReloadRequest, admin_user: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Reload a model (clear from cache).

    **Admin Only**

    Useful for hot-reloading updated models without restarting the server.
    """
    try:
        success = ml_service.reload_model(request.model_path)

        return ModelReloadResponse(
            success=success,
            model_path=request.model_path,
            message="Model reloaded successfully" if success else "Failed to reload model",
        )

    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
