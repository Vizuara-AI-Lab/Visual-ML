"""
ML Pipeline API endpoints.
Production-ready with rate limiting, caching, and admin protection.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas.pipeline import (
    NodeExecuteRequest,
    NodeExecuteResponse,
    PipelineExecuteRequest,
    PipelineExecuteResponse,
    TrainModelRequest,
    TrainModelResponse,
    PredictRequest,
    PredictResponse,
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


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable format.
    Handles datetime objects, NaN/Infinity, Pydantic models, and nested structures.
    """
    import math
    
    # Handle None early
    if obj is None:
        return None
    
    # Handle booleans before checking numeric types
    if isinstance(obj, bool):
        return obj
    
    # Handle integers
    if isinstance(obj, int):
        return obj
    
    # Handle NaN and Infinity for regular floats
    if isinstance(obj, float):
        if math.isnan(obj):
            return None  # Convert NaN to null
        elif math.isinf(obj):
            return None  # Convert Infinity to null as well for safety
        return obj
    
    # Handle numpy types by checking type name string
    type_name = type(obj).__name__
    if type_name.startswith(('int', 'uint', 'long')) and type_name not in ('integer',):
        try:
            return int(obj)
        except (ValueError, TypeError):
            pass
    elif 'float' in type_name or type_name in ('number',):
        try:
            val = float(obj)
            if math.isnan(val):
                return None
            elif math.isinf(val):
                return None
            return val
        except (ValueError, TypeError):
            pass
    elif type_name == 'ndarray':
        try:
            return make_json_serializable(obj.tolist())
        except (AttributeError, TypeError):
            pass
    
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle Pydantic models
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump", None)):
        try:
            return make_json_serializable(obj.model_dump())
        except Exception:
            pass
    
    # Handle other objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return make_json_serializable(obj.__dict__)
        except Exception:
            pass
    
    return obj
            pass
    
    return obj
    return obj


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
        edges_config = [edge.model_dump() for edge in request.edges]

        # Execute pipeline - returns full DAG result dict
        dag_result = await ml_service.execute_pipeline(
            pipeline=pipeline_config,
            edges=edges_config,
            dry_run=request.dry_run,
            current_user=current_user,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        # Extract success and results from DAG execution result
        success = dag_result.get("success", False)
        results = dag_result.get("results", [])
        error_message = dag_result.get("error")

        # If DAG failed, raise HTTP error with the error message
        if not success and error_message:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)

        return PipelineExecuteResponse(
            success=success,
            pipeline_name=request.pipeline_name,
            results=results,
            total_execution_time_seconds=execution_time,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including our 400 errors)
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/pipeline/run/stream")
async def execute_pipeline_stream(
    request: PipelineExecuteRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a complete ML pipeline with real-time SSE progress updates.

    **Authentication Required**

    Streams Server-Sent Events (SSE) for each node execution:
    - node_started: When a node begins execution
    - node_completed: When a node finishes successfully
    - node_failed: When a node execution fails
    - pipeline_completed: When entire pipeline finishes
    - pipeline_failed: When pipeline execution fails

    Event format:
    ```
    data: {"event": "node_started", "node_id": "node_123", "node_type": "linear_regression", "label": "Linear Regression"}
    data: {"event": "node_completed", "node_id": "node_123", "node_type": "linear_regression", "label": "Linear Regression", "success": true}
    data: {"event": "pipeline_completed", "success": true, "nodes_executed": 5}
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    # Create async queue for event streaming
    event_queue = asyncio.Queue()

    async def generate():
        """Generator function for SSE events."""
        try:
            # Start pipeline execution in background task
            execution_task = asyncio.create_task(execute_pipeline_with_progress())

            # Stream events as they arrive
            while True:
                try:
                    # Wait for event with timeout
                    event_data = await asyncio.wait_for(event_queue.get(), timeout=0.1)

                    if event_data is None:
                        # Sentinel value indicating completion
                        break

                    # Serialize to JSON with strict NaN handling
                    try:
                        json_str = json.dumps(event_data, allow_nan=False)
                        yield f"data: {json_str}\n\n"
                    except (ValueError, TypeError) as json_error:
                        # If serialization fails, clean the data and retry
                        logger.warning(f"JSON serialization error, cleaning data: {json_error}")
                        cleaned_data = make_json_serializable(event_data)
                        json_str = json.dumps(cleaned_data, allow_nan=False)
                        yield f"data: {json_str}\n\n"

                except asyncio.TimeoutError:
                    # No event available, check if task is done
                    if execution_task.done():
                        # Check for any remaining events
                        if event_queue.empty():
                            break
                    continue

            # Wait for execution to complete
            await execution_task

        except Exception as e:
            logger.error(f"SSE streaming error: {str(e)}", exc_info=True)
            error_data = make_json_serializable({'event': 'pipeline_failed', 'success': False, 'error': str(e)})
            yield f"data: {json.dumps(error_data, allow_nan=False)}\n\n"

    async def execute_pipeline_with_progress():
        """Execute pipeline and send progress events to queue."""
        try:
            start_time = datetime.utcnow()

            logger.info(f"Pipeline streaming execution request: {len(request.pipeline)} nodes")

            # Convert to dict format
            pipeline_config = [node.model_dump() for node in request.pipeline]
            edges_config = [edge.model_dump() for edge in request.edges]

            # Progress callback for SSE events
            async def progress_callback(event_data: Dict[str, Any]):
                """Send progress events to queue."""
                # Make event data JSON-serializable before queuing
                serializable_data = make_json_serializable(event_data)
                await event_queue.put(serializable_data)

            # Execute pipeline with streaming progress
            dag_result = await ml_service.execute_pipeline(
                pipeline=pipeline_config,
                edges=edges_config,
                dry_run=request.dry_run,
                current_user=current_user,
                progress_callback=progress_callback,
            )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Extract success and results from DAG execution result
            success = dag_result.get("success", False)
            results = dag_result.get("results", [])
            error_message = dag_result.get("error")

            # Send final event
            if success:
                await event_queue.put(
                    make_json_serializable(
                        {
                            "event": "pipeline_completed",
                            "success": True,
                            "nodes_executed": len(results),
                            "execution_time_seconds": execution_time,
                            "results": results,
                        }
                    )
                )
            else:
                await event_queue.put(
                    make_json_serializable(
                        {"event": "pipeline_failed", "success": False, "error": error_message}
                    )
                )

        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}", exc_info=True)
            await event_queue.put(
                make_json_serializable(
                    {"event": "pipeline_failed", "success": False, "error": str(e)}
                )
            )
        finally:
            # Send sentinel to stop generator
            await event_queue.put(None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/train/regression", response_model=TrainModelResponse)
async def train_regression_model(
    request: TrainModelRequest, admin_user: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Train a regression model (Linear Regression).


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

        # Offload to Celery
        from app.tasks.ml_tasks import train_model_task

        task = train_model_task.delay(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            algorithm=request.algorithm,
            task_type="regression",
            hyperparameters=request.hyperparameters,
            test_ratio=request.train_test_split,
            val_ratio=request.validation_split,
        )

        return TrainModelResponse(
            success=True,
            model_id=None,  # Will be available when task completes
            model_path=None,
            model_version=None,
            training_metrics={},
            test_metrics={},
            metadata={"task_id": task.id, "status": "processing"},
        )

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

        # Offload to Celery
        from app.tasks.ml_tasks import train_model_task

        task = train_model_task.delay(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            algorithm=request.algorithm,
            task_type="classification",
            hyperparameters=request.hyperparameters,
            test_ratio=request.train_test_split,
            val_ratio=request.validation_split,
        )

        return TrainModelResponse(
            success=True,
            model_id=None,
            model_path=None,
            model_version=None,
            training_metrics={},
            test_metrics={},
            metadata={"task_id": task.id, "status": "processing"},
        )

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


# ========== ASYNC EXECUTION ENDPOINTS (with Celery) ==========


@router.post("/pipeline/run-async")
async def execute_pipeline_async(
    request: PipelineExecuteRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a complete ML pipeline asynchronously using Celery.

    **Authentication Required**

    This endpoint immediately returns a task_id that can be used to poll
    for progress and results. Ideal for long-running pipelines.

    **Advantages over synchronous endpoint:**
    - Non-blocking: Returns immediately
    - Progress tracking: Real-time progress updates
    - No timeout: Can run for hours if needed
    - Resumable: Can check status later

    **Workflow:**
    1. POST /ml/pipeline/run-async → Get task_id
    2. GET /api/v1/tasks/{task_id}/status → Check progress
    3. GET /api/v1/tasks/{task_id}/result → Get final results

    Example request:
    ```json
    {
        "pipeline": [
            {"node_type": "upload_file", "input": {...}},
            {"node_type": "encoding", "input": {...}},
            {"node_type": "split", "input": {...}},
            {"node_type": "linear_regression", "input": {...}}
        ],
        "pipeline_name": "My ML Pipeline"
    }
    ```

    Example response:
    ```json
    {
        "success": true,
        "task_id": "abc-123-def-456",
        "message": "Pipeline execution started",
        "status_url": "/api/v1/tasks/abc-123-def-456/status",
        "result_url": "/api/v1/tasks/abc-123-def-456/result"
    }
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    try:
        logger.info(f"🚀 Async pipeline execution request: {len(request.pipeline)} nodes")

        # Convert to dict format
        pipeline_config = [node.model_dump() for node in request.pipeline]

        # Import Celery task
        from app.tasks.pipeline_tasks import execute_pipeline_task

        # Start async task
        task = execute_pipeline_task.delay(
            pipeline=pipeline_config,
            pipeline_name=request.pipeline_name or "Pipeline",
            dry_run=request.dry_run,
            current_user=current_user,
        )

        logger.info(f"✅ Pipeline task created: {task.id}")

        return {
            "success": True,
            "task_id": task.id,
            "message": "Pipeline execution started",
            "status_url": f"/api/v1/tasks/{task.id}/status",
            "result_url": f"/api/v1/tasks/{task.id}/result",
            "cancel_url": f"/api/v1/tasks/{task.id}",
        }

    except Exception as e:
        logger.error(f"Failed to start async pipeline execution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start pipeline execution: {str(e)}",
        )


@router.post("/nodes/run-async")
async def execute_node_async(
    request: NodeExecuteRequest, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a single ML node asynchronously using Celery.

    **Authentication Required**

    Similar to pipeline async execution, but for individual nodes.
    Useful for expensive operations like:
    - Large dataset uploads
    - Complex feature engineering
    - Model training

    Example request:
    ```json
    {
        "node_type": "linear_regression",
        "input_data": {
            "train_dataset_id": "dataset_123",
            "target_column": "price",
            "fit_intercept": true
        }
    }
    ```

    Example response:
    ```json
    {
        "success": true,
        "task_id": "xyz-789",
        "message": "Node execution started",
        "node_type": "linear_regression",
        "status_url": "/api/v1/tasks/xyz-789/status"
    }
    ```
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    try:
        logger.info(f"🔧 Async node execution request: {request.node_type}")

        # Import Celery task
        from app.tasks.pipeline_tasks import execute_node_task

        # Start async task
        task = execute_node_task.delay(
            node_type=request.node_type,
            input_data=request.input_data,
            dry_run=request.dry_run,
            current_user=current_user,
        )

        logger.info(f"✅ Node task created: {task.id}")

        return {
            "success": True,
            "task_id": task.id,
            "message": f"{request.node_type} execution started",
            "node_type": request.node_type,
            "status_url": f"/api/v1/tasks/{task.id}/status",
            "result_url": f"/api/v1/tasks/{task.id}/result",
            "cancel_url": f"/api/v1/tasks/{task.id}",
        }

    except Exception as e:
        logger.error(f"Failed to start async node execution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start node execution: {str(e)}",
        )
