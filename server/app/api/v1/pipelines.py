"""
ML Pipeline API endpoints.
Production-ready with rate limiting, caching, and admin protection.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from pydantic import BaseModel
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
    if type_name.startswith(("int", "uint", "long")) and type_name not in ("integer",):
        try:
            return int(obj)
        except (ValueError, TypeError):
            pass
    elif "float" in type_name or type_name in ("number",):
        try:
            val = float(obj)
            if math.isnan(val):
                return None
            elif math.isinf(val):
                return None
            return val
        except (ValueError, TypeError):
            pass
    elif type_name == "ndarray":
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
            from app.ml.error_formatter import format_error
            error_data = make_json_serializable(
                {"event": "pipeline_failed", "success": False, "error": format_error(e)}
            )
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
            from app.ml.error_formatter import format_error
            await event_queue.put(
                make_json_serializable(
                    {"event": "pipeline_failed", "success": False, "error": format_error(e)}
                )
            )
        finally:
            # Send sentinel to stop generator
            await event_queue.put(None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevents Nginx from buffering SSE
        },
    )


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


# ========== INTERACTIVE PREDICTION ENDPOINT ==========


class InteractivePredictRequest(BaseModel):
    """Request for interactive prediction with rich model-specific output."""
    model_path: str
    model_type: str  # "random_forest" or "decision_tree"
    task_type: str   # "classification" or "regression"
    features: Dict[str, Any]


@router.post("/predict/interactive")
async def predict_interactive(
    request: InteractivePredictRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Interactive prediction endpoint that returns rich, model-specific output
    for the prediction playground UI.

    For Random Forest: returns per-tree predictions, vote summary, probabilities.
    For Decision Tree: returns the decision path through the tree.
    """
    import joblib
    import numpy as np
    from pathlib import Path

    try:
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=400, detail=f"Model not found: {model_path}")

        model_data = joblib.load(model_path)

        # model_data is a dict: { model, feature_names, class_names, training_metadata, ... }
        sklearn_model = model_data["model"]
        feature_names = model_data.get("feature_names", [])
        class_names = model_data.get("class_names", [])
        training_metadata = model_data.get("training_metadata", {})

        # Build feature DataFrame in correct order
        import pandas as pd
        feature_values = {fn: float(request.features.get(fn, 0)) for fn in feature_names}
        df = pd.DataFrame([feature_values])
        X = df[feature_names].values

        # Core prediction
        prediction_raw = sklearn_model.predict(X)[0]
        prediction = prediction_raw.item() if hasattr(prediction_raw, "item") else prediction_raw

        # Probabilities (classification only)
        probabilities = {}
        confidence = 1.0
        if request.task_type == "classification" and hasattr(sklearn_model, "predict_proba"):
            proba = sklearn_model.predict_proba(X)[0]
            classes = sklearn_model.classes_
            for cls, p in zip(classes, proba):
                probabilities[str(cls)] = round(float(p), 4)
            pred_idx = list(classes).index(prediction_raw) if prediction_raw in classes else 0
            confidence = round(float(proba[pred_idx]), 4)

        result: Dict[str, Any] = {
            "prediction": str(prediction) if request.task_type == "classification" else round(float(prediction), 4),
            "probabilities": probabilities,
            "confidence": confidence,
            "model_type": request.model_type,
            "task_type": request.task_type,
            "class_names": [str(c) for c in class_names] if class_names else [],
            "feature_stats": training_metadata.get("feature_stats", []),
        }

        # --- Random Forest specifics: per-tree predictions ---
        if request.model_type == "random_forest" and hasattr(sklearn_model, "estimators_"):
            per_tree = []
            vote_summary: Dict[str, Any] = {}
            trees = sklearn_model.estimators_[:10]  # cap at 10 trees for UI
            for i, tree in enumerate(trees):
                tree_pred_raw = tree.predict(X)[0]
                tree_pred = tree_pred_raw.item() if hasattr(tree_pred_raw, "item") else tree_pred_raw
                entry: Dict[str, Any] = {"tree_index": i}
                if request.task_type == "classification":
                    label = str(tree_pred)
                    entry["prediction"] = label
                    vote_summary[label] = vote_summary.get(label, 0) + 1
                    # Per-tree class probabilities
                    if hasattr(tree, "predict_proba"):
                        try:
                            tp = tree.predict_proba(X)[0]
                            entry["probabilities"] = {
                                str(c): round(float(p), 4)
                                for c, p in zip(sklearn_model.classes_, tp)
                            }
                        except Exception:
                            pass
                else:
                    val = round(float(tree_pred), 4)
                    entry["prediction"] = str(val)
                    entry["numeric_value"] = val
                # Extract full tree structure (capped at depth 4) + highlight prediction path
                try:
                    ts = tree.tree_
                    MAX_DEPTH = 4
                    # Walk prediction path first to know which nodes are on it
                    path_node_ids = set()
                    nid = 0
                    while True:
                        path_node_ids.add(int(nid))
                        if ts.children_left[nid] == ts.children_right[nid]:
                            break
                        fi = ts.feature[nid]
                        fn = feature_names[fi] if fi < len(feature_names) else f"feature_{fi}"
                        fv = float(feature_values.get(fn, 0))
                        nid = ts.children_left[nid] if fv <= ts.threshold[nid] else ts.children_right[nid]

                    # BFS to extract all nodes up to MAX_DEPTH
                    tree_nodes = []
                    bfs_queue = [(0, 0)]  # (node_id, depth)
                    while bfs_queue:
                        nid, depth = bfs_queue.pop(0)
                        if depth > MAX_DEPTH:
                            continue
                        is_leaf = ts.children_left[nid] == ts.children_right[nid]
                        node_info: Dict[str, Any] = {
                            "id": int(nid),
                            "depth": depth,
                            "n_samples": int(ts.n_node_samples[nid]),
                            "impurity": round(float(ts.impurity[nid]), 4),
                            "on_path": int(nid) in path_node_ids,
                        }
                        if is_leaf or depth == MAX_DEPTH:
                            node_info["type"] = "leaf"
                            if request.task_type == "classification":
                                ci = int(np.argmax(ts.value[nid]))
                                node_info["class_label"] = str(class_names[ci]) if ci < len(class_names) else str(ci)
                            else:
                                node_info["value"] = round(float(ts.value[nid][0][0]), 4)
                        else:
                            node_info["type"] = "internal"
                            fi = ts.feature[nid]
                            node_info["feature"] = feature_names[fi] if fi < len(feature_names) else f"feature_{fi}"
                            node_info["threshold"] = round(float(ts.threshold[nid]), 4)
                            lc = int(ts.children_left[nid])
                            rc = int(ts.children_right[nid])
                            node_info["left_child"] = lc
                            node_info["right_child"] = rc
                            bfs_queue.append((lc, depth + 1))
                            bfs_queue.append((rc, depth + 1))
                        tree_nodes.append(node_info)
                    entry["tree_structure"] = tree_nodes
                    entry["tree_depth"] = int(tree.get_depth())
                    entry["n_leaves"] = int(tree.get_n_leaves())
                except Exception:
                    pass  # Graceful degradation

                per_tree.append(entry)

            result["per_tree_predictions"] = per_tree
            if request.task_type == "classification":
                result["vote_summary"] = vote_summary
            else:
                vals = [e["numeric_value"] for e in per_tree if "numeric_value" in e]
                result["regression_mean"] = round(sum(vals) / len(vals), 4) if vals else None

        # --- Decision Tree specifics: decision path ---
        if request.model_type == "decision_tree" and hasattr(sklearn_model, "tree_"):
            tree = sklearn_model.tree_
            path_nodes = []
            node_id = 0
            while True:
                is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
                if is_leaf:
                    # Leaf node
                    if request.task_type == "classification":
                        class_idx = int(np.argmax(tree.value[node_id]))
                        leaf_label = str(class_names[class_idx]) if class_idx < len(class_names) else str(class_idx)
                        dist = tree.value[node_id][0].tolist()
                        total = sum(dist)
                        leaf_probs = {}
                        for ci, cnt in enumerate(dist):
                            cn = str(class_names[ci]) if ci < len(class_names) else str(ci)
                            leaf_probs[cn] = round(cnt / total, 4) if total > 0 else 0
                        path_nodes.append({
                            "node_id": int(node_id),
                            "type": "leaf",
                            "prediction": leaf_label,
                            "samples": int(tree.n_node_samples[node_id]),
                            "probabilities": leaf_probs,
                        })
                    else:
                        val = round(float(tree.value[node_id][0][0]), 4)
                        path_nodes.append({
                            "node_id": int(node_id),
                            "type": "leaf",
                            "prediction": str(val),
                            "numeric_value": val,
                            "samples": int(tree.n_node_samples[node_id]),
                        })
                    break

                feat_idx = tree.feature[node_id]
                threshold = round(float(tree.threshold[node_id]), 4)
                feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
                feat_value = float(feature_values.get(feat_name, 0))
                go_left = feat_value <= threshold
                path_nodes.append({
                    "node_id": int(node_id),
                    "type": "split",
                    "feature": feat_name,
                    "threshold": threshold,
                    "value": round(feat_value, 4),
                    "direction": "left" if go_left else "right",
                    "samples": int(tree.n_node_samples[node_id]),
                    "impurity": round(float(tree.impurity[node_id]), 4),
                })
                node_id = tree.children_left[node_id] if go_left else tree.children_right[node_id]

            result["decision_path"] = path_nodes

        return make_json_serializable(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interactive prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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


# ========== CAMERA CAPTURE ENDPOINTS ==========


class CameraDatasetRequest(BaseModel):
    """Request body for building a camera capture dataset."""
    class_names: List[str]
    target_size: str = "28x28"
    images_per_class: Dict[str, List[List[float]]]  # className → list of pixel arrays


class CameraDatasetResponse(BaseModel):
    dataset_id: str
    n_rows: int
    n_columns: int
    n_classes: int
    class_names: List[str]
    image_width: int
    image_height: int


class CameraPredictRequest(BaseModel):
    """Request body for live-camera inference."""
    model_path: str
    pixels: List[float]          # flat grayscale pixel array


class CameraPredictResponse(BaseModel):
    class_name: str
    confidence: float
    all_scores: List[Dict[str, Any]]


@router.post("/camera/dataset", response_model=CameraDatasetResponse)
async def build_camera_dataset(
    request: CameraDatasetRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Build a CSV dataset from camera-captured pixel arrays.

    The frontend captures images per class, converts them to flattened grayscale
    pixel arrays, and sends them here.  We save a CSV in the uploads directory
    (same format as image_dataset_node) and return a dataset_id usable by the
    rest of the image pipeline.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from app.core.config import settings
    from app.utils.ids import generate_id

    try:
        size_parts = (request.target_size or "28x28").lower().split("x")
        width = int(size_parts[0]) if size_parts else 28
        height = int(size_parts[1]) if len(size_parts) > 1 else 28
        n_pixels = width * height

        rows = []
        for class_idx, class_name in enumerate(request.class_names):
            pixel_arrays = request.images_per_class.get(class_name, [])
            for pixels in pixel_arrays:
                if len(pixels) != n_pixels:
                    raise ValueError(
                        f"Expected {n_pixels} pixels for {width}×{height} image, "
                        f"got {len(pixels)} for class '{class_name}'"
                    )
                row = {f"pixel_{i}": float(v) for i, v in enumerate(pixels)}
                row["label"] = class_idx
                rows.append(row)

        if not rows:
            raise ValueError("No images provided — capture at least 1 image per class")

        df = pd.DataFrame(rows)
        dataset_id = generate_id("cam")
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{dataset_id}.csv"
        df.to_csv(file_path, index=False)

        logger.info(
            f"Camera dataset built: {dataset_id}, {len(df)} images, "
            f"{len(request.class_names)} classes, {width}×{height}px"
        )

        return CameraDatasetResponse(
            dataset_id=dataset_id,
            n_rows=len(df),
            n_columns=len(df.columns),
            n_classes=len(request.class_names),
            class_names=request.class_names,
            image_width=width,
            image_height=height,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Camera dataset build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# In-memory cache for CNN models (avoid reloading on every camera frame)
_cnn_model_cache: Dict[str, Any] = {}


@router.post("/camera/predict", response_model=CameraPredictResponse)
async def camera_live_predict(
    request: CameraPredictRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Run inference on a single camera frame (flat grayscale pixel array).

    Uses Transfer Learning (MobileNetV2) model. Reads .meta.json for image dimensions.
    Designed for low-latency live-camera testing in the Image Predictions node.
    """
    import numpy as np
    from pathlib import Path
    import json as _json

    try:
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")

        # Read meta.json for image dimensions and class names
        meta_path = model_path.with_suffix(".meta.json")
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)

        X = np.array(request.pixels, dtype=float).reshape(1, -1)

        # Normalize incoming [0-255] camera pixels to [0-1] to match training
        if X.max() > 1.0:
            X = X / 255.0

        # Transfer Learning (MobileNetV2) — the only image model type
        import tensorflow as tf

        # Load from cache or disk
        cache_key = str(model_path)
        if cache_key not in _cnn_model_cache:
            _cnn_model_cache[cache_key] = tf.keras.models.load_model(str(model_path))
        tl_model = _cnn_model_cache[cache_key]

        target_size = meta.get("target_size", 96)
        img_width = meta.get("image_width", 28)
        img_height = meta.get("image_height", 28)

        # Reshape flat grayscale -> resize to target_size -> RGB -> preprocess
        X_img = X.reshape(1, img_height, img_width, 1).astype(np.float32)
        X_resized = tf.image.resize(X_img, [target_size, target_size]).numpy()
        X_rgb = np.repeat(X_resized, 3, axis=-1)
        X_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(X_rgb * 255.0)

        proba = tl_model.predict(X_rgb, verbose=0)[0]
        predicted_idx = int(np.argmax(proba))
        confidence = float(proba[predicted_idx])

        class_names = meta.get("class_names", [str(i) for i in range(len(proba))])
        all_scores = [
            {"class_name": str(class_names[i] if i < len(class_names) else i), "score": round(float(p), 4)}
            for i, p in sorted(enumerate(proba), key=lambda x: -x[1])
        ]

        return CameraPredictResponse(
            class_name=str(predicted_idx),
            confidence=round(confidence, 4),
            all_scores=all_scores,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Camera live prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


# ========== POSE CAPTURE ENDPOINTS ==========

N_POSE_LANDMARKS = 33
N_POSE_FEATURES = N_POSE_LANDMARKS * 4  # x, y, z, visibility per landmark


class PoseDatasetRequest(BaseModel):
    """Request body for building a pose landmark dataset."""
    class_names: List[str]
    landmarks_per_class: Dict[str, List[List[float]]]  # className → list of 132-float arrays


class PoseDatasetResponse(BaseModel):
    dataset_id: str
    n_rows: int
    n_columns: int
    n_classes: int
    class_names: List[str]


class PosePredictRequest(BaseModel):
    """Request body for live-pose inference."""
    model_path: str
    landmarks: List[float]  # flat 132-float array


class PosePredictResponse(BaseModel):
    class_name: str
    confidence: float
    all_scores: List[Dict[str, Any]]


@router.post("/pose/dataset", response_model=PoseDatasetResponse)
async def build_pose_dataset(
    request: PoseDatasetRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Build a CSV dataset from pose landmark arrays captured in the browser.

    The frontend uses MediaPipe PoseLandmarker to extract 33 body landmarks
    (x, y, z, visibility) per frame. Each capture is a flat 132-float array.
    We save as CSV with columns: lm_0_x, lm_0_y, lm_0_z, lm_0_vis, ..., lm_32_vis, label
    """
    import pandas as pd
    from pathlib import Path
    from app.core.config import settings

    if len(request.class_names) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 pose classes are required",
        )

    rows = []
    for class_idx, class_name in enumerate(request.class_names):
        landmark_arrays = request.landmarks_per_class.get(class_name, [])
        for landmarks in landmark_arrays:
            if len(landmarks) != N_POSE_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {N_POSE_FEATURES} values per pose (33 landmarks x 4), "
                           f"got {len(landmarks)} for class '{class_name}'",
                )
            row: Dict[str, Any] = {}
            for i in range(N_POSE_LANDMARKS):
                base = i * 4
                row[f"lm_{i}_x"] = landmarks[base]
                row[f"lm_{i}_y"] = landmarks[base + 1]
                row[f"lm_{i}_z"] = landmarks[base + 2]
                row[f"lm_{i}_vis"] = landmarks[base + 3]
            row["label"] = class_idx
            rows.append(row)

    if len(rows) == 0:
        raise HTTPException(status_code=400, detail="No pose data provided")

    df = pd.DataFrame(rows)

    # Generate dataset ID and save
    import uuid
    dataset_id = f"pose_{uuid.uuid4().hex[:12]}"
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{dataset_id}.csv"
    df.to_csv(file_path, index=False)

    # Also save a metadata JSON for the dataset
    meta = {
        "data_type": "pose",
        "n_landmarks": N_POSE_LANDMARKS,
        "n_features": N_POSE_FEATURES,
        "n_classes": len(request.class_names),
        "class_names": request.class_names,
    }
    meta_path = upload_dir / f"{dataset_id}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Built pose dataset: {dataset_id} with {len(df)} samples, "
        f"{len(request.class_names)} classes"
    )

    return PoseDatasetResponse(
        dataset_id=dataset_id,
        n_rows=len(df),
        n_columns=len(df.columns),
        n_classes=len(request.class_names),
        class_names=request.class_names,
    )


# Cache for pose models
_pose_model_cache: Dict[str, Any] = {}


@router.post("/pose/predict", response_model=PosePredictResponse)
async def pose_live_predict(
    request: PosePredictRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Run inference on a single pose (132 landmark values) using a trained model."""
    import numpy as np
    import joblib
    from pathlib import Path

    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    # Read metadata
    meta_path = model_path.with_suffix(".meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    class_names = meta.get("class_names", [])

    # Load model (cached)
    cache_key = str(model_path)
    if cache_key not in _pose_model_cache:
        _pose_model_cache[cache_key] = joblib.load(model_path)
    model = _pose_model_cache[cache_key]

    # Prepare input
    X = np.array(request.landmarks, dtype=float).reshape(1, -1)

    # Predict
    predicted_idx = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        all_scores = [
            {
                "class_name": str(class_names[i] if i < len(class_names) else i),
                "score": round(float(p), 4),
            }
            for i, p in sorted(enumerate(proba), key=lambda x: -x[1])
        ]
        confidence = float(proba[predicted_idx]) if predicted_idx < len(proba) else 1.0
    else:
        all_scores = [{"class_name": str(predicted_idx), "score": 1.0}]
        confidence = 1.0

    return PosePredictResponse(
        class_name=str(
            class_names[predicted_idx] if predicted_idx < len(class_names) else predicted_idx
        ),
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )
