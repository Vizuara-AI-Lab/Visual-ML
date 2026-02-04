"""
Task status and management API endpoints.
Used to check status of async Celery tasks.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any
from app.core.celery_app import celery_app
from app.core.security import get_current_user
from app.core.logging import logger
from celery.result import AsyncResult

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get("/{task_id}/status")
async def get_task_status(task_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get status of a Celery task.

    **Authentication Required**

    Returns task state and progress information:
    - PENDING: Task is waiting to be executed
    - PROGRESS: Task is currently running (with progress info)
    - SUCCESS: Task completed successfully
    - FAILURE: Task failed with error
    - REVOKED: Task was cancelled

    Example response for in-progress task:
    ```json
    {
        "task_id": "abc-123",
        "status": "PROGRESS",
        "result": {
            "status": "Executing node: linear_regression",
            "current_node": 3,
            "total_nodes": 5,
            "percent": 60,
            "node_type": "linear_regression"
        }
    }
    ```

    Example response for completed task:
    ```json
    {
        "task_id": "abc-123",
        "status": "SUCCESS",
        "result": {
            "success": true,
            "pipeline_name": "My Pipeline",
            "results": [...]
        }
    }
    ```
    """
    try:
        task = AsyncResult(task_id, app=celery_app)

        response = {
            "task_id": task_id,
            "status": task.state,
        }

        if task.state == "PENDING":
            response["result"] = {"status": "Task is waiting in queue...", "percent": 0}
        elif task.state == "PROGRESS":
            response["result"] = task.info  # Progress metadata
        elif task.state == "SUCCESS":
            response["result"] = task.result
        elif task.state == "FAILURE":
            response["result"] = {"error": str(task.info), "status": "Task failed"}
        elif task.state == "REVOKED":
            response["result"] = {"status": "Task was cancelled", "percent": 0}
        else:
            response["result"] = task.info

        return response

    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}",
        )


@router.delete("/{task_id}")
async def cancel_task(task_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Cancel a running Celery task.

    **Authentication Required**

    Attempts to revoke (cancel) a task. If the task is already running,
    it may not be cancelled immediately.

    Returns:
    ```json
    {
        "success": true,
        "task_id": "abc-123",
        "message": "Task cancelled successfully"
    }
    ```
    """
    try:
        task = AsyncResult(task_id, app=celery_app)
        task.revoke(terminate=True)  # Force terminate if running

        logger.info(f"Task {task_id} cancelled by user {current_user.get('id')}")

        return {"success": True, "task_id": task_id, "message": "Task cancelled successfully"}

    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}",
        )


@router.get("/{task_id}/result")
async def get_task_result(task_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get the final result of a completed task.

    **Authentication Required**

    Only returns result if task is in SUCCESS state.
    Otherwise returns error or pending status.

    Returns:
    ```json
    {
        "task_id": "abc-123",
        "status": "SUCCESS",
        "result": {
            "success": true,
            "results": [...]
        }
    }
    ```
    """
    try:
        task = AsyncResult(task_id, app=celery_app)

        if task.state == "SUCCESS":
            return {"task_id": task_id, "status": "SUCCESS", "result": task.result}
        elif task.state == "PENDING":
            return {"task_id": task_id, "status": "PENDING", "message": "Task is still pending"}
        elif task.state == "PROGRESS":
            return {
                "task_id": task_id,
                "status": "PROGRESS",
                "message": "Task is still running",
                "progress": task.info,
            }
        elif task.state == "FAILURE":
            return {"task_id": task_id, "status": "FAILURE", "error": str(task.info)}
        else:
            return {"task_id": task_id, "status": task.state, "info": task.info}

    except Exception as e:
        logger.error(f"Failed to get task result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task result: {str(e)}",
        )
