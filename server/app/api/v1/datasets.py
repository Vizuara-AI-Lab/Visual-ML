"""
Dataset API endpoints for dataset upload, management, and retrieval.
Production-ready with S3 support, validation, and metadata caching.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.models.user import Student
from app.core.security import get_current_student
from app.core.logging import logger
from app.core.redis_cache import redis_cache
from app.db.session import get_db
from app.ml.nodes.upload import UploadFileNode, UploadFileInput
from app.services.s3_service import s3_service
from app.core.config import settings
from datetime import datetime
import pandas as pd
import io

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file to upload"),
    project_id: int = Query(..., description="Project ID to associate dataset with"),
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Upload a dataset file (CSV) to S3 and store metadata in database.

    **Flow:**
    1. User creates a project
    2. User enters playground and adds "Upload Dataset" node
    3. User uploads CSV through this endpoint
    4. File is validated, uploaded to S3, and metadata is stored
    5. Dataset reference is returned for use in ML pipeline

    **Authentication Required - JWT from HTTP-only cookies**
    """
    logger.info(
        f"📤 Dataset upload request - Project: {project_id}, User: {student.id} ({student.emailId})"
    )
    logger.info(f"📄 File: {file.filename}, Content-Type: {file.content_type}")

    # Verify project ownership
    from app.models.genai import GenAIPipeline

    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        logger.warning(f"Project {project_id} not found or not owned by user {student.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or you don't have permission to upload datasets to it",
        )

    try:
        student_id = student.id

        # Read file content
        file_content = await file.read()
        logger.info(
            f"✅ File content read - Size: {len(file_content)} bytes ({len(file_content) / (1024 * 1024):.2f} MB)"
        )

        # Validate file extension
        filename = file.filename or "unknown.csv"
        logger.info(f"📝 Validating file: {filename}")
        if not filename.endswith((".csv", ".txt", ".json")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: CSV, TXT, JSON",
            )

        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large ({file_size_mb:.2f}MB). Max: {settings.MAX_UPLOAD_SIZE_MB}MB",
            )

        # Create upload node (hardcode s3 for now)
        upload_node = UploadFileNode(storage_backend="s3")
        logger.info(f"🔧 Upload node created with storage backend: s3")

        # Set DB session on node
        upload_node.db = db

        # Prepare input (removed node_type as it's not in UploadFileInput schema)
        upload_input = UploadFileInput(
            file_content=file_content,
            filename=file.filename or "unknown.csv",
            content_type=file.content_type or "text/csv",
            project_id=project_id,
            user_id=student_id,
            node_id=None,
        )

        logger.info(f"📦 Upload input prepared - User ID: {student_id}")
        result = await upload_node.execute(upload_input.model_dump())

        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "dataset": {
                "dataset_id": result.dataset_id,
                "filename": result.filename,
                "file_path": result.file_path,
                "storage_backend": result.storage_backend,
                "s3_bucket": result.s3_bucket,
                "s3_key": result.s3_key,
                "n_rows": result.n_rows,
                "n_columns": result.n_columns,
                "columns": result.columns,
                "dtypes": result.dtypes,
                "memory_usage_mb": result.memory_usage_mb,
                "file_size": result.file_size,
            },
        }

    except HTTPException:
        # Re-raise HTTP exceptions (401, 403, 404, etc.)
        raise
    except FileUploadError as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File upload failed: {str(e)}",
        )
    except InvalidDatasetError as e:
        logger.error(f"Invalid dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Dataset upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload dataset: {str(e)}",
        )


@router.get("/project/{project_id}", response_model=List[dict])
async def list_project_datasets(
    project_id: int,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    List all datasets for a specific project.

    **Student Authentication Required**
    """
    # Verify project ownership
    from app.models.genai import GenAIPipeline

    project = (
        db.query(GenAIPipeline)
        .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
        .first()
    )

    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # Try cache first
    cache_key = f"datasets:project:{project_id}"
    cached = await redis_cache.get(cache_key)
    if cached:
        logger.debug(f"Returning cached datasets for project {project_id}")
        return cached

    # Get all datasets for this project
    datasets = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id, Dataset.is_deleted == False)
        .order_by(Dataset.created_at.desc())
        .all()
    )

    result = [dataset.to_dict() for dataset in datasets]

    # Cache for 10 minutes
    await redis_cache.set(cache_key, result, ttl=600)

    return result


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get dataset metadata by ID.

    **Student Authentication Required**
    """
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == student.id,
            Dataset.is_deleted == False,
        )
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    # Update last accessed timestamp
    from sqlalchemy import update

    db.execute(
        update(Dataset)
        .where(Dataset.dataset_id == dataset_id)
        .values(last_accessed_at=datetime.utcnow())
    )
    db.commit()

    return dataset.to_dict()


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get presigned URL for downloading dataset from S3.
    For local storage, returns direct download link.

    **Student Authentication Required**
    """
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == student.id,
            Dataset.is_deleted == False,
        )
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    try:
        if dataset.storage_backend == "s3":
            # Generate presigned URL for S3 download
            s3_key_val = str(dataset.s3_key) if dataset.s3_key else ""
            download_url = await s3_service.get_presigned_url(
                s3_key=s3_key_val, expiry=settings.S3_PRESIGNED_URL_EXPIRY
            )

            return {
                "download_url": download_url,
                "expires_in": settings.S3_PRESIGNED_URL_EXPIRY,
                "filename": str(dataset.filename),
                "storage": "s3",
            }
        else:
            # For local storage, return file path
            # In production, you'd want to use FileResponse or stream the file
            from fastapi.responses import FileResponse

            local_path_val = str(dataset.local_path) if dataset.local_path else ""
            filename_val = str(dataset.filename) if dataset.filename else "dataset.csv"
            return FileResponse(local_path_val, filename=filename_val, media_type="text/csv")

    except Exception as e:
        logger.error(f"Failed to generate download URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL",
        )


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get cached preview of dataset (first 10 rows) without downloading full file.
    Super fast - returns data from database cache.

    **Student Authentication Required**
    """
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.dataset_id == dataset_id,
            Dataset.user_id == student.id,
            Dataset.is_deleted == False,
        )
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "n_rows": dataset.n_rows,
        "n_columns": dataset.n_columns,
        "columns": dataset.columns,
        "dtypes": dataset.dtypes,
        "memory_usage_mb": dataset.memory_usage_mb,
        "preview": dataset.preview_data,
        "created_at": dataset.created_at.isoformat(),
    }


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    permanent: bool = Query(False, description="Permanently delete from S3 (default: soft delete)"),
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Delete dataset (soft delete by default, or permanent delete from S3).

    **Student Authentication Required**
    """
    dataset = (
        db.query(Dataset)
        .filter(Dataset.dataset_id == dataset_id, Dataset.user_id == student.id)
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    try:
        if permanent:
            # Permanently delete from S3
            if dataset.storage_backend == "s3":
                s3_key_val = str(dataset.s3_key) if dataset.s3_key else ""
                if s3_key_val:
                    await s3_service.delete_file(s3_key_val)

            # Delete from database
            db.delete(dataset)
            db.commit()

            logger.info(f"Dataset permanently deleted: {dataset_id}")
            return {"message": "Dataset permanently deleted"}
        else:
            # Soft delete
            from sqlalchemy import update

            db.execute(
                update(Dataset)
                .where(Dataset.dataset_id == dataset_id)
                .values(is_deleted=True, updated_at=datetime.utcnow())
            )
            db.commit()

            logger.info(f"Dataset soft deleted: {dataset_id}")
            return {"message": "Dataset deleted (soft delete)"}

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete dataset"
        )


@router.get("")
async def list_all_datasets(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List all datasets for the current student across all projects.

    **Student Authentication Required**
    """
    datasets = (
        db.query(Dataset)
        .filter(Dataset.user_id == student.id, Dataset.is_deleted == False)
        .order_by(Dataset.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )

    total = (
        db.query(Dataset).filter(Dataset.user_id == student.id, Dataset.is_deleted == False).count()
    )

    return {
        "datasets": [dataset.to_dict() for dataset in datasets],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/file/{file_id}/download")
async def download_file_from_uploads(
    file_id: str,
    student: Student = Depends(get_current_student),
):
    """
    Download a file directly from the uploads folder.

    This endpoint is used for downloading processed datasets (scaled, encoded, etc.)
    that are saved in the uploads folder but not tracked in the database.

    **Student Authentication Required**
    """
    from fastapi.responses import FileResponse
    from pathlib import Path

    try:
        # Construct file path - check both with and without .csv extension
        file_path = Path(settings.UPLOAD_DIR) / f"{file_id}.csv"

        if not file_path.exists():
            # Try without extension
            file_path = Path(settings.UPLOAD_DIR) / file_id
            if not file_path.exists():
                logger.error(f"File not found: {file_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {file_id}"
                )

        # Security check - ensure file is within uploads directory
        if not str(file_path.resolve()).startswith(str(Path(settings.UPLOAD_DIR).resolve())):
            logger.error(f"Security violation: Attempted path traversal for {file_id}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        logger.info(f"Downloading file from uploads: {file_path}")

        return FileResponse(path=str(file_path), filename=f"{file_id}.csv", media_type="text/csv")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to download file"
        )
