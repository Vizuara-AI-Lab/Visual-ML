"""
Dataset API endpoints for dataset upload, management, and retrieval.
Production-ready with S3 support, validation, and metadata caching.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.models.user import Student
from app.core.security import get_current_student
from app.core.logging import logger
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

    **Student Authentication Required**
    """
    try:
        # Verify project exists and belongs to student
        from app.models.genai import GenAIPipeline

        project = (
            db.query(GenAIPipeline)
            .filter(GenAIPipeline.id == project_id, GenAIPipeline.studentId == student.id)
            .first()
        )

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found or not owned by you",
            )

        # Read file content
        file_content = await file.read()

        # Validate file extension
        filename = file.filename or "unknown.csv"
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

        # Create upload node and process
        upload_node = UploadFileNode()

        # Prepare input
        upload_input = UploadFileInput(
            node_type="upload_file",
            file_content=file_content,
            filename=file.filename or "unknown.csv",
            content_type=file.content_type or "text/csv",
            project_id=project_id,
            user_id=student.id,
            node_id=None,
        )

        # Execute upload (handles S3 upload + DB metadata)
        result = await upload_node.execute(upload_input)

        logger.info(
            f"Dataset uploaded - ID: {result.dataset_id}, "
            f"Project: {project_id}, Student: {student.id}, "
            f"Storage: {result.storage_backend}"
        )

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
                "preview": result.preview,
            },
        }

    except HTTPException:
        raise
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

    # Get all datasets for this project
    datasets = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id, Dataset.is_deleted == False)
        .order_by(Dataset.created_at.desc())
        .all()
    )

    return [dataset.to_dict() for dataset in datasets]


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
