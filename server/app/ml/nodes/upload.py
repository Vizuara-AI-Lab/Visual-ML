"""
Upload File Node - Handles dataset uploads with validation.
Stores files in S3 for production scalability with DB metadata.
"""

from typing import Type, Optional, Dict, Any
import pandas as pd
import io
from pathlib import Path
from pydantic import Field, field_validator
from datetime import datetime
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.core.exceptions import FileUploadError, InvalidDatasetError
from app.core.config import settings
from app.core.logging import logger
from app.utils.ids import generate_id
from app.services.s3_service import s3_service


class UploadFileInput(NodeInput):
    """Input schema for UploadFile node."""

    file_content: bytes = Field(..., description="File content as bytes")
    filename: str = Field(..., description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")
    project_id: Optional[int] = Field(None, description="Project ID for dataset association")
    user_id: Optional[int] = Field(None, description="User ID for ownership")
    node_id: Optional[int] = Field(None, description="Node ID for linking")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename extension."""
        if not v:
            raise ValueError("Filename cannot be empty")

        # Extract extension
        ext = Path(v).suffix.lower()

        # Validate extension
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension: {ext}. "
                f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        return v

    @field_validator("file_content")
    @classmethod
    def validate_file_size(cls, v: bytes) -> bytes:
        """Validate file size."""
        size_mb = len(v) / (1024 * 1024)

        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"File size ({size_mb:.2f} MB) exceeds maximum "
                f"allowed size ({settings.MAX_UPLOAD_SIZE_MB} MB)"
            )

        return v


class UploadFileOutput(NodeOutput):
    """Output schema for UploadFile node."""

    dataset_id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Stored filename")
    file_path: str = Field(..., description="Path to stored file (S3 key or local path)")
    storage_backend: str = Field(..., description="Storage backend used (s3 or local)")
    s3_bucket: Optional[str] = Field(None, description="S3 bucket if using S3")
    s3_key: Optional[str] = Field(None, description="S3 object key if using S3")
    n_rows: int = Field(..., description="Number of rows in dataset")
    n_columns: int = Field(..., description="Number of columns in dataset")
    columns: list[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types of columns")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    preview: list[Dict[str, Any]] = Field(..., description="First 10 rows preview")
    file_size: int = Field(..., description="File size in bytes")


class UploadFileNode(BaseNode):
    """
    Upload File Node - First node in ML pipeline.

    Responsibilities:
    - Accept CSV dataset upload
    - Validate file type, extension, size
    - Parse and validate CSV structure
    - Store file (local filesystem, optionally S3)
    - Generate unique dataset ID
    - Return dataset metadata and reference

    Production features:
    - File size validation
    - Extension whitelist
    - CSV structure validation
    - Memory usage tracking
    - S3 support for scalability
    """

    node_type = "upload_file"

    def __init__(self, node_id: Optional[str] = None, storage_backend: str = None):
        """
        Initialize Upload File node.

        Args:
            node_id: Unique node identifier
            storage_backend: Storage backend to use ('local' or 's3', auto-detects from settings)
        """
        super().__init__(node_id)

        # Auto-detect storage backend from settings if not specified
        if storage_backend is None:
            self.storage_backend = "s3" if settings.USE_S3 else "local"
        else:
            self.storage_backend = storage_backend

        # Create upload directory if using local storage
        if self.storage_backend == "local":
            Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

        logger.info(f"Upload node initialized with storage backend: {self.storage_backend}")

    def get_input_schema(self) -> Type[NodeInput]:
        """Return input schema."""
        return UploadFileInput

    def get_output_schema(self) -> Type[NodeOutput]:
        """Return output schema."""
        return UploadFileOutput

    async def _execute(self, input_data: UploadFileInput) -> UploadFileOutput:
        """
        Execute file upload and validation.

        Args:
            input_data: Validated input data

        Returns:
            Upload result with dataset metadata
        """
        try:
            logger.info(f"Processing file upload: {input_data.filename}")

            # Generate unique dataset ID
            dataset_id = generate_id("dataset")

            # Parse CSV
            df = self._parse_csv(input_data.file_content, input_data.filename)

            # Validate CSV structure
            self._validate_csv_structure(df)

            # Store file (S3 or local)
            storage_result = await self._store_file(
                dataset_id=dataset_id,
                filename=input_data.filename,
                file_content=input_data.file_content,
                project_id=input_data.project_id,
                user_id=input_data.user_id,
                df=df,
            )

            # Calculate memory usage
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

            # Get data types
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Generate preview (first 10 rows)
            preview = df.head(10).to_dict(orient="records")

            logger.info(
                f"File uploaded successfully - Dataset ID: {dataset_id}, "
                f"Storage: {storage_result['storage_backend']}, "
                f"Rows: {len(df)}, Columns: {len(df.columns)}"
            )

            return UploadFileOutput(
                node_type=self.node_type,
                execution_time_ms=0,  # Set by base class
                dataset_id=dataset_id,
                filename=input_data.filename,
                file_path=storage_result["file_path"],
                storage_backend=storage_result["storage_backend"],
                s3_bucket=storage_result.get("s3_bucket"),
                s3_key=storage_result.get("s3_key"),
                n_rows=len(df),
                n_columns=len(df.columns),
                columns=df.columns.tolist(),
                dtypes=dtypes,
                memory_usage_mb=round(memory_usage_mb, 2),
                preview=preview,
                file_size=len(input_data.file_content),
            )

        except FileUploadError:
            raise
        except InvalidDatasetError:
            raise
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}", exc_info=True)
            raise FileUploadError(reason=str(e), filename=input_data.filename)

    def _parse_csv(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse CSV file content.

        Args:
            file_content: File bytes
            filename: Original filename

        Returns:
            Parsed DataFrame

        Raises:
            FileUploadError: If parsing fails
        """
        try:
            # Try to parse CSV
            df = pd.read_csv(io.BytesIO(file_content))

            if df.empty:
                raise FileUploadError(reason="CSV file is empty", filename=filename)

            return df

        except pd.errors.EmptyDataError:
            raise FileUploadError(reason="CSV file is empty or contains no data", filename=filename)
        except pd.errors.ParserError as e:
            raise FileUploadError(reason=f"CSV parsing error: {str(e)}", filename=filename)
        except Exception as e:
            raise FileUploadError(reason=f"Failed to read CSV: {str(e)}", filename=filename)

    def _validate_csv_structure(self, df: pd.DataFrame) -> None:
        """
        Validate CSV structure.

        Args:
            df: DataFrame to validate

        Raises:
            InvalidDatasetError: If validation fails
        """
        # Check for minimum rows
        if len(df) < 2:
            raise InvalidDatasetError(
                reason=f"Dataset has only {len(df)} rows (minimum 2 required)",
                expected_format="CSV with at least 2 rows (header + data)",
            )

        # Check for columns
        if len(df.columns) < 1:
            raise InvalidDatasetError(
                reason="Dataset has no columns", expected_format="CSV with at least 1 column"
            )

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            raise InvalidDatasetError(
                reason=f"Duplicate column names found: {set(duplicates)}",
                expected_format="CSV with unique column names",
            )

        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
        if unnamed_cols:
            raise InvalidDatasetError(
                reason=f"Dataset has unnamed columns: {unnamed_cols}",
                expected_format="CSV with all columns named in header row",
            )

    async def _store_file(
        self,
        dataset_id: str,
        filename: str,
        file_content: bytes,
        project_id: Optional[int] = None,
        user_id: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Store file to S3 or local storage and save metadata to database.

        Args:
            dataset_id: Unique dataset identifier
            filename: Original filename
            file_content: File bytes
            project_id: Project ID for association
            user_id: User ID for ownership
            df: Parsed DataFrame for metadata

        Returns:
            Dict with storage details (storage_backend, file_path, s3_bucket, s3_key)
        """
        # Generate storage filename
        ext = Path(filename).suffix or ".csv"
        storage_filename = f"{dataset_id}{ext}"

        if self.storage_backend == "s3" and settings.USE_S3:
            try:
                # Generate S3 key with project-based structure
                s3_key = s3_service.generate_s3_key(
                    project_id=project_id or 0,
                    dataset_id=dataset_id,
                    filename=filename,
                    use_date_partition=settings.S3_USE_DATE_PARTITION,
                )

                # Upload to S3
                s3_url = await s3_service.upload_file(
                    file_content=file_content,
                    s3_key=s3_key,
                    content_type="text/csv",
                    metadata={
                        "dataset_id": dataset_id,
                        "project_id": str(project_id) if project_id else "",
                        "original_filename": filename,
                    },
                )

                logger.info(f"File uploaded to S3: {s3_url}")

                # Save metadata to database (if db available)
                if project_id and user_id and df is not None:
                    await self._save_dataset_metadata(
                        dataset_id=dataset_id,
                        filename=filename,
                        file_content=file_content,
                        project_id=project_id,
                        user_id=user_id,
                        df=df,
                        storage_backend="s3",
                        s3_bucket=settings.S3_BUCKET,
                        s3_key=s3_key,
                    )

                return {
                    "storage_backend": "s3",
                    "file_path": s3_key,
                    "s3_bucket": settings.S3_BUCKET,
                    "s3_key": s3_key,
                }

            except Exception as e:
                logger.error(f"S3 upload failed, falling back to local: {str(e)}")
                # Fallback to local storage
                self.storage_backend = "local"

        # Local storage (default or fallback)
        file_path = Path(settings.UPLOAD_DIR) / storage_filename
        file_path.write_bytes(file_content)
        logger.info(f"File stored locally: {file_path}")

        # Save metadata to database (if db available)
        if project_id and user_id and df is not None:
            await self._save_dataset_metadata(
                dataset_id=dataset_id,
                filename=filename,
                file_content=file_content,
                project_id=project_id,
                user_id=user_id,
                df=df,
                storage_backend="local",
                local_path=str(file_path),
            )

        return {
            "storage_backend": "local",
            "file_path": str(file_path),
            "s3_bucket": None,
            "s3_key": None,
        }

    async def _save_dataset_metadata(
        self,
        dataset_id: str,
        filename: str,
        file_content: bytes,
        project_id: int,
        user_id: int,
        df: pd.DataFrame,
        storage_backend: str,
        s3_bucket: Optional[str] = None,
        s3_key: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        """
        Save dataset metadata to database.

        Args:
            dataset_id: Unique dataset ID
            filename: Original filename
            file_content: File bytes
            project_id: Project ID
            user_id: User ID
            df: Parsed DataFrame
            storage_backend: Storage type (s3 or local)
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Local file path
        """
        try:
            from app.models.dataset import Dataset
            from app.db.session import SessionLocal

            # Calculate metadata
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            preview_data = df.head(10).to_dict(orient="records")

            # Create database record
            db = SessionLocal()
            try:
                dataset = Dataset(
                    dataset_id=dataset_id,
                    project_id=project_id,
                    user_id=user_id,
                    filename=filename,
                    content_type="text/csv",
                    file_size=len(file_content),
                    file_extension=Path(filename).suffix or ".csv",
                    storage_backend=storage_backend,
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    s3_region=settings.S3_REGION if storage_backend == "s3" else None,
                    local_path=local_path,
                    n_rows=len(df),
                    n_columns=len(df.columns),
                    columns=df.columns.tolist(),
                    dtypes=dtypes,
                    memory_usage_mb=round(memory_usage_mb, 2),
                    preview_data=preview_data,
                    is_validated=True,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(dataset)
                db.commit()
                logger.info(f"Dataset metadata saved to database: {dataset_id}")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Failed to save dataset metadata: {str(e)}")
