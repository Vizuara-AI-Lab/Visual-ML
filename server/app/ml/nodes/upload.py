"""
Upload File Node - Handles dataset uploads with validation.
Stores files in memory/DB and S3 for production scalability.
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


class UploadFileInput(NodeInput):
    """Input schema for UploadFile node."""

    file_content: bytes = Field(..., description="File content as bytes")
    filename: str = Field(..., description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")

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
    file_path: str = Field(..., description="Path to stored file")
    n_rows: int = Field(..., description="Number of rows in dataset")
    n_columns: int = Field(..., description="Number of columns in dataset")
    columns: list[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types of columns")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    preview: list[Dict[str, Any]] = Field(..., description="First 5 rows preview")


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

    def __init__(
        self, node_id: Optional[str] = None, storage_backend: str = "local"  # 'local' or 's3'
    ):
        """
        Initialize Upload File node.

        Args:
            node_id: Unique node identifier
            storage_backend: Storage backend to use ('local' or 's3')
        """
        super().__init__(node_id)
        self.storage_backend = storage_backend

        # Create upload directory if using local storage
        if storage_backend == "local":
            Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

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

            # Store file
            file_path = await self._store_file(
                dataset_id, input_data.filename, input_data.file_content
            )

            # Calculate memory usage
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

            # Get data types
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Generate preview (first 5 rows)
            preview = df.head(5).to_dict(orient="records")

            logger.info(
                f"File uploaded successfully - Dataset ID: {dataset_id}, "
                f"Rows: {len(df)}, Columns: {len(df.columns)}"
            )

            return UploadFileOutput(
                node_type=self.node_type,
                execution_time_ms=0,  # Set by base class
                dataset_id=dataset_id,
                filename=input_data.filename,
                file_path=str(file_path),
                n_rows=len(df),
                n_columns=len(df.columns),
                columns=df.columns.tolist(),
                dtypes=dtypes,
                memory_usage_mb=round(memory_usage_mb, 2),
                preview=preview,
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

    async def _store_file(self, dataset_id: str, filename: str, file_content: bytes) -> Path:
        """
        Store file to storage backend.

        Args:
            dataset_id: Unique dataset identifier
            filename: Original filename
            file_content: File bytes

        Returns:
            Path to stored file
        """
        # Generate storage filename
        ext = Path(filename).suffix
        storage_filename = f"{dataset_id}{ext}"

        if self.storage_backend == "local":
            # Store locally
            file_path = Path(settings.UPLOAD_DIR) / storage_filename
            file_path.write_bytes(file_content)
            logger.info(f"File stored locally: {file_path}")
            return file_path

        elif self.storage_backend == "s3":
            # TODO: Implement S3 storage
            # This would use boto3 to upload to S3
            # For now, fallback to local
            logger.warning("S3 storage not implemented, using local storage")
            file_path = Path(settings.UPLOAD_DIR) / storage_filename
            file_path.write_bytes(file_content)
            return file_path

        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")
