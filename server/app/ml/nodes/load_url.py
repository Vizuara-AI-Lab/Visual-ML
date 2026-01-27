"""
Load URL Node - Downloads datasets from URLs and stores them using UploadFileNode.
Supports CSV, JSON, and Excel formats from HTTP(S) URLs.
"""

from typing import Optional, Dict, Any
import pandas as pd
import io
import requests
from pydantic import Field, field_validator
from app.ml.nodes.base import BaseNode, NodeInput, NodeOutput
from app.ml.nodes.upload import UploadFileNode, UploadFileInput
from app.core.logging import logger


class LoadUrlInput(NodeInput):
    """Input schema for LoadUrl node."""

    url: str = Field(..., description="HTTP(S) URL to dataset")
    format: str = Field(default="csv", description="File format (csv, json, excel)")
    project_id: Optional[int] = Field(None, description="Project ID for dataset association")
    user_id: Optional[int] = Field(None, description="User ID for ownership")

    # Metadata fields (auto-populated after first execution)
    dataset_id: Optional[str] = Field(None, description="Generated dataset ID")
    filename: Optional[str] = Field(None, description="Filename from URL")
    n_rows: Optional[int] = Field(None, description="Number of rows (metadata)")
    n_columns: Optional[int] = Field(None, description="Number of columns (metadata)")
    columns: Optional[list[str]] = Field(None, description="Column names (metadata)")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v:
            raise ValueError("URL cannot be empty")

        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        # Handle GitHub URLs - convert to raw content URL
        if "github.com" in v and "/blob/" in v:
            v = v.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            logger.info(f"Converted GitHub URL to raw: {v}")

        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate file format."""
        allowed_formats = ["csv", "json", "excel"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Format must be one of: {', '.join(allowed_formats)}")
        return v.lower()


class LoadUrlOutput(NodeOutput):
    """Output schema for LoadUrl node."""

    dataset_id: str = Field(..., description="Generated dataset identifier")
    filename: str = Field(..., description="Derived filename")
    file_path: str = Field(..., description="Storage path")
    storage_backend: str = Field(..., description="Storage backend (s3 or local)")
    n_rows: int = Field(..., description="Number of rows in dataset")
    n_columns: int = Field(..., description="Number of columns in dataset")
    columns: list[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Data types of columns")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")


class LoadUrlNode(BaseNode):
    """
    Load URL Node - Downloads datasets from URLs and stores them in S3/local.

    Same workflow as upload_file but fetches from URL instead of form upload.
    """

    node_type = "load_url"

    def get_input_schema(self) -> type[NodeInput]:
        """Return input schema."""
        return LoadUrlInput

    def get_output_schema(self) -> type[NodeOutput]:
        """Return output schema."""
        return LoadUrlOutput

    async def _execute(self, input_data: LoadUrlInput) -> LoadUrlOutput:
        """
        Download from URL and use UploadFileNode to store.

        Args:
            input_data: Validated input containing URL

        Returns:
            LoadUrlOutput with dataset metadata
        """
        try:
            # If dataset_id already exists, just return metadata
            if input_data.dataset_id and not input_data.url:
                return LoadUrlOutput(
                    node_type=self.node_type,
                    execution_time_ms=0,
                    dataset_id=input_data.dataset_id,
                    filename=input_data.filename or "dataset.csv",
                    file_path=f"datasets/{input_data.dataset_id}",
                    storage_backend="local",
                    n_rows=input_data.n_rows or 0,
                    n_columns=input_data.n_columns or 0,
                    columns=input_data.columns or [],
                    dtypes={},
                    memory_usage_mb=0.0,
                )

            url = input_data.url
            file_format = input_data.format

            logger.info(f"📥 Downloading dataset from URL: {url}")

            # Download file content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            file_content = response.content

            # Extract filename from URL
            filename = url.split("/")[-1].split("?")[0] or f"dataset.{file_format}"
            if not filename.endswith((".csv", ".json", ".xlsx", ".xls")):
                filename = f"{filename}.{file_format}"

            # Parse based on format and convert to CSV bytes
            if file_format == "csv":
                df = pd.read_csv(io.BytesIO(file_content))
                csv_content = df.to_csv(index=False).encode("utf-8")
            elif file_format == "json":
                df = pd.read_json(io.BytesIO(file_content))
                csv_content = df.to_csv(index=False).encode("utf-8")
                filename = filename.replace(".json", ".csv")
            elif file_format == "excel":
                df = pd.read_excel(io.BytesIO(file_content))
                csv_content = df.to_csv(index=False).encode("utf-8")
                filename = filename.replace(".xlsx", ".csv").replace(".xls", ".csv")
            else:
                raise ValueError(f"Unsupported format: {file_format}")

            # Validate dataset
            if df.empty:
                raise ValueError("Downloaded dataset is empty")
            if len(df.columns) == 0:
                raise ValueError("Dataset has no columns")

            logger.info(f"✅ Downloaded: {filename} ({len(df)} rows × {len(df.columns)} cols)")

            # Use UploadFileNode to handle storage
            upload_node = UploadFileNode()
            upload_input = UploadFileInput(
                file_content=csv_content,
                filename=filename,
                content_type="text/csv",
                project_id=input_data.project_id,
                user_id=input_data.user_id,
            )

            upload_result = await upload_node._execute(upload_input)

            return LoadUrlOutput(
                node_type=self.node_type,
                execution_time_ms=0,
                dataset_id=upload_result.dataset_id,
                filename=upload_result.filename,
                file_path=upload_result.file_path,
                storage_backend=upload_result.storage_backend,
                n_rows=upload_result.n_rows,
                n_columns=upload_result.n_columns,
                columns=upload_result.columns,
                dtypes=upload_result.dtypes,
                memory_usage_mb=upload_result.memory_usage_mb,
            )

        except requests.RequestException as e:
            logger.error(f"❌ Failed to download from URL: {str(e)}")
            raise ValueError(f"Failed to download from URL: {str(e)}")
        except pd.errors.ParserError as e:
            logger.error(f"❌ Failed to parse file: {str(e)}")
            raise ValueError(f"Failed to parse file: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Failed to load URL dataset: {str(e)}")
            raise ValueError(f"Failed to load dataset from URL: {str(e)}")
